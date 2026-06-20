import math
import threading
import time
import uuid

import gradio as gr
import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import (
    load_generator_checkpoint,
    place_vae_for_streaming,
    prepare_multi_scene_inputs,
    prepare_single_prompt_inputs,
    save_video,
)
from utils.lightvae_5b_wrapper import LightVAE5BWrapper

MERGED_CHECKPOINT_PATH = "checkpoints/LongLive-2.0-5B/model_bf16.pt"
LIGHTVAE_CHECKPOINT_PATH = "wan_models/Matrix-Game-3.0/MG-LightVAE_v2.pth"
OUTPUT_DIR = "videos/gradio"
DEVICE = torch.device("cuda")

QUALITY = "Calidad (resolución completa, ~60s de decode)"
FAST = "Rápido (resolución completa + VAE podado, ~8x más rápido en decode)"
TURBO = "Turbo (media resolución + VAE podado, ~4x denoise y decode, menor nitidez)"

torch.set_grad_enabled(False)

print("Loading LongLive-2.0-5B (BF16) pipeline, this takes a few minutes...")
_config = normalize_config(OmegaConf.load("configs/inference.yaml"))
pipe = CausalDiffusionInferencePipeline(_config, device=DEVICE)
load_generator_checkpoint(pipe.generator, MERGED_CHECKPOINT_PATH)
pipe = pipe.to(device=DEVICE, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, _config)
pipe.generator.model.eval().requires_grad_(False)
print("Pipeline ready.")

# Resolution is only fixed by `args.image_or_video_shape` at pipeline
# construction time via `frame_seq_length` (math.prod(shape[-2:]) // 4); the
# transformer itself derives its token grid from the actual input tensor at
# call time, so switching resolution per request just means swapping these
# two values back and forth before calling pipe.inference().
FULL_SHAPE = list(_config.image_or_video_shape)
HALF_SHAPE = FULL_SHAPE[:3] + [FULL_SHAPE[3] // 2, FULL_SHAPE[4] // 2]
FULL_FRAME_SEQ_LEN = math.prod(FULL_SHAPE[-2:]) // 4
HALF_FRAME_SEQ_LEN = math.prod(HALF_SHAPE[-2:]) // 4

FPS = 24
FRAMES_PER_BLOCK = int(_config.num_frame_per_block)
SCENE_CUT_PREFIX = pipe.scene_cut_prefix

LANDSCAPE = "16:9 horizontal (704×1280)"
PORTRAIT = "9:16 vertical (1280×704)"
PORTRAIT_SHAPE = FULL_SHAPE[:3] + [FULL_SHAPE[4], FULL_SHAPE[3]]

print("Loading mg_lightvae_v2 (fast decode mode)...")
lightvae = LightVAE5BWrapper(
    vae_path=LIGHTVAE_CHECKPOINT_PATH, device=DEVICE, dtype=torch.bfloat16
).eval()
print(f"mg_lightvae_v2 ready (pruning_rate={lightvae.pruning_rate}).")

# Single GPU, single model instance: serialize concurrent Gradio requests.
_generation_lock = threading.Lock()


@torch.inference_mode()
def generate(prompt, num_output_frames, sampling_steps, guidance_scale, seed, decode_mode):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    num_output_frames = int(num_output_frames)
    if num_output_frames % 8 != 0:
        num_output_frames = (num_output_frames // 8) * 8 or 8

    with _generation_lock:
        if decode_mode == TURBO:
            _config.image_or_video_shape = HALF_SHAPE
            pipe.frame_seq_length = HALF_FRAME_SEQ_LEN
        else:
            _config.image_or_video_shape = FULL_SHAPE
            pipe.frame_seq_length = FULL_FRAME_SEQ_LEN

        _config.num_output_frames = num_output_frames
        pipe.sampling_steps = int(sampling_steps)
        pipe.guidance_scale = float(guidance_scale)

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        noise, prompts = prepare_single_prompt_inputs(
            _config, prompt.strip(), DEVICE, generator=generator
        )

        t0 = time.time()
        latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
        print(f"[generate] denoising done in {time.time() - t0:.1f}s")

        # Free the transformer's KV/cross-attn caches before VAE decode: the
        # standard "wan" VAE has no chunked/cached decode mode of its own, so
        # without this a full-length decode OOMs on a single 49GB GPU.
        pipe.kv_cache_pos = None
        pipe.kv_cache_neg = None
        pipe.crossattn_cache_pos = None
        pipe.crossattn_cache_neg = None
        torch.cuda.empty_cache()

        t0 = time.time()
        if decode_mode in (FAST, TURBO):
            video = lightvae.decode_to_pixel(latents, use_cache=False)
        else:
            video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print(f"[generate] VAE decode ({decode_mode}) done in {time.time() - t0:.1f}s")

        output_path = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp4"
        save_video(video[0], output_path, fps=FPS)
        return output_path


def _parse_scenes(raw_text: str) -> list[tuple[str, int]]:
    """Parse 'segundos :: prompt' lines into (prompt, frames) tuples."""
    scenes = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if "::" not in line:
            raise gr.Error(f"Línea {line_no}: falta '::' para separar duración y prompt.")
        duration_str, prompt_text = line.split("::", 1)
        try:
            seconds = float(duration_str.strip())
        except ValueError:
            raise gr.Error(f"Línea {line_no}: duración inválida '{duration_str.strip()}'.")
        if seconds <= 0:
            raise gr.Error(f"Línea {line_no}: la duración debe ser mayor a 0.")
        prompt_text = prompt_text.strip()
        if not prompt_text:
            raise gr.Error(f"Línea {line_no}: falta el texto del prompt.")
        frames = max(FRAMES_PER_BLOCK, round(seconds * FPS / FRAMES_PER_BLOCK) * FRAMES_PER_BLOCK)
        scenes.append((prompt_text, frames))
    if not scenes:
        raise gr.Error("Agrega al menos una escena (formato: 'segundos :: prompt').")
    return scenes


def _build_block_prompts(scenes: list[tuple[str, int]]) -> list[str]:
    """Expand (prompt, frames) scenes into one prompt per block, prefixing scene cuts."""
    block_prompts = []
    for scene_idx, (text, frames) in enumerate(scenes):
        num_blocks = frames // FRAMES_PER_BLOCK
        for b in range(num_blocks):
            if scene_idx > 0 and b == 0:
                block_prompts.append(SCENE_CUT_PREFIX + text)
            else:
                block_prompts.append(text)
    return block_prompts


@torch.inference_mode()
def generate_multiscene(scenes_text, sampling_steps, guidance_scale, seed, decode_mode):
    scenes = _parse_scenes(scenes_text)
    block_prompts = _build_block_prompts(scenes)
    total_frames = sum(f for _, f in scenes)
    cut_blocks = [i for i, p in enumerate(block_prompts) if p.startswith(SCENE_CUT_PREFIX)]
    print(
        f"[generate_multiscene] {len(scenes)} escena(s), {len(block_prompts)} bloques "
        f"({total_frames} frames), cortes de escena en bloques: {cut_blocks}"
    )

    with _generation_lock:
        if decode_mode == TURBO:
            _config.image_or_video_shape = HALF_SHAPE
            pipe.frame_seq_length = HALF_FRAME_SEQ_LEN
        else:
            _config.image_or_video_shape = FULL_SHAPE
            pipe.frame_seq_length = FULL_FRAME_SEQ_LEN

        _config.num_output_frames = total_frames
        pipe.sampling_steps = int(sampling_steps)
        pipe.guidance_scale = float(guidance_scale)

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        noise, prompts = prepare_multi_scene_inputs(
            _config, block_prompts, DEVICE, generator=generator
        )

        t0 = time.time()
        latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
        print(f"[generate_multiscene] denoising done in {time.time() - t0:.1f}s")

        pipe.kv_cache_pos = None
        pipe.kv_cache_neg = None
        pipe.crossattn_cache_pos = None
        pipe.crossattn_cache_neg = None
        torch.cuda.empty_cache()

        t0 = time.time()
        if decode_mode in (FAST, TURBO):
            video = lightvae.decode_to_pixel(latents, use_cache=False)
        else:
            video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print(f"[generate_multiscene] VAE decode ({decode_mode}) done in {time.time() - t0:.1f}s")

        output_path = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp4"
        save_video(video[0], output_path, fps=FPS)
        return output_path


def _shape_for(aspect_ratio: str, decode_mode: str) -> tuple[list[int], int]:
    """Resolve image_or_video_shape + frame_seq_length for an aspect ratio / decode mode pair."""
    base = PORTRAIT_SHAPE if aspect_ratio == PORTRAIT else FULL_SHAPE
    if decode_mode == TURBO:
        shape = base[:3] + [base[3] // 2, base[4] // 2]
    else:
        shape = base
    return shape, math.prod(shape[-2:]) // 4


# The VAE's temporal encoder compresses 4 pixel frames per latent frame
# (causal conv, first latent frame special-cased): empirically confirmed
# latent_T = (pixel_T - 1) // 4 + 1 (see diagnose_vae_encode_temporal.py).
# To get exactly FRAMES_PER_BLOCK (8) latent frames - required by the
# num_input_frames % num_frame_per_block == 0 assert below, since our config
# doesn't set independent_first_frame=True - repeat the still image across
# 32 pixel frames (any value in [29,32] maps to latent_T=8; 32 is the round one).
REFERENCE_IMAGE_PIXEL_FRAMES = 32


def _encode_reference_image(image, shape: list[int]) -> torch.Tensor:
    """Encode a PIL reference image into a initial_latent matching `shape` (one frozen block)."""
    latent_h, latent_w = shape[3], shape[4]
    px_h, px_w = latent_h * 16, latent_w * 16  # vae_stride = 16 in both H and W
    image = image.convert("RGB").resize((px_w, px_h))
    pixel = TF.to_tensor(image).sub_(0.5).div_(0.5)  # [0,1] -> [-1,1]
    pixel = pixel.unsqueeze(0).unsqueeze(2).repeat(1, 1, REFERENCE_IMAGE_PIXEL_FRAMES, 1, 1)
    pixel = pixel.to(device=DEVICE, dtype=torch.bfloat16)
    # encode_to_latent always returns float32 (internal .float() cast); the
    # generator's conv weights are bf16, so this must be cast back to match.
    return pipe.vae.encode_to_latent(pixel).to(dtype=torch.bfloat16)


@torch.inference_mode()
def generate_image_conditioned(
    prompt, image, aspect_ratio, num_output_frames,
    sampling_steps, guidance_scale, seed, decode_mode,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    num_output_frames = int(num_output_frames)
    if num_output_frames % FRAMES_PER_BLOCK != 0:
        num_output_frames = (num_output_frames // FRAMES_PER_BLOCK) * FRAMES_PER_BLOCK or FRAMES_PER_BLOCK

    with _generation_lock:
        shape, frame_seq_len = _shape_for(aspect_ratio, decode_mode)
        _config.image_or_video_shape = shape
        pipe.frame_seq_length = frame_seq_len
        _config.num_output_frames = num_output_frames
        pipe.sampling_steps = int(sampling_steps)
        pipe.guidance_scale = float(guidance_scale)

        initial_latent = _encode_reference_image(image, shape) if image is not None else None
        print(
            f"[generate_image_conditioned] aspect={aspect_ratio} decode_mode={decode_mode} "
            f"imagen_referencia={'sí' if image is not None else 'no'}"
        )

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
        noise, prompts = prepare_single_prompt_inputs(
            _config, prompt.strip(), DEVICE, generator=generator
        )

        t0 = time.time()
        latents = pipe.inference(
            noise=noise, text_prompts=prompts, initial_latent=initial_latent, return_latents=True
        )
        print(f"[generate_image_conditioned] denoising done in {time.time() - t0:.1f}s")

        pipe.kv_cache_pos = None
        pipe.kv_cache_neg = None
        pipe.crossattn_cache_pos = None
        pipe.crossattn_cache_neg = None
        torch.cuda.empty_cache()

        t0 = time.time()
        if decode_mode in (FAST, TURBO):
            video = lightvae.decode_to_pixel(latents, use_cache=False)
        else:
            video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print(f"[generate_image_conditioned] VAE decode ({decode_mode}) done in {time.time() - t0:.1f}s")

        output_path = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp4"
        save_video(video[0], output_path, fps=FPS)
        return output_path


with gr.Blocks(title="LongLive-2.0 (RunPod / BF16)") as demo:
    gr.Markdown(
        "# LongLive-2.0-5B — Text-to-Video (BF16)\n"
        "Running on a single RTX A6000 (Ampere, BF16 inference path)."
    )
    with gr.Tabs():
        with gr.Tab("Generación simple"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="A compact silver robot walks through a clean robotics lab.",
                        lines=3,
                    )
                    num_output_frames = gr.Slider(
                        label="Latent frames (duration)", minimum=32, maximum=128, step=8, value=96
                    )
                    sampling_steps = gr.Slider(
                        label="Sampling steps", minimum=1, maximum=8, step=1, value=4
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0
                    )
                    seed = gr.Number(label="Seed", value=0, precision=0)
                    decode_mode = gr.Radio(
                        label="Velocidad / calidad",
                        choices=[QUALITY, FAST, TURBO],
                        value=QUALITY,
                        info=(
                            "Calidad: máxima nitidez, más lento. "
                            "Rápido: mismo detalle de imagen, decode ~8x más rápido. "
                            "Turbo: media resolución + decode rápido, ~4x más rápido en total, menor nitidez."
                        ),
                    )
                    run_btn = gr.Button("Generate video", variant="primary")
                with gr.Column():
                    output_video = gr.Video(label="Output")

            run_btn.click(
                fn=generate,
                inputs=[prompt, num_output_frames, sampling_steps, guidance_scale, seed, decode_mode],
                outputs=output_video,
            )

        with gr.Tab("Multi-escena (beta)"):
            with gr.Row():
                with gr.Column():
                    scenes_input = gr.Textbox(
                        label="Escenas",
                        value="4 :: A compact silver robot walks through a clean robotics lab.",
                        lines=6,
                        info=(
                            "Una escena por línea: 'segundos :: prompt'. "
                            "La transición entre escenas se maneja automáticamente "
                            "(no hace falta escribir nada especial)."
                        ),
                    )
                    ms_sampling_steps = gr.Slider(
                        label="Sampling steps", minimum=1, maximum=8, step=1, value=4
                    )
                    ms_guidance_scale = gr.Slider(
                        label="Guidance scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0
                    )
                    ms_seed = gr.Number(label="Seed", value=0, precision=0)
                    ms_decode_mode = gr.Radio(
                        label="Velocidad / calidad",
                        choices=[QUALITY, FAST, TURBO],
                        value=QUALITY,
                    )
                    ms_run_btn = gr.Button("Generate multi-scene video", variant="primary")
                with gr.Column():
                    ms_output_video = gr.Video(label="Output")

            ms_run_btn.click(
                fn=generate_multiscene,
                inputs=[scenes_input, ms_sampling_steps, ms_guidance_scale, ms_seed, ms_decode_mode],
                outputs=ms_output_video,
            )

        with gr.Tab("Imagen + formato (beta)"):
            with gr.Row():
                with gr.Column():
                    if_prompt = gr.Textbox(
                        label="Prompt",
                        value="A compact silver robot walks through a clean robotics lab.",
                        lines=3,
                    )
                    if_image = gr.Image(
                        label=(
                            "Imagen de referencia (opcional) — si la subes, el video empieza "
                            "desde esta imagen; si la dejas vacía, es texto-a-video normal"
                        ),
                        type="pil",
                    )
                    if_aspect_ratio = gr.Radio(
                        label="Formato", choices=[LANDSCAPE, PORTRAIT], value=LANDSCAPE
                    )
                    if_num_output_frames = gr.Slider(
                        label="Frames a generar", minimum=32, maximum=128, step=8, value=96,
                        info=(
                            "Si subes imagen de referencia, el video final tendrá "
                            "1 frame más (el de referencia) al inicio."
                        ),
                    )
                    if_sampling_steps = gr.Slider(
                        label="Sampling steps", minimum=1, maximum=8, step=1, value=4
                    )
                    if_guidance_scale = gr.Slider(
                        label="Guidance scale", minimum=0.5, maximum=3.0, step=0.1, value=1.0
                    )
                    if_seed = gr.Number(label="Seed", value=0, precision=0)
                    if_decode_mode = gr.Radio(
                        label="Velocidad / calidad",
                        choices=[QUALITY, FAST, TURBO],
                        value=QUALITY,
                    )
                    if_run_btn = gr.Button("Generate video", variant="primary")
                with gr.Column():
                    if_output_video = gr.Video(label="Output")

            if_run_btn.click(
                fn=generate_image_conditioned,
                inputs=[
                    if_prompt, if_image, if_aspect_ratio, if_num_output_frames,
                    if_sampling_steps, if_guidance_scale, if_seed, if_decode_mode,
                ],
                outputs=if_output_video,
            )

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
