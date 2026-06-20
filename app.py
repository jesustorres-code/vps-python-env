import math
import threading
import time
import uuid

import gradio as gr
import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import (
    load_generator_checkpoint,
    place_vae_for_streaming,
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
        save_video(video[0], output_path, fps=24)
        return output_path


with gr.Blocks(title="LongLive-2.0 (RunPod / BF16)") as demo:
    gr.Markdown(
        "# LongLive-2.0-5B — Text-to-Video (BF16)\n"
        "Running on a single RTX A6000 (Ampere, BF16 inference path)."
    )
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

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
