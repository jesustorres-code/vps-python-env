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

MERGED_CHECKPOINT_PATH = "checkpoints/LongLive-2.0-5B/model_bf16.pt"
OUTPUT_DIR = "videos/gradio"
DEVICE = torch.device("cuda")

torch.set_grad_enabled(False)

print("Loading LongLive-2.0-5B (BF16) pipeline, this takes a few minutes...")
_config = normalize_config(OmegaConf.load("configs/inference.yaml"))
pipe = CausalDiffusionInferencePipeline(_config, device=DEVICE)
load_generator_checkpoint(pipe.generator, MERGED_CHECKPOINT_PATH)
pipe = pipe.to(device=DEVICE, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, _config)
pipe.generator.model.eval().requires_grad_(False)
print("Pipeline ready.")

# Single GPU, single model instance: serialize concurrent Gradio requests.
_generation_lock = threading.Lock()


@torch.inference_mode()
def generate(prompt, num_output_frames, sampling_steps, guidance_scale, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    num_output_frames = int(num_output_frames)
    if num_output_frames % 8 != 0:
        num_output_frames = (num_output_frames // 8) * 8 or 8

    with _generation_lock:
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
        video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        print(f"[generate] VAE decode done in {time.time() - t0:.1f}s")

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
            run_btn = gr.Button("Generate video", variant="primary")
        with gr.Column():
            output_video = gr.Video(label="Output")

    run_btn.click(
        fn=generate,
        inputs=[prompt, num_output_frames, sampling_steps, guidance_scale, seed],
        outputs=output_video,
    )

if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
