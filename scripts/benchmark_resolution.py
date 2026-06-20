import time
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

prompt = "A compact silver robot walks through a clean robotics lab."
merged_checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
config.num_output_frames = 64

# Half resolution: latent 22x40 -> pixel 352x640 (vs default latent 44x80 ->
# pixel 704x1280). Both dims stay divisible by 2 (patch_embedding stride) and
# by 16 (VAE spatial compression factor), so this should be architecturally
# valid without any model code changes.
config.image_or_video_shape = [1, config.image_or_video_shape[1], 48, 22, 40]

device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Building pipeline at HALF resolution (latent 22x40 / pixel 352x640)...")
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)
print(f"Pipeline ready. frame_seq_length={pipe.frame_seq_length}")

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
print(f"noise shape: {tuple(noise.shape)}")

t0 = time.time()
latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
torch.cuda.synchronize()
dt_denoise = time.time() - t0
print(f"Denoising done in {dt_denoise:.1f}s, latents shape={tuple(latents.shape)}")

pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
video = (video * 0.5 + 0.5).clamp(0, 1)
torch.cuda.synchronize()
dt_decode = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Decode done in {dt_decode:.1f}s, peak_mem={peak:.1f}GB, shape={tuple(video.shape)}")

save_video(video[0], "videos/lightvae_compare/half_resolution.mp4", fps=24)

print(f"\nSummary (half-res, 64 frames): denoise={dt_denoise:.1f}s decode={dt_decode:.1f}s total={dt_denoise+dt_decode:.1f}s")
print("Reference (full-res, same 64 frames): denoise=54.2s decode=60.4s total=114.6s")
print("Done.")
