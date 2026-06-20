import time
import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import (
    load_generator_checkpoint,
    place_vae_for_streaming,
    prepare_single_prompt_inputs,
)
from utils.lightvae_5b_wrapper import convert_to_channels_last_3d

prompt = "A compact silver robot walks through a clean robotics lab."
merged_checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
config.num_output_frames = 64
device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Building pipeline (standard wan VAE)...")
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)
print("Pipeline ready.")

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
latents_cpu = latents.cpu()
del latents

# --- Baseline: standard contiguous (NCDHW) decode ---
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
lat = latents_cpu.to(device)
t0 = time.time()
video_before = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=16)
torch.cuda.synchronize()
dt_before = time.time() - t0
peak_before = torch.cuda.max_memory_allocated() / 1e9
print(f"[contiguous (NCDHW)]   time={dt_before:6.1f}s peak_mem={peak_before:.1f}GB")
video_before_cpu = video_before.float().cpu()
del video_before, lat
torch.cuda.empty_cache()

# --- channels_last_3d: convert VAE weights, then run again with channels_last input ---
convert_to_channels_last_3d(pipe.vae)
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
lat = latents_cpu.to(device).to(memory_format=torch.channels_last_3d)
t0 = time.time()
video_after = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=16)
torch.cuda.synchronize()
dt_after = time.time() - t0
peak_after = torch.cuda.max_memory_allocated() / 1e9
print(f"[channels_last_3d]     time={dt_after:6.1f}s peak_mem={peak_after:.1f}GB")

# Sanity check: outputs should match closely (memory format is purely a perf
# optimization, must not change numerical results beyond float rounding).
video_after_cpu = video_after.float().cpu()
if video_after_cpu.shape == video_before_cpu.shape:
    max_diff = (video_after_cpu - video_before_cpu).abs().max().item()
    print(f"max abs diff vs contiguous baseline: {max_diff:.6f}")
else:
    print(f"WARNING: shape mismatch {video_after_cpu.shape} vs {video_before_cpu.shape}")

print(f"\nSpeedup: {dt_before/dt_after:.2f}x")
print("Done.")
