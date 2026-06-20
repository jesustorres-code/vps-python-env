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
from utils.lightvae_5b_wrapper import LightVAE5BWrapper

prompt = "A compact silver robot walks through a clean robotics lab."
merged_checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
config.num_output_frames = 64
device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Building pipeline (standard wan VAE for the generator side)...")
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)
print("Pipeline ready.")

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
t0 = time.time()
latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
print(f"Denoising done in {time.time()-t0:.1f}s, latents shape={latents.shape}")

# Free the generator's KV/cross-attn caches: not needed for decode, and both
# VAE variants need headroom to run their own benchmark in isolation.
pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
latents_cpu = latents.cpu()
del latents

# --- Baseline: standard "wan" VAE, chunked decode (current app.py behavior) ---
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
lat = latents_cpu.to(device)
t0 = time.time()
video_wan = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=16)
video_wan = (video_wan * 0.5 + 0.5).clamp(0, 1)
torch.cuda.synchronize()
dt_wan = time.time() - t0
peak_wan = torch.cuda.max_memory_allocated() / 1e9
print(f"[wan]          time={dt_wan:6.1f}s  peak_mem={peak_wan:.1f}GB  shape={tuple(video_wan.shape)}")
save_video(video_wan[0], "videos/lightvae_compare/standard_wan.mp4", fps=24)
del video_wan, lat
torch.cuda.empty_cache()

# --- mg_lightvae_v2: same latent space, pruned decoder, streaming step decode ---
print("Loading mg_lightvae_v2...")
lightvae = LightVAE5BWrapper(
    vae_path="wan_models/Matrix-Game-3.0/MG-LightVAE_v2.pth",
    device=device,
    dtype=torch.bfloat16,
).eval()
print(f"mg_lightvae_v2 loaded, inferred pruning_rate={lightvae.pruning_rate}")

torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
lat = latents_cpu.to(device=device, dtype=torch.bfloat16)
t0 = time.time()
video_light = lightvae.decode_to_pixel(lat, use_cache=False)
video_light = (video_light * 0.5 + 0.5).clamp(0, 1)
torch.cuda.synchronize()
dt_light = time.time() - t0
peak_light = torch.cuda.max_memory_allocated() / 1e9
print(f"[mg_lightvae_v2] time={dt_light:6.1f}s  peak_mem={peak_light:.1f}GB  shape={tuple(video_light.shape)}")
save_video(video_light[0], "videos/lightvae_compare/mg_lightvae_v2.mp4", fps=24)
del video_light, lat
torch.cuda.empty_cache()

print(f"\nSummary: wan={dt_wan:.1f}s/{peak_wan:.1f}GB  mg_lightvae_v2={dt_light:.1f}s/{peak_light:.1f}GB")
print("Done.")
