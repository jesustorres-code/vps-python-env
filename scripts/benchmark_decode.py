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

prompt = "A compact silver robot walks through a clean robotics lab."
merged_checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
config.num_output_frames = 64
device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Building pipeline...")
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

pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"Free/total GPU mem before decode sweep (GB): {free/1e9:.1f} / {total/1e9:.1f}")

# Keep a CPU copy of latents so each chunk_size trial starts from the same
# clean GPU memory state (decode_to_pixel_chunk mutates nothing on `latents`
# itself, but resetting peak stats per-trial requires an isolated baseline).
latents_cpu = latents.cpu()
del latents

for chunk_size in (16, 24, 32, 48):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lat = latents_cpu.to(device)
    t0 = time.time()
    try:
        video = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=chunk_size)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"chunk_size={chunk_size:3d}  time={dt:6.1f}s  peak_mem={peak:.1f}GB  video_shape={tuple(video.shape)}")
        del video
    except torch.OutOfMemoryError as e:
        print(f"chunk_size={chunk_size:3d}  OOM: {e}")
    del lat
    torch.cuda.empty_cache()

print("Done.")
