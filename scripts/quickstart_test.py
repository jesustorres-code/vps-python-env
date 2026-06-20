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
device = torch.device("cuda")

torch.set_grad_enabled(False)

t0 = time.time()
print("Building pipeline...")
pipe = CausalDiffusionInferencePipeline(config, device=device)
print(f"Pipeline built in {time.time()-t0:.1f}s")

t0 = time.time()
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
print(f"Checkpoint loaded in {time.time()-t0:.1f}s")

pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
print("noise shape:", noise.shape)

t0 = time.time()
latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
print(f"Denoising done in {time.time()-t0:.1f}s, latents shape={latents.shape}")

# Free the transformer's KV/cross-attn caches (~12GB) before VAE decode; they
# are not needed once we have the final latents, and the standard "wan" VAE
# decode path has no chunked/cached mode of its own to fall back on.
pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
print("Peak GPU mem before decode (GB):", torch.cuda.max_memory_allocated() / 1e9)
print("Free/total GPU mem (GB):", [x / 1e9 for x in torch.cuda.mem_get_info()])

t0 = time.time()
video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
video = (video * 0.5 + 0.5).clamp(0, 1)
print(f"Chunked VAE decode done in {time.time()-t0:.1f}s, video shape={video.shape}")

save_video(video[0], "videos/quickstart/sample.mp4", fps=24)
print("Saved videos/quickstart/sample.mp4")
print("Peak GPU mem (GB):", torch.cuda.max_memory_allocated() / 1e9)
