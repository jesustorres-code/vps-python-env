import sys
import time
import torch
from omegaconf import OmegaConf

# Toggle via argv so each run is a fresh process (cudnn's autotune cache is
# per-process and shape-keyed; mixing modes in one process would muddy it).
BENCHMARK_MODE = sys.argv[1] if len(sys.argv) > 1 else "off"
torch.backends.cudnn.benchmark = (BENCHMARK_MODE == "on")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
config.num_output_frames = 32
device = torch.device("cuda")
torch.set_grad_enabled(False)

print(f"=== cudnn.benchmark={torch.backends.cudnn.benchmark} ===")
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)

noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
pipe.kv_cache_pos = None
pipe.kv_cache_neg = None
pipe.crossattn_cache_pos = None
pipe.crossattn_cache_neg = None
torch.cuda.empty_cache()
latents_cpu = latents.cpu()
del latents

# Same shape, 3 calls back to back: call 1 pays any autotune cost under
# benchmark=True; calls 2-3 show the steady-state speed either mode settles on.
for i in range(3):
    lat = latents_cpu.to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    video = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=16)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"decode call {i+1}: {dt:.2f}s")
    del video, lat
    torch.cuda.empty_cache()

print("Done.")
