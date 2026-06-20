import time
import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming

merged_checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
device = torch.device("cuda")
torch.set_grad_enabled(False)

t_start = time.time()
t0 = time.time()
pipe = CausalDiffusionInferencePipeline(config, device=device)
print(f"Pipeline built in {time.time()-t0:.1f}s")

t0 = time.time()
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
print(f"Checkpoint loaded in {time.time()-t0:.1f}s")

pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)

print(f"TOTAL LOAD TIME: {time.time()-t_start:.1f}s")
