import time
import torch
from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.wan_5b_wrapper import WanDiffusionWrapper
from utils.inference_utils import _torch_load, unwrap_generator_state_dict

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
model_kwargs = dict(getattr(config, "model_kwargs", {}))
checkpoint_path = "checkpoints/LongLive-2.0-5B/model_bf16.pt"

print("=== torch.load(model_bf16.pt) without mmap (current behavior) ===")
t0 = time.time()
checkpoint = _torch_load(checkpoint_path)
print(f"_torch_load (no mmap): {time.time()-t0:.1f}s")
state_dict = unwrap_generator_state_dict(checkpoint, use_ema=False)

print("\n=== torch.load(model_bf16.pt, mmap=True) ===")
t0 = time.time()
checkpoint_mmap = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
print(f"torch.load(mmap=True): {time.time()-t0:.1f}s")

print("\n=== generator.load_state_dict, copy_ (current) vs assign=True ===")
gen1 = WanDiffusionWrapper(**model_kwargs, is_causal=True)
t0 = time.time()
gen1.load_state_dict(state_dict, strict=True)
print(f"load_state_dict (copy_, current behavior): {time.time()-t0:.1f}s")
del gen1

gen2 = WanDiffusionWrapper(**model_kwargs, is_causal=True)
t0 = time.time()
gen2.load_state_dict(state_dict, strict=True, assign=True)
print(f"load_state_dict (assign=True): {time.time()-t0:.1f}s")
del gen2

print("\nDone.")
