import time
import torch
from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, build_vae_5b
from wan_5b.modules.causal_model import CausalWanModel

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
model_kwargs = dict(getattr(config, "model_kwargs", {}))

print("=== Profiling CausalDiffusionInferencePipeline.__init__'s sub-steps ===")

t0 = time.time()
text_encoder = WanTextEncoder()
print(f"WanTextEncoder() total: {time.time()-t0:.1f}s")

t0 = time.time()
generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
print(f"WanDiffusionWrapper(is_causal=True) total: {time.time()-t0:.1f}s")

t0 = time.time()
vae = build_vae_5b(config)
print(f"build_vae_5b(config) total: {time.time()-t0:.1f}s")

print("\n=== Isolating the from_pretrained() weight-load cost inside WanDiffusionWrapper ===")

t0 = time.time()
model_name = model_kwargs.get("model_name", "Wan2.2-TI2V-5B")
model_fp = CausalWanModel.from_pretrained(
    f"wan_models/{model_name}/",
    local_attn_size=model_kwargs.get("local_attn_size", -1),
    sink_size=model_kwargs.get("sink_size", 0),
    num_frame_per_block=model_kwargs.get("num_frame_per_block", 1),
)
print(f"CausalWanModel.from_pretrained (reads ~20GB fp32 safetensors): {time.time()-t0:.1f}s")
del model_fp

t0 = time.time()
config_dict, unused = CausalWanModel.load_config(f"wan_models/{model_name}/", return_unused_kwargs=True)
model_cfg_only = CausalWanModel.from_config(
    config_dict,
    local_attn_size=model_kwargs.get("local_attn_size", -1),
    sink_size=model_kwargs.get("sink_size", 0),
    num_frame_per_block=model_kwargs.get("num_frame_per_block", 1),
)
print(f"CausalWanModel.from_config (architecture only, no weight file read): {time.time()-t0:.1f}s")
del model_cfg_only

print("\nDone.")
