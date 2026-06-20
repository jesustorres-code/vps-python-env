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
config.num_output_frames = 32
device = torch.device("cuda")
torch.set_grad_enabled(False)

print("Building pipeline...")
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, merged_checkpoint_path)
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
pipe.generator.model.eval().requires_grad_(False)
print("Pipeline ready.")

ok = pipe.generator.configure_torch_compile(
    backend="inductor",
    mode="default",
    fullgraph=False,
    dynamic=False,
    suppress_errors=True,
)
print(f"configure_torch_compile returned: {ok}")


def run_once(label, frames):
    config.num_output_frames = frames
    noise, prompts = prepare_single_prompt_inputs(config, prompt, device)
    t0 = time.time()
    latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[{label}] frames={frames} denoising time={dt:.1f}s latents_shape={tuple(latents.shape)}")
    return dt


t1 = run_once("call 1 (cold, includes compile)", 32)
t2 = run_once("call 2 (warm, same shape)", 32)
t3 = run_once("call 3 (different frame count: 64)", 64)
t4 = run_once("call 4 (back to 32 again)", 32)

print(f"\nSummary: call1={t1:.1f}s call2={t2:.1f}s call3(64f)={t3:.1f}s call4={t4:.1f}s")
print("Done.")
