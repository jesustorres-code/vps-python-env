import torch
from omegaconf import OmegaConf

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming
from utils.lightvae_5b_wrapper import LightVAE5BWrapper

device = torch.device("cuda")
torch.set_grad_enabled(False)

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, "checkpoints/LongLive-2.0-5B/model_bf16.pt")
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)

lightvae = LightVAE5BWrapper(
    vae_path="wan_models/Matrix-Game-3.0/MG-LightVAE_v2.pth", device=device, dtype=torch.bfloat16
).eval()
print("Ready.\n")

latent_h, latent_w = 44, 80

print("=== Standard VAE (decode_to_pixel_chunk, chunk_size=16) ===")
for latent_t in [8, 16, 24, 32, 40, 48, 64, 96]:
    latents = torch.randn(1, latent_t, 48, latent_h, latent_w, device=device, dtype=torch.bfloat16)
    video = pipe.vae.decode_to_pixel_chunk(latents, use_cache=False, chunk_size=16)
    pixel_t = video.shape[1]
    print(f"latent_T={latent_t:3d} -> pixel_T={pixel_t:4d}  (ratio={pixel_t/latent_t:.4f})")
    del latents, video
    torch.cuda.empty_cache()

print("\n=== LightVAE (decode_to_pixel, no chunking) ===")
for latent_t in [8, 16, 24, 32, 40, 48, 64, 96]:
    latents = torch.randn(1, latent_t, 48, latent_h, latent_w, device=device, dtype=torch.bfloat16)
    video = lightvae.decode_to_pixel(latents, use_cache=False)
    pixel_t = video.shape[1]
    print(f"latent_T={latent_t:3d} -> pixel_T={pixel_t:4d}  (ratio={pixel_t/latent_t:.4f})")
    del latents, video
    torch.cuda.empty_cache()

print("\nDone.")
