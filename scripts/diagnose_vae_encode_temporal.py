import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as TF

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import load_generator_checkpoint, place_vae_for_streaming

device = torch.device("cuda")
torch.set_grad_enabled(False)

config = normalize_config(OmegaConf.load("configs/inference.yaml"))
pipe = CausalDiffusionInferencePipeline(config, device=device)
load_generator_checkpoint(pipe.generator, "checkpoints/LongLive-2.0-5B/model_bf16.pt")
pipe = pipe.to(device=device, dtype=torch.bfloat16)
place_vae_for_streaming(pipe, config)
print("Pipeline ready.\n")

image = Image.open("/tmp/longlive_test_images/reference_frame.png").convert("RGB").resize((1280, 704))
pixel_1 = TF.to_tensor(image).sub_(0.5).div_(0.5)

for num_pixel_frames in [1, 4, 8, 9, 13, 16, 17]:
    pixel = pixel_1.unsqueeze(0).unsqueeze(2).repeat(1, 1, num_pixel_frames, 1, 1)
    pixel = pixel.to(device=device, dtype=torch.bfloat16)
    latent = pipe.vae.encode_to_latent(pixel)
    print(f"pixel_frames={num_pixel_frames:3d} -> latent shape={tuple(latent.shape)} (latent T={latent.shape[1]})")

print("Done.")
