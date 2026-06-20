import json
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from pipeline import CausalDiffusionInferencePipeline
from utils.config import normalize_config
from utils.inference_utils import (
    load_generator_checkpoint,
    place_vae_for_streaming,
    prepare_single_prompt_inputs,
    save_video,
)
from utils.lightvae_5b_wrapper import LightVAE5BWrapper

PROMPT = "A compact silver robot walks through a clean robotics lab."
OUT_DIR = "videos/lightvae_compare"
BACKENDS = [
    ("standard_wan", None),
    ("mg_lightvae_v2", "wan_models/Matrix-Game-3.0/MG-LightVAE_v2.pth"),
    ("mg_lightvae_v1", "wan_models/Matrix-Game-3.0/MG-LightVAE.pth"),
]


def to_luma_bt601(rgb):
    # rgb: (T, H, W, 3) float [0, 1]
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def main():
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

    noise, prompts = prepare_single_prompt_inputs(config, PROMPT, device)
    t0 = time.time()
    latents = pipe.inference(noise=noise, text_prompts=prompts, return_latents=True)
    print(f"Denoising done in {time.time()-t0:.1f}s, latents shape={latents.shape}")

    pipe.kv_cache_pos = None
    pipe.kv_cache_neg = None
    pipe.crossattn_cache_pos = None
    pipe.crossattn_cache_neg = None
    torch.cuda.empty_cache()
    latents_cpu = latents.cpu()
    del latents

    results = {}
    videos = {}

    for name, vae_path in BACKENDS:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if vae_path is None:
            lat = latents_cpu.to(device)
            video = pipe.vae.decode_to_pixel_chunk(lat, use_cache=False, chunk_size=16)
            pruning_rate_used = 0.0
        else:
            lat = latents_cpu.to(device=device, dtype=torch.bfloat16)
            lightvae = LightVAE5BWrapper(
                vae_path=vae_path, device=device, dtype=torch.bfloat16
            ).eval()
            pruning_rate_used = lightvae.pruning_rate
            video = lightvae.decode_to_pixel(lat, use_cache=False)
            del lightvae
        video = (video * 0.5 + 0.5).clamp(0, 1)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        results[name] = {
            "time_s": dt,
            "peak_mem_gb": peak,
            "pruning_rate": pruning_rate_used,
            "num_frames": int(video.shape[1]),
        }
        print(
            f"[{name:15s}] time={dt:6.1f}s peak_mem={peak:5.1f}GB "
            f"pruning_rate={pruning_rate_used} frames={video.shape[1]}"
        )
        save_video(video[0], f"{OUT_DIR}/{name}.mp4", fps=24)
        videos[name] = video[0].detach().cpu().float()
        del video, lat
        torch.cuda.empty_cache()

    # decode_to_pixel_chunk (standard) and the lightvae streaming decode
    # produce different total frame counts from the SAME input latents
    # (chunk-boundary truncation vs continuous decode) - not a bug, just two
    # different decode algorithms. Both start from the same t=0 latent, so
    # truncating every video to the shortest one's length keeps them aligned
    # for a frame-by-frame quality comparison.
    min_frames = min(v.shape[0] for v in videos.values())
    print(f"\nTruncating all videos to the common min frame count: {min_frames}")
    videos = {name: v[:min_frames] for name, v in videos.items()}

    baseline_np = videos["standard_wan"].permute(0, 2, 3, 1).numpy()
    baseline_luma = to_luma_bt601(baseline_np)
    quality = {}
    for name in ("mg_lightvae_v2", "mg_lightvae_v1"):
        test_np = videos[name].permute(0, 2, 3, 1).numpy()
        if test_np.shape != baseline_np.shape:
            print(
                f"WARNING: {name} shape {test_np.shape} != baseline shape "
                f"{baseline_np.shape}; skipping quality comparison."
            )
            quality[name] = None
            continue
        test_luma = to_luma_bt601(test_np)
        psnr_y = peak_signal_noise_ratio(baseline_luma, test_luma, data_range=1.0)
        # structural_similarity treats every non-channel axis as spatial, so
        # passing the full (T,H,W,3) array would do a 3D SSIM over the time
        # axis too (wrong, and extremely slow). Compute per-frame 2D SSIM and
        # average, matching ffmpeg's ssim filter convention.
        ssim_all_frames = [
            structural_similarity(baseline_np[t], test_np[t], channel_axis=-1, data_range=1.0)
            for t in range(baseline_np.shape[0])
        ]
        ssim_y_frames = [
            structural_similarity(baseline_luma[t], test_luma[t], data_range=1.0)
            for t in range(baseline_luma.shape[0])
        ]
        ssim_all = float(np.mean(ssim_all_frames))
        ssim_y = float(np.mean(ssim_y_frames))
        quality[name] = {
            "psnr_y": float(psnr_y),
            "ssim_y": float(ssim_y),
            "ssim_all": float(ssim_all),
        }

    print("\n=== Speed / memory ===")
    print(f"{'backend':15s} {'time(s)':>8s} {'peak(GB)':>9s} {'pruning':>8s} {'speedup_vs_wan':>15s}")
    wan_t = results["standard_wan"]["time_s"]
    for name, r in results.items():
        speedup = wan_t / r["time_s"] if r["time_s"] > 0 else float("inf")
        print(f"{name:15s} {r['time_s']:8.1f} {r['peak_mem_gb']:9.1f} {r['pruning_rate']:8.2f} {speedup:14.2f}x")

    print("\n=== Quality vs standard_wan baseline ===")
    print(f"{'backend':15s} {'PSNR_Y(dB)':>11s} {'SSIM_Y':>8s} {'SSIM_All':>9s}")
    for name, q in quality.items():
        if q is None:
            print(f"{name:15s} {'N/A':>11s} {'N/A':>8s} {'N/A':>9s}")
        else:
            print(f"{name:15s} {q['psnr_y']:11.2f} {q['ssim_y']:8.4f} {q['ssim_all']:9.4f}")

    summary = {"speed": results, "quality": quality}
    with open(f"{OUT_DIR}/v1_vs_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_DIR}/v1_vs_v2_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
