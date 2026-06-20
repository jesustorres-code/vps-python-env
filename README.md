# VPS Python Environment Config — LongLive-2.0 on RunPod

Scripts to reproduce a working LongLive-2.0 (NVlabs, text-to-video diffusion)
installation on a RunPod GPU pod (or any single-GPU Ubuntu VPS with an NVIDIA
GPU), including a Gradio web UI, in future environments.

Official repo: https://github.com/NVlabs/LongLive

## What this captures

This was built and verified end-to-end on a RunPod pod with a single
**RTX A6000** (Ampere, 49 GB VRAM, no NVFP4/Blackwell support) — so the whole
setup runs the **BF16** inference path. If your GPU is Blackwell-class (sm_100+)
you can instead follow the NVFP4 instructions in the upstream
`docs/getting_started.md`.

## Layout

```
install.sh                          # end-to-end setup: clone repo, venv, deps, weights, app.py, fix
app.py                               # Gradio web UI (not provided by the upstream repo)
utils/wan_5b_wrapper.py              # patched copy of the upstream file (text encoder load fix)
utils/inference_utils.py             # patched copy of the upstream file (generator checkpoint load fix)
scripts/
  benchmark_load.py                  # measures pipeline build + checkpoint load time
  benchmark_decode.py                # sweeps VAE decode_to_pixel_chunk chunk_size
  benchmark_compile.py               # evaluates torch.compile (dynamic=False) on the generator
  benchmark_compile_dynamic.py       # evaluates torch.compile (dynamic=True) on the generator
  benchmark_lightvae.py              # standard Wan2.2 VAE vs mg_lightvae_v2 decode, same latents
  benchmark_resolution.py            # half-resolution denoise+decode speedup/safety check
  benchmark_build_breakdown.py       # times each CausalDiffusionInferencePipeline.__init__ sub-step
  benchmark_textencoder_breakdown.py # times WanTextEncoder's construction/load/cuda sub-steps
  benchmark_textencoder_meta.py      # validates the device='meta'+assign=True fix in isolation
  benchmark_generator_checkpoint_load.py # validates mmap+assign on the generator checkpoint load
  quickstart_test.py                 # minimal end-to-end smoke test (build->denoise->decode->save)
  setup_path.sh                      # generic ~/.local/bin PATH setup (unrelated utility)
```

## Quick start

```bash
git clone <this-repo> ~/configs/vps-python-env
cd ~/configs/vps-python-env
./install.sh
```

Defaults to cloning LongLive into `/workspace/LongLive`, the venv into
`/opt/longlive_venv`, and model weights into `/opt/longlive_models`. Override
with `REPO_DIR`, `VENV_DIR`, `MODELS_DIR` env vars if needed.

Then launch the server:

```bash
cd /workspace/LongLive
source /opt/longlive_venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u app.py
```

The app listens on `0.0.0.0:7860`. Expose it either via the RunPod dashboard
("Expose HTTP Ports" → 7860, gives you `https://<POD_ID>-7860.proxy.runpod.net`)
or with a Cloudflare quick tunnel:

```bash
cloudflared tunnel --url http://localhost:7860
```

## Why things are done this way

- **venv and model weights live under `/opt`, not `/workspace`.** On this
  RunPod pod, `/workspace` is a network-mounted filesystem (MooseFS) that
  reports hundreds of TB free in `df -h` but silently enforces a much smaller
  per-pod quota (~50GB) — installs can fail with "Disk quota exceeded" despite
  apparent free space. `/` (the container overlay) is local disk and isn't
  subject to that quota.
- **`transformers==4.49.0` pinned.** Newer transformers releases (5.x) removed
  an import (`x_clip_loss`) that `wan_5b/modules/causal_model.py` depends on.
- **`gradio<6` pinned.** Gradio 6.x requires `huggingface-hub>=1.2.0,<2.0`,
  which conflicts with the older huggingface-hub that `transformers==4.49.0`
  needs.
- **`app.py` decodes the VAE output in chunks via
  `WanVAEWrapper.decode_to_pixel_chunk(..., chunk_size=16)`, after explicitly
  clearing `pipe.kv_cache_pos/neg` and `pipe.crossattn_cache_pos/neg`.** The
  standard "wan" VAE backend has no built-in chunked/streaming decode, and the
  causal pipeline keeps ~12GB of now-unneeded KV/cross-attention cache
  resident after `inference()` returns — without freeing it first, a full
  decode at typical output lengths OOMs on a single 49GB GPU.

## Performance findings (RTX A6000, BF16)

These came out of benchmarking load and inference time on this pod;
`scripts/benchmark_*.py` reproduce each one.

### Load time: 109.1s -> 42.9s (~61% faster)

- **Moving model weights from the network mount to local disk** cut total
  load time from ~165-210s to **109.1s** (pipeline build 74.3s + checkpoint
  load 21.7s) — roughly 40-48% faster, checkpoint load alone 3-5x faster.
  This is the starting baseline for everything below.
- **`benchmark_build_breakdown.py` isolated the 74.3s "pipeline build"**
  into its three sub-components: `WanTextEncoder()` (T5) = 62.2s,
  `WanDiffusionWrapper` (generator) = 0.6s, `build_vae_5b` = 1.6s. The
  generator's `from_pretrained()` looked suspicious (loads ~20GB of fp32
  weights from `wan_models/Wan2.2-TI2V-5B/` that get fully overwritten a
  moment later by the real LongLive checkpoint) but turned out to be a
  red herring: diffusers mmaps those safetensors lazily (0.6s), and the
  alternative (`from_config`, architecture-only) is actually *slower*
  (43.0s, since it falls back to real random init without the meta-device
  trick diffusers already uses). **WanTextEncoder was the real target.**
- **`benchmark_textencoder_breakdown.py` isolated WanTextEncoder's 62.2s**
  further: `umt5_xxl()` construction = 56.5s, `load_state_dict` = 2.5s,
  `.cuda()` = 2.4s. The construction cost is CPU-bound random weight
  initialization (Xavier/normal init over ~11B parameters) for a T5
  encoder that gets **entirely discarded** a moment later when its real
  checkpoint is loaded — this is the dominant cost in the whole 109.1s
  baseline.
- **Fix 1 (small, included for completeness): `dtype=torch.float32` ->
  `torch.bfloat16`** in the `umt5_xxl(...)` call inside
  `WanTextEncoder.__init__` (`utils/wan_5b_wrapper.py`). Behavior-neutral:
  `app.py` already casts the whole pipeline to bf16 at the end of setup,
  so this only changes the dtype the encoder is built/loaded in, not the
  dtype it runs inference in. Net effect alone: 109.1s -> 101.2s.
- **Fix 2 (small): `torch.load(..., mmap=True)`** for the T5 checkpoint
  file (11.36GB `.pth`). Avoids eagerly deserializing the whole file via
  pickle. Net effect on top of fix 1: 101.2s -> 101.7s (within noise —
  the checkpoint load was never the bottleneck, the file read in isolation
  dropped from several seconds to 0.03s, but that's a small slice of the
  62s).
- **Fix 3 (the real win): construct `umt5_xxl()` with `device='meta'`,
  then `load_state_dict(..., assign=True)`.** Meta-device construction
  allocates shape/dtype metadata only — no real memory, no init compute —
  since `assign=True` replaces every meta parameter with the checkpoint's
  real tensor directly (a plain `copy_` would fail on meta storage;
  `assign=True` is the PyTorch 2.1+ mechanism for this exact pattern).
  Validated in isolation first (`benchmark_textencoder_meta.py`):
  `missing=[] unexpected=[]` and 0 meta parameters survive the assign,
  confirming full checkpoint coverage before relying on this in
  production. No T5 buffers exist outside `nn.Parameter`s in this repo's
  T5 implementation (the relative-position bias is a learned
  `nn.Embedding` per block, not a precomputed buffer), so nothing is left
  uninitialized.
  **Net effect: "pipeline build" 74.3s -> 5.6s, total load 109.1s -> 42.9s.**
- **Fix 4: the same mmap+assign pattern, one step further down the load
  path.** `CausalWanModel.from_pretrained()` (used to build the generator)
  already does meta-device+assign internally via diffusers'
  `low_cpu_mem_usage` path — that's why it measured at only 0.6s in fix 3's
  breakdown. But `load_generator_checkpoint()`
  (`utils/inference_utils.py`) was then overwriting those placeholder
  weights with a plain `strict` `load_state_dict` (`copy_` semantics),
  forcing the `from_pretrained` placeholder's mmap'd fp32 data to
  materialize just to immediately discard it. Same fix, same file pattern:
  `mmap=True` on the `torch.load()` of the checkpoint `.pt` file
  (`model_bf16.pt`, 9.4GB) in `_torch_load()`, plus `assign=True` on
  `generator.load_state_dict(...)`. Validated in isolation first
  (`benchmark_generator_checkpoint_load.py`): `torch.load` without mmap
  5.5s -> mmap=True 0.0s; `load_state_dict` with `copy_` 12.5s ->
  `assign=True` 0.0s.
  **Net effect: "Checkpoint loaded" 23.7s -> 0.1s, total load 42.9s -> 12.3s.**

| Stage           | Baseline | +bf16  | +mmap  | +meta+assign | +generator mmap+assign |
|-----------------|---------:|-------:|-------:|-------------:|------------------------:|
| Pipeline built  |   74.3s  |  70.7s |  66.0s |        5.6s  |                   7.2s* |
| Checkpoint load |   21.7s  |  17.5s |  23.5s |       23.7s* |                   0.1s  |
| **Total**       | **109.1s**| **101.2s** | **101.7s** | **42.9s** | **12.3s** |

\* run-to-run noise of a few seconds (page cache state) on checkpoint-load
timings prior to fix 4, and on pipeline-build after it; the headline
numbers (74.3s -> 5.6s, 23.7s -> 0.1s) are the stable, reproducible signal.

**Combined result: 109.1s -> 12.3s, a ~89% reduction in total load time**,
with no quality regression (verified via a live generation after each
fix) and no behavior change at inference time — every fix here only
changes *how* the same final bf16 weights get into GPU memory.

### Inference: VAE backend + resolution switch

- **VAE decode `chunk_size` doesn't matter for speed.** Swept 16/24/32/48 on
  64 latent frames: all landed within ~1s of each other (58.3s-59.1s), with
  peak memory creeping up slightly (34.8GB→35.2GB) as chunk_size grew. The
  decode is compute-bound by the VAE's 3D convolutions, not by per-chunk
  sync overhead — so `app.py` keeps `chunk_size=16` for the "Calidad" mode,
  the most memory-safe option, since there's no speed tradeoff to make.
- **`torch.compile` on the generator is not viable, in either static or
  dynamic shape mode.** `configure_torch_compile(backend="inductor",
  mode="default", dynamic=False)`: PyTorch warned Torchinductor can't
  codegen the model's complex-number RoPE ops, and worse, the model
  **recompiled on every block** (~70-80s each), not just the first —
  some per-block scalar (likely the sliding-window KV-cache write offset)
  invalidates compile guards every time under static shapes. Tried
  `dynamic=True` next, hoping it would stop the per-block recompiles:
  first call's first step took 200.6s compiling, and the *second* call's
  first step still hadn't finished compiling after 12 minutes (killed).
  Eager mode (~1.2-1.4s/step) stays strictly faster either way. The
  server runs in eager mode.
- **`mg_lightvae_v2` (Skywork/Matrix-Game-3.0) is a drop-in faster VAE
  decoder**: same latent space as the standard Wan2.2 VAE (z_dim=48,
  no retraining needed), `pruning_rate=0.75` fewer channels, plus a
  genuinely streaming step-by-step decode algorithm instead of the
  standard VAE's manual chunking. Measured on 64 latent frames: standard
  VAE 60.4s/34.8GB peak vs. mg_lightvae_v2 7.5s/31.2GB peak (~8x faster
  decode) — at a real, quantified quality cost (PSNR ~18.7dB, SSIM
  Y:0.787 against the standard VAE's output on the same latents). Exposed
  as an opt-in choice in `app.py`'s "Velocidad / calidad" switch rather
  than silently swapped in, since the quality tradeoff is real.
- **Halving output resolution is architecturally safe.** The transformer
  derives its token grid from the actual input tensor's shape at forward
  time (`_forward_inference` in `wan_5b/modules/causal_model.py`), not
  from any precomputed constant — so `frame_seq_length` can be swapped
  between calls on an already-built pipeline, no rebuild needed. Measured
  ~3.95x speedup on 64 frames (half-res denoise 13.6s vs full-res 54.2s)
  with no errors and coherent geometry, at a real sharpness cost. `app.py`
  combines this with `mg_lightvae_v2` as the "Turbo" mode: ~4x faster
  end-to-end (denoise+decode) than "Calidad", at the lowest sharpness of
  the three options.

`app.py`'s "Velocidad / calidad" radio switch keeps all backends (both
VAEs, both resolution configs) resident on GPU simultaneously, so picking
between them is instant — no reload between generations.

## scripts/setup_path.sh

Unrelated generic utility, kept here on request: appends
`export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` if not already present.
Not required for the LongLive setup above.
