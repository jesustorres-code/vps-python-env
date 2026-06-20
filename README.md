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
install.sh              # end-to-end setup: clone repo, venv, deps, weights, app.py
app.py                  # Gradio web UI (not provided by the upstream repo)
scripts/
  benchmark_load.py     # measures pipeline build + checkpoint load time
  benchmark_decode.py   # sweeps VAE decode_to_pixel_chunk chunk_size
  benchmark_compile.py  # evaluates torch.compile on the generator
  setup_path.sh         # generic ~/.local/bin PATH setup (unrelated utility)
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

- **Moving model weights from the network mount to local disk** cut total
  load time from ~165-210s to **109.1s** (pipeline build 74.3s + checkpoint
  load 21.7s) — roughly 40-48% faster, checkpoint load alone 3-5x faster.
- **VAE decode `chunk_size` doesn't matter for speed.** Swept 16/24/32/48 on
  64 latent frames: all landed within ~1s of each other (58.3s-59.1s), with
  peak memory creeping up slightly (34.8GB→35.2GB) as chunk_size grew. The
  decode is compute-bound by the VAE's 3D convolutions, not by per-chunk
  sync overhead — so `app.py` keeps `chunk_size=16`, the most memory-safe
  option, since there's no speed tradeoff to make.
- **`torch.compile` on the generator is not viable as configured.** Tried
  `configure_torch_compile(backend="inductor", mode="default", dynamic=False)`.
  PyTorch warned that Torchinductor can't codegen the model's complex-number
  RoPE ops. More importantly: the model **recompiled on every block**, not
  just the first one — block 1's first step took ~80s (compiling), but
  block 2's first step took ~67s too, instead of staying fast. This points to
  a scalar (likely the sliding-window KV-cache write offset) that changes
  per block and, with `dynamic=False`, invalidates compile guards every time.
  Net effect: every block would pay ~70s of recompilation instead of the
  ~1.2-1.4s/step eager mode already gets — strictly worse. The server runs in
  eager mode.

## scripts/setup_path.sh

Unrelated generic utility, kept here on request: appends
`export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` if not already present.
Not required for the LongLive setup above.
