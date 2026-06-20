#!/bin/bash
# Reproduces the LongLive-2.0 (NVlabs) text-to-video setup on a fresh
# RunPod-style GPU instance: NVIDIA GPU (Ampere or newer), Ubuntu 22.04/24.04,
# CUDA 12.x driver already installed.
#
# Key decisions baked into this script (see README.md for the why):
#   - Python venv and model weights live under /opt, NOT /workspace, because
#     network-mounted /workspace can carry a hidden disk quota even when
#     `df -h` reports huge free space.
#   - transformers is pinned to 4.49.0 (newer releases removed an import the
#     LongLive causal_model module needs).
#   - gradio is pinned to <6 (gradio 6.x needs huggingface-hub>=1.2.0, which
#     conflicts with the huggingface-hub version transformers==4.49.0 wants).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/LongLive}"
VENV_DIR="${VENV_DIR:-/opt/longlive_venv}"
MODELS_DIR="${MODELS_DIR:-/opt/longlive_models}"

echo "==> Cloning LongLive (main, single branch, shallow)..."
if [ ! -d "$REPO_DIR" ]; then
  git clone --single-branch --branch main --depth 1 \
    https://github.com/NVlabs/LongLive.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> Creating venv at $VENV_DIR (kept off /workspace on purpose)..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "==> Installing PyTorch (CUDA 12.8 build)..."
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing repo requirements..."
pip install -r requirements.txt

echo "==> Pinning transformers/gradio to known-good versions..."
pip install "transformers==4.49.0" "gradio<6"

echo "==> Installing flash-attn (optional, speeds up attention)..."
pip install flash-attn --no-build-isolation || echo "flash-attn build failed, continuing without it"

echo "==> Downloading model weights to local disk ($MODELS_DIR)..."
mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/wan_models"
hf download Efficient-Large-Model/LongLive-2.0-5B \
  --local-dir "$MODELS_DIR/checkpoints/LongLive-2.0-5B" \
  --exclude "*.safetensors" --exclude "*fp4*"
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir "$MODELS_DIR/wan_models/Wan2.2-TI2V-5B"
hf download Skywork/Matrix-Game-3.0 MG-LightVAE_v2.pth \
  --local-dir "$MODELS_DIR/wan_models/Matrix-Game-3.0"

echo "==> Symlinking weights into the repo (keeps repo tree path-compatible)..."
rm -rf "$REPO_DIR/checkpoints" "$REPO_DIR/wan_models"
ln -s "$MODELS_DIR/checkpoints" "$REPO_DIR/checkpoints"
ln -s "$MODELS_DIR/wan_models" "$REPO_DIR/wan_models"

echo "==> Copying the Gradio web app into the repo..."
cp "$(dirname "$0")/app.py" "$REPO_DIR/app.py"
mkdir -p "$REPO_DIR/videos/gradio"

echo "==> Applying the WanTextEncoder load-time fix (bf16 + mmap + meta device)..."
cp "$(dirname "$0")/utils/wan_5b_wrapper.py" "$REPO_DIR/utils/wan_5b_wrapper.py"

echo "==> Applying the generator checkpoint load-time fix (mmap + assign)..."
cp "$(dirname "$0")/utils/inference_utils.py" "$REPO_DIR/utils/inference_utils.py"

cat <<'EOF'

==> Done. To launch the server:

  cd /workspace/LongLive
  source /opt/longlive_venv/bin/activate
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u app.py

The app listens on 0.0.0.0:7860. Expose port 7860 as an HTTP port in the
RunPod dashboard, or run a Cloudflare quick tunnel:

  cloudflared tunnel --url http://localhost:7860

EOF
