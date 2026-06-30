#!/usr/bin/env bash
# Run ONCE on the madeira LOGIN node (it has internet; compute nodes may not).
# Sets up the uv venv + pre-fetches every network asset into shared /nfs/data8.
# Assumes the repo is already at  $WORK/modular-brain  (see CLUSTER.md, step 1).
#
#   cd /nfs/data8/<you> && bash modular-brain/experiments/pvsg_hierarchy/cluster/env_setup.sh
set -euo pipefail

WORK="$(pwd)"                       # run this from your /nfs/data8/<you> directory
echo ">> WORK=$WORK"
mkdir -p "$WORK"/{torch_cache,hf_cache,uv_cache,cache,logs}

# 1) uv — user-space, no sudo
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$WORK/uv_cache"   # keep uv's cache off the (small) home quota

# 2) venv + deps (the .venv lives on shared storage so compute nodes see it)
cd "$WORK/modular-brain"
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install imageio imageio-ffmpeg huggingface_hub
# CUDA torch: check `nvidia-smi` on a GPU node first (srun), then pick the wheel.
# cu121 suits recent drivers; use cu118 if nvidia-smi shows a CUDA 11 driver.
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) pre-fetch DINO weights into the SHARED torch cache (offline-safe at job time)
export TORCH_HOME="$WORK/torch_cache"
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',trust_repo=True); print('DINO cached')"

# 4) pre-fetch the PVSG VidOR subset (~1.8 GB) and unzip
export HF_HOME="$WORK/hf_cache"
huggingface-cli download Jingkang/PVSG --repo-type dataset --include "VidOR/*" --local-dir "$WORK/pvsg"
( cd "$WORK/pvsg/VidOR" && for z in *.zip; do unzip -n "$z"; done )

echo ">> setup complete. Inspect the unzipped layout and note the video/mask dirs:"
echo "   ls $WORK/pvsg/VidOR"
echo ">> then edit cluster/job.sh --videos/--masks paths to match, and: sbatch -p major .../job.sh"
