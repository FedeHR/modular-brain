#!/usr/bin/env bash
#
#SBATCH --job-name=dino-feats
#SBATCH --output=/nfs/data8/%u/logs/feats_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=major
# NOTE: submit with  `sbatch -p major job.sh`  (use `minor` if major is full;
# never `-p all`, and never with a `stud` account). Raise --time for the full set.
set -euo pipefail

WORK="/nfs/data8/$USER"               # adjust if your data8 dir isn't $USER
source "$WORK/modular-brain/.venv/bin/activate"
export TORCH_HOME="$WORK/torch_cache"
export HF_HOME="$WORK/hf_cache"
cd "$WORK/modular-brain"

# sanity: confirm we actually got a GPU
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --videos/--masks: set to the dirs produced by unzipping (verify with `ls`).
python -m experiments.pvsg_hierarchy.precompute_all \
    --videos "$WORK/pvsg/VidOR/videos" \
    --masks  "$WORK/pvsg/VidOR/masks" \
    --out    "$WORK/cache" \
    --model  dinov2_vitb14 \
    --device cuda \
    --limit  25          # start small; drop --limit for the full subset
    # --mask-stride 2     # ≈2.5 FPS to roughly halve time/space (Tresp-approved)
