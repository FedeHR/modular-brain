#!/usr/bin/env bash
#
# Mask-timeline extraction — CPU only, cheap (PNG scan, no model, no GPU).
# Produces $WORK/timelines/<video_id>.pt used for: feature-cache row→frame
# alignment, V5 degradation bins, V6a occlusion/exit events.
#
#SBATCH --job-name=mask-timelines
#SBATCH --output=/nfs/data8/%u/logs/timelines_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=major
# Submit:  sbatch -p major experiments/pvsg_hierarchy/cluster/job_timelines.sh
# (use `minor` if major is full; never `-p all`, never a `stud` account).
# Resumable: finished videos are skipped, so requeueing after a timeout is safe.
set -euo pipefail

WORK="/nfs/data8/$USER"               # adjust if your data8 dir isn't $USER
source "$WORK/modular-brain/.venv/bin/activate"
cd "$WORK/modular-brain"

# The VidOR zips extract with the uploader's absolute path preserved; adjust if
# your layout differs (must be the dir whose children are <video_id>/*.png).
MASKS="$WORK/pvsg/VidOR/mnt/lustre/jkyang/CVPR23/openvsg/data/vidor/masks"
[ -d "$MASKS" ] || MASKS="$WORK/pvsg/VidOR/masks"   # fallback: flat layout
echo "masks root: $MASKS"
ls "$MASKS" | head -3

python -m experiments.pvsg_hierarchy.mask_timelines \
    --masks "$MASKS" \
    --out   "$WORK/timelines"
