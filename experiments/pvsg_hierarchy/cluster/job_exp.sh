#!/usr/bin/env bash
#
# Generic experiment job. First arg is the runner module (run_v1, run_v2, ...),
# the rest is passed through:
#   sbatch -p major job_exp.sh run_v1 --cache $WORK/cache --seeds 0 1 2 3 4
#   sbatch -p major job_exp.sh run_v2 --cache $WORK/cache --timelines $WORK/timelines
#
#SBATCH --job-name=tb-exp
#SBATCH --output=/nfs/data8/%u/logs/exp_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=minor
set -euo pipefail

WORK="/nfs/data8/$USER"
source "$WORK/modular-brain/.venv/bin/activate"
export TORCH_HOME="$WORK/torch_cache"
export HF_HOME="$WORK/hf_cache"
export TRACKIO_DIR="$WORK/trackio"    # sqlite DBs; fetch_results.sh pulls them home
cd "$WORK/modular-brain"

nvidia-smi

MODULE="$1"; shift
# --device defaults to auto -> CUDA on the node
python -m "experiments.pvsg_hierarchy.$MODULE" "$@"
