#!/usr/bin/env bash
#
# Pull cluster trackio DBs + results JSONs to the Mac. Safe to re-run while a
# job is writing (reads a snapshot; never contends with the writer).
#   bash experiments/pvsg_hierarchy/cluster/fetch_results.sh <you>@madeira.dbs.ifi.lmu.de
# View the fetched runs:
#   TRACKIO_DIR=~/.cache/huggingface/trackio-cluster uv run trackio show
# (separate dir on purpose — never clobbers the local runs' project DBs)
set -euo pipefail

HOST="${1:?usage: fetch_results.sh <you>@madeira.dbs.ifi.lmu.de}"
RUSER="${HOST%%@*}"
DEST="$HOME/.cache/huggingface/trackio-cluster"

mkdir -p "$DEST"
rsync -avz "$HOST:/nfs/data8/$RUSER/trackio/" "$DEST/"
rsync -avz "$HOST:/nfs/data8/$RUSER/modular-brain/results/" results/

echo
echo "view cluster runs:  TRACKIO_DIR=$DEST uv run trackio show"
