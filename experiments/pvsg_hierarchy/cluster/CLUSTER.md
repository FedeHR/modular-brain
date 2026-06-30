# Precomputing DINO features on the LMU DBS SLURM cluster (uv)

End-to-end walkthrough for the `madeira` cluster. Produces a per-video feature
cache that drops into `run_h1.py` (replacing the stand-in features).

## Mental model — three places, different rules
- **`madeira` login node**: has internet; **never run compute here** (cluster rule).
  Use it only for setup, downloads, `git`, and `sbatch`/`srun`.
- **GPU compute nodes** (`-p major` / `-p minor`, **never `-p all`**, never on a
  `stud` account): where work runs. **Assume no internet** → everything is
  pre-fetched in setup.
- **`/nfs/data8/<you>/`**: shared big storage. The venv, caches, data, and outputs
  all live here so both nodes see them. Below, `WORK=/nfs/data8/<you>`.

## 0. Get the code onto the cluster (from your Mac)
The feature work lives on the `stage-b-pvsg-wordnet` branch. Either `git push` it
and `git clone` on the cluster, or rsync the working tree:

```sh
# on your Mac, from the worktree root:
rsync -avz --exclude .venv --exclude '__pycache__' --exclude '*.pt' \
  ./ <you>@madeira.dbs.ifi.lmu.de:/nfs/data8/<you>/modular-brain/
```
(`pvsg_categories.json` / `pvsg_instances.json` are small and come along; the big
PVSG videos/masks are downloaded *on the cluster* in the next step.)

## 1. One-time setup (login node)
```sh
ssh <you>@madeira.dbs.ifi.lmu.de
cd /nfs/data8/<you>
bash modular-brain/experiments/pvsg_hierarchy/cluster/env_setup.sh
```
This installs uv (user-space), creates `.venv` on shared storage, installs the
CUDA torch wheel, and **pre-fetches the DINO weights + the VidOR subset** into
`$WORK/torch_cache` and `$WORK/pvsg`. Check the CUDA wheel matches the driver
(`cu121` default; see step 2 for `nvidia-smi`).

After it finishes, look at how the zips unpacked and fix the paths in `job.sh`:
```sh
ls /nfs/data8/<you>/pvsg/VidOR        # find the videos dir and the masks dir
```

## 2. Debug interactively before batching (this is what `srun` is for)
```sh
srun -p major --gres=gpu:1 --time=00:20:00 --pty bash
source /nfs/data8/<you>/modular-brain/.venv/bin/activate
export TORCH_HOME=/nfs/data8/<you>/torch_cache
nvidia-smi                                    # note the CUDA version; reinstall torch if mismatched
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# smoke-test the loop with no model first, then for real on a couple of videos:
cd /nfs/data8/<you>/modular-brain
python -m experiments.pvsg_hierarchy.precompute_all \
  --videos $WORK/pvsg/VidOR/videos --masks $WORK/pvsg/VidOR/masks \
  --out $WORK/cache --device cuda --model dinov2_vitb14 --limit 2
exit                                          # release the interactive GPU
```
You should see one `*.pt` per video appear in `$WORK/cache`.

## 3. Submit the batch job
```sh
sbatch -p major /nfs/data8/<you>/modular-brain/experiments/pvsg_hierarchy/cluster/job.sh
squeue -u $USER                               # watch it
tail -f /nfs/data8/<you>/logs/feats_*.txt     # live progress (one line per video)
```
`job.sh` is **resumable**: each video writes its own `cache/<id>.pt` and finished
ones are skipped, so if the job hits the `--time` limit just resubmit. Start with
`--limit 25`; drop it for the full VidOR subset and raise `--time`.

## 4. Bring the features home
```sh
# on your Mac:
rsync -avz <you>@madeira.dbs.ifi.lmu.de:/nfs/data8/<you>/cache/ ./cache/
```
Only the features come back (~MBs–2 GB). Leave the videos on the cluster.

## 5. Use them (replaces the H1 stand-in features)
Each `cache/<vid>.pt` holds `{object_id: tensor[n_frames, D] fp16}`. Pool to one
vector per instance with `features.aggregate_instance` (or
`precompute_features.instance_features`), key by `(video_id, object_id)` to match
`pvsg_data.ObjectInstance`, and feed those into `run_h1.py` instead of
`fixed_features` (set the TB `dim` to `D`).

## Upgrading to DINOv3 (later)
DINOv2 ViT-B is the frictionless default. For DINOv3 (better dense features):
accept its license + `huggingface-cli login` on the login node, pre-fetch the
weights into `$HF_HOME`, and add a DINOv3 branch to `DinoExtractor` (HF/torch.hub
loader, patch 16). Everything else is unchanged.

## Gotcha checklist
- [ ] venv + caches + data + outputs all under `/nfs/data8` (not home).
- [ ] DINO weights pre-fetched on the login node (`TORCH_HOME` shared) — jobs run offline.
- [ ] torch CUDA wheel matches `nvidia-smi` (cu121 vs cu118).
- [ ] `-p major`/`-p minor`, never `-p all`; never run on the login node.
- [ ] `--time` large enough (or rely on resume); outputs to `$WORK/logs`.
