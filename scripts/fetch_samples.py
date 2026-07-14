"""Pull a stratified EgoDex sample into DATA_ROOT/egodex/test.

Source: Wehere/EgoDex (HF dataset), one .tar PER TASK (~4.5GB each, all episodes
of that task). We can't partial-extract from HF, so per task we: download the tar,
extract only K episode pairs (hdf5 + mp4), delete the tar. Disk stays bounded
(one tar + kept episodes); bandwidth is ~4.5GB per task (run this on the AWS box).

  118 tasks x 5 episodes ~= 590 clips, ~530GB downloaded, tens of GB kept.
  Use --tasks / --k to scope. Honors FETCH_DISK_BUDGET_GB.

Usage:
  python scripts/fetch_samples.py --k 5                 # all tasks x 5 episodes
  python scripts/fetch_samples.py --k 5 --tasks 30      # first 30 tasks x 5
  python scripts/fetch_samples.py --k 5 --only sort_beads,assemble_jenga
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

REPO = os.environ.get("EGODEX_HF_REPO", "Wehere/EgoDex")


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def episode_pairs(tf: tarfile.TarFile):
    """Group tar members into {stem: {'hdf5': member, 'mp4': member}} (complete pairs only)."""
    by = defaultdict(dict)
    for m in tf.getmembers():
        if not m.isfile():
            continue
        p = Path(m.name)
        if p.suffix == ".hdf5":
            by[p.stem]["hdf5"] = m
        elif p.suffix == ".mp4":
            by[p.stem]["mp4"] = m
    return {s: d for s, d in by.items() if "hdf5" in d and "mp4" in d}


def fetch_task(task: str, k: int, dest: Path, tmp: Path, budget: float) -> int:
    from huggingface_hub import hf_hub_download

    out_dir = dest / task
    have = len(list(out_dir.glob("*.hdf5"))) if out_dir.exists() else 0
    if have >= k:
        print(f"  [{task}] have {have} >= {k}, skip")
        return have
    if free_gb(tmp) < budget:
        print(f"  ! disk under budget ({free_gb(tmp):.0f}GB), stop"); return -1

    tar_path = tmp / f"{task}.tar"
    print(f"  [{task}] downloading tar ...")
    hf_hub_download(REPO, f"{task}.tar", repo_type="dataset", local_dir=str(tmp),
                    token=os.environ.get("HF_TOKEN"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        pairs = episode_pairs(tf)
        for stem in sorted(pairs)[:k]:
            for key in ("hdf5", "mp4"):
                m = pairs[stem][key]
                m.name = f"{stem}.{key if key == 'mp4' else 'hdf5'}"  # flatten into out_dir
                tf.extract(m, out_dir)
            n += 1
    tar_path.unlink(missing_ok=True)
    print(f"  [{task}] kept {n} episodes -> {out_dir}")
    return n


def main() -> int:
    from egobench import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="episodes per task")
    ap.add_argument("--tasks", type=int, default=0, help="cap number of tasks (0=all)")
    ap.add_argument("--only", default="", help="comma list of specific task names")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    p = config.paths().ensure()
    budget = config.disk_budget_gb()
    dest = Path(p.data_root) / "egodex" / "test"
    tmp = Path(os.environ.get("EGOBENCH_STORE", p.data_root)) / "egodex_tars"
    tmp.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    all_tasks = sorted(f[:-4] for f in api.list_repo_files(REPO, repo_type="dataset")
                       if f.endswith(".tar"))
    if args.only:
        tasks = [t for t in all_tasks if t in set(args.only.split(","))]
    else:
        tasks = all_tasks[:args.tasks] if args.tasks else all_tasks
    print(f"EgoDex fetch: {len(tasks)} tasks x {args.k} ep -> {dest}  (budget {budget:.0f}GB)")

    total = 0
    for t in tasks:
        n = fetch_task(t, args.k, dest, tmp, budget)
        if n < 0:
            break
        total += max(n, 0)
    print(f"\ndone: ~{total} episodes across {len(tasks)} tasks in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
