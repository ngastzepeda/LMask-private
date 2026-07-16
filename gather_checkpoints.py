#!/usr/bin/env python3
"""Gather each run's best/last checkpoint into a dedicated folder and git-add it.

Runs on the cluster from the repo root, ALWAYS via the bash wrapper (which
loads the Python module and sources the venv):

    bash gather_checkpoints.sh                # copy into checkpoints/ (no git)
    bash gather_checkpoints.sh --git-add      # ...and git add the folder
    bash gather_checkpoints.sh --dest somedir # different destination folder

IMPORTANT: resumed runs create one timestamp dir per attempt
(logs/runs/<name>/<YYYY-MM-DD_HH>/), and the checkpoint callback only writes a
new best.ckpt into the CURRENT attempt's dir if a new best occurs after the
resume. The true best of a run may therefore live in an EARLIER attempt dir
than the newest last.ckpt.

BUT not every timestamp dir belongs to the same training trajectory: a fresh
restart (e.g. after fixing a bug) also creates a new dir, and then the old
attempt's checkpoints are stale and must NOT be considered. The two cases are
told apart by the wandb run id: resumes continue the same id (resume_jobs.py
passes logger.wandb.id=...), fresh restarts get a new one. This script
therefore only scans the attempts of the CURRENT LINEAGE - the newest
attempt's wandb id and every older attempt sharing it:

  - best.ckpt: among lineage attempts, pick the one with the highest
    best_model_score stored in the checkpoint (monitor mode "max",
    e.g. val/max_aug_reward). Falls back to the newest attempt's best.ckpt
    if no score is readable.
  - last.ckpt: among lineage attempts, pick the one with the highest epoch
    stored in the checkpoint. Falls back to the newest attempt's last.ckpt.

Copies go to flat files <dest>/<run_name>_{best,last}.ckpt plus a manifest.csv
recording source path, epoch, and score for provenance. logs/ is gitignored,
so copying out is required for the checkpoints to be committable at all.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import torch


def ckpt_info(path):
    """Return (epoch, best_model_score) stored in a Lightning checkpoint."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  ! unreadable checkpoint {path}: {exc}", file=sys.stderr)
        return None, None
    epoch = blob.get("epoch")
    epoch = int(epoch) if epoch is not None else None
    score = None
    for key, state in (blob.get("callbacks") or {}).items():
        if "ModelCheckpoint" in str(key) and isinstance(state, dict):
            s = state.get("best_model_score")
            if s is not None:
                score = float(s)
    return epoch, score


def attempts(run_dir):
    """Timestamp subdirs of a run, oldest -> newest (name format sorts chronologically)."""
    return sorted(
        d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))
    )


def wandb_run_id(ts_dir):
    """Extract the wandb run id from <ts_dir>/wandb/run-<date>_<time>-<id>/."""
    wdir = os.path.join(ts_dir, "wandb")
    if not os.path.isdir(wdir):
        return None
    latest = os.path.join(wdir, "latest-run")
    cand = None
    if os.path.exists(latest):
        cand = os.path.basename(os.path.realpath(latest))
    if not cand or not cand.startswith("run-"):
        runs = sorted(glob.glob(os.path.join(wdir, "run-*")))
        cand = os.path.basename(runs[-1]) if runs else None
    if cand and cand.startswith("run-"):
        return cand.rsplit("-", 1)[-1]
    return None


def lineage_attempts(run_dir):
    """Attempts belonging to the current training trajectory, oldest -> newest.

    The newest attempt's wandb id defines the lineage; older attempts count
    only if they share it (i.e. the newest attempt resumed them). A fresh
    restart has a new id, which cuts older attempts off. If no attempt has a
    wandb id (local/dev runs), fall back to all attempts.
    """
    atts = attempts(run_dir)
    ids = {ts: wandb_run_id(os.path.join(run_dir, ts)) for ts in atts}
    current = next((ids[ts] for ts in reversed(atts) if ids[ts]), None)
    if current is None:
        return atts
    kept = [ts for ts in atts if ids[ts] == current]
    dropped = [ts for ts in atts if ts not in kept]
    if dropped:
        print(f"  (ignoring stale attempt(s) of {os.path.basename(run_dir)}: "
              f"{', '.join(dropped)} - different wandb id than current lineage "
              f"'{current}')", file=sys.stderr)
    return kept


def pick(run_dir, lineage, filename, rank_key):
    """Best candidate of `filename` across the given lineage attempts.

    rank_key maps (epoch, score) -> sortable value or None to express
    'no usable ranking signal'. Ties / no-signal cases resolve to the newest
    attempt because candidates are scanned oldest -> newest.
    """
    best = None  # (rank_value_or_None, path, epoch, score)
    for ts in lineage:
        path = os.path.join(run_dir, ts, "checkpoints", filename)
        if not os.path.isfile(path):
            continue
        epoch, score = ckpt_info(path)
        rank = rank_key(epoch, score)
        if best is None or (rank is not None and (best[0] is None or rank >= best[0])):
            best = (rank, path, epoch, score)
        elif rank is None and best[0] is None:
            best = (rank, path, epoch, score)  # keep newest among unranked
    return best  # or None if the file exists in no attempt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs/runs", help="root of run dirs")
    ap.add_argument("--dest", default="checkpoints", help="destination folder")
    ap.add_argument("--git-add", action="store_true",
                    help="git add the destination folder after copying")
    args = ap.parse_args()

    if not os.path.isdir(args.logs_dir):
        sys.exit(f"logs dir not found: {args.logs_dir}")

    rows = []
    for name in sorted(os.listdir(args.logs_dir)):
        run_dir = os.path.join(args.logs_dir, name)
        if not os.path.isdir(run_dir) or name == "None":
            continue

        # only the current lineage (newest wandb id + its resumed ancestors)
        lineage = lineage_attempts(run_dir)
        # best: highest stored best_model_score (mode "max"); last: highest epoch
        best = pick(run_dir, lineage, "best.ckpt", lambda ep, sc: sc)
        last = pick(run_dir, lineage, "last.ckpt", lambda ep, sc: ep)
        if best is None and last is None:
            print(f"- {name}: no checkpoints in any attempt -> skip", file=sys.stderr)
            continue

        os.makedirs(args.dest, exist_ok=True)
        for kind, cand in (("best", best), ("last", last)):
            if cand is None:
                print(f"- {name}: no {kind}.ckpt found", file=sys.stderr)
                continue
            _, src, epoch, score = cand
            dst = os.path.join(args.dest, f"{name}_{kind}.ckpt")
            shutil.copy2(src, dst)
            rows.append(dict(run=name, kind=kind, source=src, epoch=epoch, score=score))
            print(f"{name:24} {kind}: epoch={epoch} score={score}  <- {src}")

    if not rows:
        sys.exit("Nothing gathered.")

    manifest = os.path.join(args.dest, "manifest.csv")
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run", "kind", "source", "epoch", "score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote manifest: {manifest}")

    if args.git_add:
        subprocess.run(["git", "add", args.dest], check=True)
        print(f"git add {args.dest} done -- review with 'git status', then commit.")


if __name__ == "__main__":
    main()
