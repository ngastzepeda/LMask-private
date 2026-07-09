#!/usr/bin/env python3
"""Discover unfinished training runs and emit sbatch commands to resume them.

Runs on the cluster from the repo root, ALWAYS via the bash wrapper (which
loads the Python module and sources the venv):

    bash resume_jobs.sh                    # scan logs/runs and sbatch the resumes
    bash resume_jobs.sh --dry_run          # only print what would be submitted
    bash resume_jobs.sh --include-finished # also resume runs that reached max_epochs

For each run under

    logs/runs/<name>/<YYYY-MM-DD_HH>/

it picks the most recent attempt that has a ``checkpoints/last.ckpt``, reads the
wandb run id (from the ``wandb/`` folder) and the original ``experiment=`` /
``seed=`` overrides (from ``.hydra/overrides.yaml``), checks how far training
got (``epoch`` inside last.ckpt vs ``trainer.max_epochs``), and writes a resume
command for every run that has not reached ``max_epochs``.

Each submitted job resumes training from last.ckpt AND continues the same
wandb run:

    sbatch --job-name=<name>_resume start_job.sh <experiment> \\
        ckpt_path=<abs .../last.ckpt> \\
        logger.wandb.id=<id> +logger.wandb.resume=allow \\
        seed=<seed>

Use --dry_run first to review the sbatch commands before submitting.
"""
import argparse
import glob
import os
import subprocess
import sys

import yaml


def read_epoch(ckpt_path):
    """Return the last completed epoch stored in a Lightning checkpoint, or None."""
    try:
        import torch

        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ep = blob.get("epoch")
        return int(ep) if ep is not None else None
    except Exception as exc:  # torch missing, corrupt ckpt, unpicklable hparams, ...
        print(f"  ! could not read epoch from {ckpt_path}: {exc}", file=sys.stderr)
        return None


def wandb_id(ts_dir):
    """Extract the wandb run id from <ts_dir>/wandb/run-<date>_<time>-<id>/."""
    wdir = os.path.join(ts_dir, "wandb")
    if not os.path.isdir(wdir):
        return None
    # Prefer the 'latest-run' symlink; fall back to the newest run-* directory.
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


def read_overrides(ts_dir):
    """Return (experiment, seed) recorded in .hydra/overrides.yaml."""
    path = os.path.join(ts_dir, ".hydra", "overrides.yaml")
    experiment = seed = None
    if os.path.isfile(path):
        with open(path) as fh:
            for item in yaml.safe_load(fh) or []:
                if not isinstance(item, str):
                    continue
                if item.startswith("experiment="):
                    experiment = item.split("=", 1)[1]
                elif item.startswith("seed="):
                    seed = item.split("=", 1)[1]
    return experiment, seed


def read_max_epochs(ts_dir, default=1000):
    path = os.path.join(ts_dir, ".hydra", "config.yaml")
    try:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg["trainer"]["max_epochs"])
    except Exception:
        return default


def latest_attempt_with_ckpt(run_dir):
    """Newest <YYYY-MM-DD_HH> subdir that contains checkpoints/last.ckpt."""
    subdirs = sorted(
        (d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))),
        reverse=True,  # timestamp format sorts chronologically
    )
    for ts in subdirs:
        if os.path.isfile(os.path.join(run_dir, ts, "checkpoints", "last.ckpt")):
            return ts
    return None


def discover(logs_dir):
    rows = []
    for name in sorted(os.listdir(logs_dir)):
        run_dir = os.path.join(logs_dir, name)
        if not os.path.isdir(run_dir) or name == "None":
            continue
        ts = latest_attempt_with_ckpt(run_dir)
        if ts is None:
            print(f"- {name}: no last.ckpt in any attempt -> skip (resubmit fresh)",
                  file=sys.stderr)
            continue
        ts_dir = os.path.join(run_dir, ts)
        ckpt = os.path.abspath(os.path.join(ts_dir, "checkpoints", "last.ckpt"))
        experiment, seed = read_overrides(ts_dir)
        wid = wandb_id(ts_dir)
        max_epochs = read_max_epochs(ts_dir)
        epoch = read_epoch(ckpt)
        finished = epoch is not None and (epoch + 1) >= max_epochs
        rows.append(
            dict(name=name, attempt=ts, experiment=experiment, seed=seed,
                 wandb_id=wid, ckpt=ckpt, epoch=epoch, max_epochs=max_epochs,
                 finished=finished)
        )
    return rows


def build_command(row, start_job, resume_mode):
    parts = [
        f"sbatch --job-name={row['name']}_resume {start_job}",
        row["experiment"],
        f"ckpt_path={row['ckpt']}",
    ]
    if row["wandb_id"]:
        parts.append(f"logger.wandb.id={row['wandb_id']} +logger.wandb.resume={resume_mode}")
    if row["seed"]:
        parts.append(f"seed={row['seed']}")
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs/runs", help="root of run dirs")
    ap.add_argument("--start-job", default="start_job.sh", help="launcher script")
    ap.add_argument("--resume-mode", default="allow", choices=["allow", "must"],
                    help="wandb resume mode")
    ap.add_argument("--include-finished", action="store_true",
                    help="also resume runs that already reached max_epochs")
    ap.add_argument("--dry_run", action="store_true",
                    help="print sbatch commands without submitting")
    args = ap.parse_args()

    if not os.path.isdir(args.logs_dir):
        sys.exit(f"logs dir not found: {args.logs_dir}")

    rows = discover(args.logs_dir)

    # summary table
    print(f"\n{'run':28} {'attempt':16} {'epoch/max':11} {'wandb':10} experiment")
    print("-" * 90)
    for r in rows:
        ep = f"{r['epoch']}/{r['max_epochs']}" if r["epoch"] is not None else f"?/{r['max_epochs']}"
        tag = "DONE" if r["finished"] else "resume"
        print(f"{r['name']:28} {r['attempt']:16} {ep:11} {str(r['wandb_id']):10} "
              f"{r['experiment']}  [{tag}]")

    todo = [r for r in rows if args.include_finished or not r["finished"]]
    missing = [r for r in todo if not r["experiment"]]
    todo = [r for r in todo if r["experiment"]]
    for r in missing:
        print(f"- {r['name']}: no experiment= in overrides.yaml -> skip", file=sys.stderr)

    if not todo:
        print("\nNothing to resume.")
        return

    print()
    for r in todo:
        cmd = " ".join(build_command(r, args.start_job, args.resume_mode)).split()
        if args.dry_run:
            print("  [dry-run] " + " ".join(cmd))
            continue
        print(f"submitting: {r['name']}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        out = (result.stdout or result.stderr).strip()
        print(f"  {out}")


if __name__ == "__main__":
    main()
