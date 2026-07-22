"""Evaluate the LMask TSPTW checkpoints in checkpoints/ on the AMAI test
instances, producing the same metrics (and txt/csv format) as the AMAI repo's
source/evaluation/eval_checkpoints.py so the two can be compared directly.

The checkpoints in checkpoints/ are flat files named ``{run}_{mode}.ckpt`` where
``run = {variant}_{size}`` (variant in {amai, lmask}; size in
{n20, n50, n100_sw, n100_mw}) and ``mode`` in {best, last}. All of them are the
same LMaskPenaltyModel (TSPTW); the two variants differ only in training budget.

Evaluation protocol mirrors the model's own val/test path (see
lmask/models/model.py::LMaskPenaltyModel.shared_step): 8 dihedral8 augmentations,
greedy decode, an instance is feasible if *any* augmentation is feasible, and its
reported cost is the minimum feasible tour length across augmentations.

Cost comparability with AMAI: both repos compute the closed-tour length
(including the return-to-depot arc) in coordinates normalised to [0, 1]. AMAI
reports ``cost * domain_size`` with ``domain_size = max_loc`` (see
AMAI2025/source/evaluation/utils.py::get_test_metrics). We de-normalise by the
same ``max_loc`` read from the npz, so costs are in the original coordinate units
and directly comparable. The test_1k.npz files here are byte-identical to AMAI's.

Gap to BKS: since the instances are identical, we optionally attach the
per-instance BKS from the AMAI baseline results csv (same convention as
AMAI's attach_bks) and report the mean gap in percent. If the csv is absent the
gap column is NaN (matching AMAI's committed checkpoint-eval rows).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyrootutils
import torch
from loguru import logger
from rl4co.utils.ops import unbatchify

ROOT_DIR = pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from lmask.models.model import LMaskPenaltyModel  # noqa: E402
from lmask.utils.data_utils import load_tsptw_npz  # noqa: E402

RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
CKPT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_BKS_CSV = (
    ROOT_DIR.parent
    / "AMAI2025"
    / "source"
    / "evaluation"
    / "results"
    / "eval_baselines_instance_results.csv"
)

log_path = ROOT_DIR / "evaluation" / "logs" / "eval_checkpoints.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger.add(str(log_path), level="TRACE", rotation="100 MB", retention=10)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def instance_dir(size: str) -> str:
    """ckpt size token -> instances dir name.

    n20 -> n20_amai, n50 -> n50_amai, n100_sw -> n100_amai_sw, n100_mw -> n100_amai_mw.
    """
    parts = size.split("_")
    return f"{parts[0]}_amai_{parts[1]}" if len(parts) == 2 else f"{size}_amai"


def get_test_file(size: str) -> str:
    return str(ROOT_DIR / "data" / "feasible" / "npz" / instance_dir(size) / "test_1k.npz")


def parse_ckpt_name(fname: str):
    """'lmask_n100_sw_best.ckpt' -> (run, mode, variant, size)."""
    stem = fname[: -len(".ckpt")]
    run, mode = stem.rsplit("_", 1)          # ('lmask_n100_sw', 'best')
    variant, size = run.split("_", 1)         # ('lmask', 'n100_sw')
    return run, mode, variant, size


def get_bks(size: str, bks_csv, n_instances: int, device):
    """Per-instance BKS tensor [n_instances] from the AMAI baseline csv, or None.

    The csv 'size' key strips the leading 'n' (e.g. 20_amai, 100_amai_sw) and
    rows are ordered by instance index, matching the npz order.
    """
    if bks_csv is None or not os.path.exists(bks_csv):
        logger.warning(f"No baseline csv ({bks_csv}); gap to BKS will be NaN.")
        return None
    df = pd.read_csv(bks_csv)
    key = instance_dir(size)[1:]  # 'n20_amai' -> '20_amai'
    df = df[(df["env"] == "tsptw") & (df["size"] == key)].sort_values("instance")
    if len(df) == n_instances and df["bks"].notna().all():
        return torch.tensor(df["bks"].to_numpy(), dtype=torch.float32, device=device)
    logger.warning(
        f"Baseline rows ({len(df)}) != instances ({n_instances}) or NaN bks for "
        f"tsptw {key}; gap to BKS NaN."
    )
    return None


@torch.inference_mode()
def evaluate_ckpt(ckpt_path, size, device, batch_size, bks_csv):
    """Evaluate one checkpoint; return a result-row dict (or None on failure)."""
    try:
        model = LMaskPenaltyModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path), map_location="cpu"
        )
    except Exception as exc:  # incompatible/corrupt ckpt -> skip, keep going
        logger.error(f"Failed to load {ckpt_path}: {exc}")
        return None

    env = model.env
    policy = model.policy.to(device).eval()
    augment = model.augment
    n_aug = model.num_augment
    round_eps = model.round_eps

    test_file = get_test_file(size)
    logger.info(f"Loading dataset: {test_file}")
    td_all = load_tsptw_npz(test_file, normalize=model.data_normalize)
    n_instances = td_all.batch_size[0]

    # De-normalisation factor: load_tsptw_npz scales coords by max_loc when the
    # raw coords are not already in ~[0, 1]. Recover that same factor so reported
    # costs are in the original units (matching AMAI's cost * domain_size).
    raw = np.load(test_file)
    max_loc = float(np.asarray(raw["max_loc"]).max()) if "max_loc" in raw else 1.0
    scale = max_loc if float(td_all["locs"].max()) <= 1.5 else 1.0
    if scale == 1.0:
        logger.warning(f"{test_file}: coords not normalised; reporting raw cost.")

    bks = get_bks(size, bks_csv, n_instances, device)

    feas_all, cost_all = [], []
    start = time.time()
    autocast = (
        torch.amp.autocast("cuda")
        if device.type == "cuda"
        else torch.inference_mode()
    )
    with autocast:
        for i in range(0, n_instances, batch_size):
            batch = td_all[i : i + batch_size].to(device)
            td = env.reset(batch)
            if n_aug > 1:
                td = augment(td)
            out = policy(td, env, phase="test", num_samples=1, return_actions=False)

            # out["reward"] is a TensorDict over B*n_aug; unbatchify to [B, n_aug, 1]
            reward_td = unbatchify(out["reward"], (n_aug, 1))
            length = -reward_td["negative_length"]                 # [B, A, 1]
            tcv = reward_td["total_constraint_violation"]          # [B, A, 1]
            sol_feas = tcv < round_eps                             # [B, A, 1]

            ins_feas = sol_feas.flatten(1).any(dim=1)              # [B]
            length_feas = length.masked_fill(~sol_feas, float("inf"))
            min_len = length_feas.flatten(1).min(dim=1).values     # [B] (inf if infeasible)
            cost = min_len * scale
            cost = torch.where(ins_feas, cost, torch.full_like(cost, float("nan")))

            feas_all.append(ins_feas.cpu())
            cost_all.append(cost.cpu())
    inference_time = time.time() - start

    feas = torch.cat(feas_all)                                     # [N] bool
    cost = torch.cat(cost_all)                                     # [N] float, NaN if infeasible

    if bks is not None:
        gap = ((cost.to(device) - bks) / bks * 100.0).cpu()        # [N]
        gap_mean = gap.nanmean().item()
    else:
        gap_mean = float("nan")

    return {
        "feas_count": int(feas.sum().item()),
        "cost": round(float(np.nanmean(cost.numpy())), 4) if feas.any() else float("nan"),
        "gap_bks": round(gap_mean, 4),
        "inference_time": round(inference_time, 3),
    }


def discover_jobs(modes):
    """Scan checkpoints/ for *_{mode}.ckpt and build one job per file."""
    jobs = []
    for path in sorted(CKPT_DIR.glob("*.ckpt")):
        try:
            run, mode, variant, size = parse_ckpt_name(path.name)
        except ValueError:
            logger.warning(f"Skipping unparseable checkpoint name: {path.name}")
            continue
        if mode not in modes:
            continue
        if instance_dir(size) not in ("n20_amai", "n50_amai", "n100_amai_sw", "n100_amai_mw"):
            logger.warning(f"Skipping {path.name}: unknown size token '{size}'.")
            continue
        label = f"TSPTW | {instance_dir(size)} - {variant}"
        jobs.append(dict(run=run, mode=mode, variant=variant, size=size,
                         ckpt=path, label=label))
    return jobs


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--modes", nargs="+", default=["best", "last"],
                   choices=["best", "last"])
    p.add_argument("--batch_size", type=int, default=250,
                   help="instances per forward pass (x8 augmentations internally)")
    p.add_argument("--bks_csv", type=str, default=str(DEFAULT_BKS_CSV),
                   help="AMAI baseline csv for per-instance BKS (gap). "
                        "Pass 'none' to skip and report NaN gaps.")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    device = get_device(args.device)
    bks_csv = None if args.bks_csv.lower() == "none" else args.bks_csv

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for mode in args.modes:
        jobs = discover_jobs([mode])
        logger.info(f"Discovered {len(jobs)} '{mode}' checkpoints")
        print(f"Discovered {len(jobs)} '{mode}' checkpoints")

        txt_path = RESULTS_DIR / f"eval_checkpoints_{mode}.txt"
        csv_path = RESULTS_DIR / f"eval_checkpoints_{mode}.csv"
        with open(txt_path, "a") as f:
            f.write("##### ----- NEW EVALUATION RUN ----- #####\n")

        rows = []
        for job in jobs:
            row = evaluate_ckpt(job["ckpt"], job["size"], device,
                                args.batch_size, bks_csv)
            if row is None:
                continue
            ckpt_rel = os.path.relpath(job["ckpt"], ROOT_DIR)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(txt_path, "a") as f:
                f.write(
                    f"{ts} | {job['label']} ({mode}):\t {row['feas_count']} & "
                    f"{row['cost']:.3f} & {row['gap_bks']:.3f} & "
                    f"{row['inference_time']:.3f} | ckpt_path: {ckpt_rel}\n"
                )
            rows.append({
                "variant": job["variant"], "size": job["size"],
                "instance_dir": instance_dir(job["size"]), "mode": mode,
                **row, "ckpt_path": ckpt_rel,
            })
            print(f"{job['label']} ({mode}): feas={row['feas_count']} "
                  f"cost={row['cost']:.3f} gap={row['gap_bks']:.3f}% "
                  f"time={row['inference_time']:.3f}s")

        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            logger.info(f"Wrote {len(rows)} rows to {csv_path}")
            print(f"Wrote {len(rows)} rows to {os.path.relpath(csv_path, ROOT_DIR)}")
