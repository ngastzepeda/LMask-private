"""Evaluate the LMask TSPTW checkpoints in checkpoints/ and record *per-instance*
results, so gaps to BKS can be recomputed later without re-running the models.

This is the per-instance sibling of evaluation/eval_checkpoints.py. It shares the
exact evaluation protocol (8 dihedral8 augmentations, greedy decode, an instance
is feasible if *any* augmentation is feasible, reported cost is the minimum
feasible closed-tour length across augmentations, de-normalised by ``max_loc`` so
costs match AMAI's ``cost * domain_size``), but instead of aggregating over the
1k test set it writes one csv row per instance.

Output layout (one csv per evaluated checkpoint)::

    evaluation/instance_results/{size}/{variant}_{mode}.csv

with ``size`` in {n20, n50, n100_sw, n100_mw}, ``variant`` in {lmask, amai}, and
``mode`` in {best, last}. Each csv has columns:

    instance     0-based instance index (matches the npz / AMAI ordering)
    feasibility  True if a feasible solution was found for this instance
    objective    closed-tour length in original coordinate units; NaN if infeasible
    runtime      wall-clock seconds attributed to this instance (its batch's
                 inference time divided evenly across the batch's instances)

Because BKS is deliberately *not* baked in here, computing per-instance gaps and
aggregating later is just a join on (env, size, instance) against whatever
baseline results csv is current.
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

INSTANCE_RESULTS_DIR = ROOT_DIR / "evaluation" / "instance_results"
CKPT_DIR = ROOT_DIR / "checkpoints"

log_path = ROOT_DIR / "evaluation" / "logs" / "eval_instance_results.log"
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


@torch.inference_mode()
def evaluate_ckpt(ckpt_path, size, device, batch_size):
    """Evaluate one checkpoint; return a per-instance DataFrame (or None on failure).

    Columns: instance, feasibility, objective, runtime.
    """
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

    feas_all, cost_all, runtime_all = [], [], []
    autocast = (
        torch.amp.autocast("cuda")
        if device.type == "cuda"
        else torch.inference_mode()
    )
    with autocast:
        for i in range(0, n_instances, batch_size):
            batch = td_all[i : i + batch_size].to(device)
            bsz = batch.batch_size[0]
            start = time.time()
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

            # Per-instance runtime: this batch's inference time shared evenly
            # across its instances (models process the batch, not one instance).
            batch_time = time.time() - start
            per_instance_time = batch_time / bsz

            feas_all.append(ins_feas.cpu())
            cost_all.append(cost.cpu())
            runtime_all.append(torch.full((bsz,), per_instance_time))

    feas = torch.cat(feas_all)                                     # [N] bool
    cost = torch.cat(cost_all)                                     # [N] float, NaN if infeasible
    runtime = torch.cat(runtime_all)                              # [N] float, seconds

    return pd.DataFrame({
        "instance": np.arange(n_instances),
        "feasibility": feas.numpy(),
        "objective": cost.numpy(),
        "runtime": runtime.numpy(),
    })


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
    p.add_argument("--modes", nargs="+", default=["best"],
                   choices=["best", "last"],
                   help="which checkpoints to evaluate (default: best only)")
    p.add_argument("--batch_size", type=int, default=250,
                   help="instances per forward pass (x8 augmentations internally)")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    device = get_device(args.device)

    for mode in args.modes:
        jobs = discover_jobs([mode])
        logger.info(f"Discovered {len(jobs)} '{mode}' checkpoints")
        print(f"Discovered {len(jobs)} '{mode}' checkpoints")

        for job in jobs:
            df = evaluate_ckpt(job["ckpt"], job["size"], device, args.batch_size)
            if df is None:
                continue
            out_dir = INSTANCE_RESULTS_DIR / job["size"]
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / f"{job['variant']}_{mode}.csv"
            df.to_csv(out_path, index=False)

            n_feas = int(df["feasibility"].sum())
            mean_obj = float(np.nanmean(df["objective"].to_numpy())) if n_feas else float("nan")
            logger.info(
                f"{job['label']} ({mode}): {n_feas}/{len(df)} feasible, "
                f"mean obj {mean_obj:.3f} -> {os.path.relpath(out_path, ROOT_DIR)}"
            )
            print(f"{job['label']} ({mode}): feas={n_feas}/{len(df)} "
                  f"mean_obj={mean_obj:.3f} -> {os.path.relpath(out_path, ROOT_DIR)}")
