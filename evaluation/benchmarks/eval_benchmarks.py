"""Evaluate the trained TSPTW checkpoints (the LMaskPenaltyModel `.ckpt` files in
checkpoints/, same ones evaluation/eval_checkpoints.py uses) on the classic
*out-of-distribution* TSPTW benchmark libraries under data/benchmarks/tsptw/
(Dumas, GendreauDumasExtended, OhlmannThomas, Langevin).

Why this pipeline is different from eval_instance_results.py
-----------------------------------------------------------
Those benchmark files are given as an explicit distance matrix (line 1 = number
of nodes n incl. depot at index 0, then an n x n matrix, then n rows of
``tw_early tw_late``), with *no coordinates*. But the LMaskPenaltyModel policy
and its env are purely coordinate-based: the tour objective and the time-window
feasibility are computed as Euclidean distances over 2D ``locs`` (see
lmask/envs/tsptw/env.py::TSPTWEnv._step / _get_reward), and the dihedral8
augmentation is a geometric transform. There is no way to feed a raw matrix in.

So we do what the user asked (and what driver/test_dumas.py does in spirit):

  1. Reconstruct 2D coordinates from each distance matrix via classical (metric)
     MDS -- these are only used to let the policy *decode a tour* (a visiting
     order of the nodes).
  2. Take the visiting order the policy produced (out["actions"], one per
     dihedral8 augmentation) and compute the EXACT objective and EXACT
     time-window feasibility from the ORIGINAL distance matrix + time windows,
     using the checker semantics from
     ../AMAI2025/source/checkers/solution_checker/checker.py::_check_route
     (single vehicle, travel_cost = 1, closed tour returning to the depot).

Difference vs driver/test_dumas.py: that script uses the *real* coordinate files
(data/dumas/locations) and the tsptw-lazymask env's built-in
``get_reward_by_distance`` to score from the matrix, with the upstream `.pth`
policy. We have no coordinate files for these benchmark sets, so we MDS-recon
them, and the penalty `.ckpt` env cannot score from a matrix, so we score the
decoded tour externally. The `.pth` models are NOT used.

Size matching (manifest-driven)
-------------------------------
Which instances a checkpoint runs on is decided by the benchmark manifest
(source/benchmarks/tsptw_benchmark_manifest.json in the AMAI2025 repo). It groups
instances by nominal size (customer count rounded to the nearest ten -- e.g. the
"20" group holds instances with 19 or 20 customers) and maps each checkpoint size
to the group(s) it should be compared on via ``amai_checkpoint_size_match``:
n20 -> [20], n50 -> [] (no exact-size benchmark, so n50 checkpoints are skipped),
n100 -> [100]. Both n100_sw and n100_mw map to the size-100 group. Sizes without a
matching model (40/60/80/150/200) are never evaluated, which drops OhlmannThomas
entirely (it is only 150/200).

Aggregation mirrors eval_instance_results.py: an instance is feasible if *any*
augmentation yields a TW-feasible tour, and its objective is the minimum exact
closed-tour length over the feasible augmentations (NaN if none feasible).

Output (one csv per checkpoint)::

    evaluation/benchmarks/results/{variant}_{size}_{mode}.csv

with columns:

    instance    manifest instance name, e.g. Dumas/n20w100.001 (family-prefixed)
    family      benchmark library (Dumas / GendreauDumasExtended / ...)
    feasible    True if a TW-feasible tour was found (exact check on the matrix)
    objective   exact closed-tour length in the benchmark's own units; NaN if not
    runtime     wall-clock seconds of model inference for this instance
                (reset + augment + policy decode; MDS/parsing/scoring excluded)

No BKS/gap is computed here -- like eval_instance_results.py, gaps are meant to
be joined on the instance name against a benchmark BKS table afterwards.
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyrootutils
import torch
from loguru import logger
from tensordict.tensordict import TensorDict

ROOT_DIR = pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from lmask.models.model import LMaskPenaltyModel  # noqa: E402

RESULTS_DIR = ROOT_DIR / "evaluation" / "benchmarks" / "results"
CKPT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_MANIFEST = (
    ROOT_DIR.parent
    / "AMAI2025"
    / "source"
    / "benchmarks"
    / "tsptw_benchmark_manifest.json"
)

log_path = ROOT_DIR / "evaluation" / "benchmarks" / "logs" / "eval_benchmarks.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger.add(str(log_path), level="TRACE", rotation="100 MB", retention=10)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


# --------------------------------------------------------------------------- #
# Benchmark instance parsing (distance-matrix format)
# --------------------------------------------------------------------------- #
def read_matrix_instance(path):
    """Parse a matrix-format TSPTW instance file.

    Layout (token stream, robust to line wrapping):
        n | n*n distance matrix | n*2 (tw_early tw_late) [| n service times]

    Returns (D [n, n], tw [n, 2], service [n]) as float64 numpy arrays. Node 0 is
    the depot. Service times default to 0 when the file only lists time windows
    (the case for Dumas/GendreauDumasExtended/OhlmannThomas/Langevin).
    """
    with open(path, "r") as f:
        tokens = f.read().split()
    vals = np.array([float(t) for t in tokens], dtype=np.float64)

    n = int(round(vals[0]))
    off = 1
    D = vals[off : off + n * n].reshape(n, n)
    off += n * n
    rest = vals[off:]

    if rest.size == 2 * n:
        tw = rest.reshape(n, 2)
        service = np.zeros(n, dtype=np.float64)
    elif rest.size == 3 * n:  # some variants append a service-time column
        rest = rest.reshape(n, 3)
        tw = rest[:, :2]
        service = rest[:, 2].copy()
    else:
        raise ValueError(
            f"{path}: unexpected token count after matrix "
            f"({rest.size}, expected {2 * n} or {3 * n} for n={n})"
        )
    return D, tw, service


# --------------------------------------------------------------------------- #
# Coordinate reconstruction (classical MDS) + model input
# --------------------------------------------------------------------------- #
def classical_mds(D, dim=2):
    """Torgerson classical MDS: distance matrix -> `dim`-D coordinates.

    Deterministic (eigendecomposition, no random init). The matrix is
    symmetrized first; negative eigenvalues (from non-Euclidean rounding) are
    clamped to 0 -- the top-`dim` components are the best low-rank Euclidean
    embedding of the given distances.
    """
    D = 0.5 * (D + D.T)
    n = D.shape[0]
    D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    with np.errstate(all="ignore"):  # some BLAS backends raise spurious FP flags
        B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    idx = np.argsort(eigvals)[::-1][:dim]
    lam = np.clip(eigvals[idx], 0.0, None)
    coords = eigvecs[:, idx] * np.sqrt(lam)
    return coords  # [n, dim]


def make_input_td(coords, tw, service, device):
    """Build the coordinate-based TensorDict the penalty env expects.

    Coordinates are shifted to be non-negative (distances are shift-invariant)
    and scaled to ~[0, 1]; time windows and service times are divided by the same
    factor so travel-time vs time-window comparisons stay consistent -- exactly
    the max_loc normalization load_tsptw_npz applies to the training data.
    """
    coords = coords - coords.min(axis=0, keepdims=True)
    scale = float(coords.max()) or 1.0
    locs = coords / scale
    tw_n = tw / scale
    st_n = service / scale

    td = TensorDict(
        {
            "locs": torch.as_tensor(locs, dtype=torch.float32).unsqueeze(0),
            "service_time": torch.as_tensor(st_n, dtype=torch.float32).unsqueeze(0),
            "time_windows": torch.as_tensor(tw_n, dtype=torch.float32).unsqueeze(0),
            "n_vehicles": torch.ones(1, dtype=torch.int64),
        },
        batch_size=[1],
    )
    return td.to(device)


# --------------------------------------------------------------------------- #
# Exact scoring on the original matrix (checker semantics)
# --------------------------------------------------------------------------- #
def score_tours(actions, D, tw, service, eps):
    """Exact objective + TW feasibility for each decoded tour.

    ``actions`` is [A, n-1] node indices (visiting order of the customers,
    excluding the depot at index 0), one row per augmentation. Mirrors
    _check_route in source/checkers/solution_checker/checker.py: start at the
    depot at its opening time, travel by the true matrix distance, a late arrival
    (> tw_late) is a violation, wait up to tw_early, then serve; finally return to
    the depot and check its closing time. travel_cost is 1, so objective == the
    closed-tour distance.

    Returns (feas [A] bool, length [A] float; length is exact even if infeasible).
    """
    A = actions.shape[0]
    feas = np.zeros(A, dtype=bool)
    length = np.zeros(A, dtype=np.float64)
    depot_early, depot_late = float(tw[0, 0]), float(tw[0, 1])
    for a in range(A):
        t = depot_early
        prev = 0
        total = 0.0
        ok = True
        for node in actions[a]:
            node = int(node)
            leg = float(D[prev, node])
            total += leg
            t += leg
            if t > float(tw[node, 1]) + eps:
                ok = False
            t = max(t, float(tw[node, 0])) + float(service[node])
            prev = node
        leg = float(D[prev, 0])
        total += leg
        t += leg
        if t > depot_late + eps:
            ok = False
        feas[a] = ok
        length[a] = total
    return feas, length


# --------------------------------------------------------------------------- #
# Manifest + checkpoint discovery
# --------------------------------------------------------------------------- #
def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_ckpt_name(fname):
    """'lmask_n100_sw_best.ckpt' -> (run, mode, variant, size)."""
    stem = fname[: -len(".ckpt")]
    run, mode = stem.rsplit("_", 1)
    variant, size = run.split("_", 1)
    return run, mode, variant, size


def ckpt_size_key(size_token):
    """Map a checkpoint size token to a manifest size-match key.

    'n20' -> 'n20', 'n50' -> 'n50', 'n100_sw'/'n100_mw' -> 'n100' (the width
    suffix is a training-distribution flavour, not a different customer count).
    """
    m = re.match(r"(n\d+)", size_token)
    return m.group(1) if m else size_token


def discover_jobs(modes):
    jobs = []
    for path in sorted(CKPT_DIR.glob("*.ckpt")):
        try:
            run, mode, variant, size = parse_ckpt_name(path.name)
        except ValueError:
            logger.warning(f"Skipping unparseable checkpoint name: {path.name}")
            continue
        if mode not in modes:
            continue
        jobs.append(dict(run=run, mode=mode, variant=variant, size=size, ckpt=path))
    return jobs


def instances_for_ckpt(job, manifest):
    """Manifest instance entries a checkpoint should be evaluated on (may be []).

    Each entry is {name, family, path, customers}; paths are repo-root relative.
    """
    key = ckpt_size_key(job["size"])
    sizes = manifest["amai_checkpoint_size_match"].get(key, [])
    instances = []
    for s in sizes:
        instances.extend(manifest["sizes"][str(s)]["instances"])
    return sizes, instances


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def run_inference(env, policy, augment, n_aug, td_input):
    """Reset -> augment -> greedy decode; return actions [A, n-1] (cpu numpy)."""
    autocast = (
        torch.amp.autocast("cuda")
        if td_input.device.type == "cuda"
        else torch.inference_mode()
    )
    with autocast:
        td = env.reset(td_input)
        if n_aug > 1:
            td = augment(td)
        out = policy(td, env, phase="test", num_samples=1, return_actions=True)
    return out["actions"].cpu().numpy()


@torch.inference_mode()
def evaluate_ckpt(job, manifest, device, eps, warmup=True):
    ckpt_name = f"{job['variant']}_{job['size']}_{job['mode']}"
    sizes, instances = instances_for_ckpt(job, manifest)
    if not instances:
        logger.warning(
            f"{ckpt_name}: no benchmark instances for size match {sizes}; skipping."
        )
        print(f"{ckpt_name}: no matching-size benchmarks (size match {sizes}); skipped.")
        return

    try:
        model = LMaskPenaltyModel.load_from_checkpoint(
            checkpoint_path=str(job["ckpt"]), map_location="cpu"
        )
    except Exception as exc:
        logger.error(f"Failed to load {job['ckpt']}: {exc}")
        return

    env = model.env
    policy = model.policy.to(device).eval()
    augment = model.augment
    n_aug = model.num_augment

    rows = []
    warmed = not warmup
    for entry in instances:
        path = ROOT_DIR / entry["path"]
        D, tw, service = read_matrix_instance(path)
        coords = classical_mds(D, dim=2)
        td_input = make_input_td(coords, tw, service, device)

        # One untimed forward per checkpoint so CUDA init doesn't skew the first
        # recorded runtime.
        if not warmed:
            run_inference(env, policy, augment, n_aug, td_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warmed = True

        start = time.time()
        actions = run_inference(env, policy, augment, n_aug, td_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runtime = time.time() - start

        feas, length = score_tours(actions, D, tw, service, eps)
        feasible = bool(feas.any())
        objective = float(length[feas].min()) if feasible else float("nan")

        rows.append({
            "instance": entry["name"],
            "family": entry["family"],
            "feasible": feasible,
            "objective": objective,
            "runtime": runtime,
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = RESULTS_DIR / f"{ckpt_name}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    n_feas = sum(r["feasible"] for r in rows)
    logger.info(
        f"{ckpt_name} (sizes {sizes}): {n_feas}/{len(rows)} feasible "
        f"-> {os.path.relpath(out_path, ROOT_DIR)}"
    )
    print(f"{ckpt_name} | sizes {sizes}: feas={n_feas}/{len(rows)} "
          f"-> {os.path.relpath(out_path, ROOT_DIR)}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--modes", nargs="+", default=["best"],
                   choices=["best", "last"],
                   help="which checkpoints to evaluate (default: best only)")
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                   help="benchmark manifest json (size grouping + checkpoint size match)")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="time-window violation tolerance for the exact feasibility check")
    p.add_argument("--no_warmup", action="store_true",
                   help="disable the untimed warm-up forward per checkpoint")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    device = get_device(args.device)
    manifest = load_manifest(args.manifest)

    jobs = []
    for mode in args.modes:
        jobs.extend(discover_jobs([mode]))
    logger.info(f"Discovered {len(jobs)} checkpoints; manifest={args.manifest}")
    print(f"Discovered {len(jobs)} checkpoints; manifest={os.path.basename(args.manifest)}")

    for job in jobs:
        evaluate_ckpt(job, manifest, device, args.eps, warmup=not args.no_warmup)
