# -----------------------------------------------------------------------------
# validate_ws_recovery_stroop.py  ⟶  Simulation‑Based Calibration for 𝑤_s
# -----------------------------------------------------------------------------
# Author : <your‑name>
# Updated: 2025‑05‑09
# Purpose: Run SBC for the **salience weight** parameter (w_s) in the NES‑DDM
#          Stroop‑like task.  The script is written to be **robust** against
#          NaNs, missing trials, or failed simulations and to be *drop‑in* —
#          you should be able to call:
#              python validate_ws_recovery_stroop.py  --seed 42  --iters 20
#          or    python validate_ws_recovery_stroop.py 42 20
#          The defaults are 100 iterations, single‑core sampler.
# -----------------------------------------------------------------------------
#  KEY CHANGES vs. previous draft
#  •  Strict validation of every trial → forced fallback instead of NaN
#  •  Summary‑stat helper that guarantees every key/value exists
#  •  Distance treats "missing" stats with a finite but large penalty
#  •  Centralised constants – no more N_TRIALS_PER_SIM vs. N_SIM_TRIALS bugs
#  •  Optional --multi flag uses PyABC's Multicore sampler (Unix only!)
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# ───────────────────────────────────────────────────────────────────────────────
# Optional progress bar (tiny dependency)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):
        return x

# ───────────────────────────────────────────────────────────────────────────────
# Project‑local imports (agent + config).  Fail early if missing.

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR / "src"
if SRC_DIR.exists() and SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))

try:
    from agent_mvnes import MVNESAgent           # Simulator wrapper
    from agent_config import (                   # Base physio / timing params
        T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    )
except Exception as e:
    print("[FATAL] Could not import MVNESAgent or agent_config →", e)
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────────────────
# pyabc (likelihood‑free inference)
try:
    import pyabc
    from pyabc import Distribution, RV, ABCSMC
    from pyabc.sampler import SingleCoreSampler, MulticoreEvalParallelSampler
except ImportError:
    print("[FATAL] pyabc is not installed.  pip install pyabc")
    sys.exit(1)

pyabc_logger = logging.getLogger("pyabc")
pyabc_logger.setLevel("WARNING")  # silence verbose info

# ======= CONSTANTS & CONFIG ====================================================
DEFAULT_ITERS            = 100
DEFAULT_SEED             = 42
N_TRIALS                 = 300        # per simulated dataset
CONFLICT_LEVELS          = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00])
CONFLICT_PROPS           = np.repeat(0.2, len(CONFLICT_LEVELS))
# Fixed NES parameters (tune as needed)
A_FIXED                  = 1.0        # threshold
WN_FIXED                 = 1.0        # norm weight – stays constant here

# Simulator base parameters (merged into every trial call)
BASE_SIM_PARAMS = dict(
    t                       = T_NONDECISION,
    noise_std_dev           = NOISE_STD_DEV,
    dt                      = DT,
    max_time                = 5.0,                # allow slow trials
    affect_stress_threshold_reduction = -0.30,
    veto_flag               = False,
)

# ABC-SMC hyper-parameters (optimized for speed)
POP_SIZE = 50    # Reduced from 120
MAX_GENS = 5     # Reduced from 8
# Prior for w_s  (salience weight)
WS_PRIOR         = Distribution(w_s=RV("uniform", 0.1, 1.5))

# ======= TASK INPUTS  (shared across all SBC iterations) =======================

def generate_task_inputs(seed: int | None = None):
    """Pre‑compute the salience & norm inputs for a block of trials."""
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(CONFLICT_LEVELS), size=N_TRIALS, p=CONFLICT_PROPS)
    S_in  = 1.0 - CONFLICT_LEVELS[idx]      # salience term
    N_in  = CONFLICT_LEVELS[idx]            # norm term
    labels= CONFLICT_LEVELS[idx]
    return S_in, N_in, labels

SAL_IN, NORM_IN, LABELS = generate_task_inputs(DEFAULT_SEED)  # global reuse

# ======= SIMULATION WRAPPERS ===================================================

def run_single_trial(agent: MVNESAgent, params: dict, s_val: float, n_val: float):
    """Run one trial safely, forcing valid outputs."""
    try:
        out = agent.run_mvnes_trial(salience_input=s_val, norm_input=n_val, params=params)
    except Exception as exc:
        print(f"[WARN] Trial error → {exc}")
        out = None

    # Fallback / validation
    if not out or any(k not in out for k in ("rt", "choice")):
        out = dict(rt=np.nan, choice=np.nan)

    rt     = out["rt"] if np.isfinite(out["rt"]) else params["max_time"]
    choice = out["choice"] if choice_is_valid(out.get("choice")) else np.nan
    return dict(rt=rt, choice=choice)

def choice_is_valid(x):
    return x in (0, 1)


def simulate_dataset(ws_sample: float, seed: int) -> pd.DataFrame:
    """Generate an entire synthetic dataset for a given w_s value."""
    agent   = MVNESAgent(config={})
    params  = dict(w_s=ws_sample, w_n=WN_FIXED, threshold_a=A_FIXED, **BASE_SIM_PARAMS)

    rng = np.random.default_rng(seed)
    rows = []
    for s_val, n_val, lbl in zip(SAL_IN, NORM_IN, LABELS):
        res = run_single_trial(agent, params, s_val, n_val)
        rows.append(dict(rt=res["rt"], choice=res["choice"], conflict_level=lbl))
    return pd.DataFrame(rows)

# ======= SUMMARY STATISTICS ====================================================

STATS_RT_FUNCS = dict(
    mean=np.nanmean,
    median=np.nanmedian,
    var=np.nanvar,
    p25=lambda x: np.nanpercentile(x, 25),
    p75=lambda x: np.nanpercentile(x, 75),
)


def stat_key(base: str, lvl: float):
    return f"{base}_lvl_{lvl:.2f}"


def compute_summary(df: pd.DataFrame) -> dict:
    template_keys = []
    for lvl in CONFLICT_LEVELS:
        for base in (
            "error_rate",
            "rt_mean_correct", "rt_median_correct", "rt_var_correct", "rt_p25_correct", "rt_p75_correct",
            "rt_mean_error",  "rt_median_error",  "rt_var_error",  "rt_p25_error",  "rt_p75_error",
            "n_correct", "n_error", "n_total",
        ):
            template_keys.append(stat_key(base, lvl))

    out = {k: np.nan for k in template_keys}

    for lvl, g in df.groupby("conflict_level"):
        lvl = float(lvl)
        tot = len(g)
        if tot == 0:    # shouldn’t happen but safety‑first
            continue
        err = g[g.choice == 0]
        cor = g[g.choice == 1]
        out[stat_key("error_rate", lvl)] = len(err) / tot
        out[stat_key("n_correct",   lvl)] = len(cor)
        out[stat_key("n_error",     lvl)] = len(err)
        out[stat_key("n_total",     lvl)] = tot

        # helper lambda
        def fill(rt_df, prefix):
            if len(rt_df) == 0:
                for k in STATS_RT_FUNCS:
                    out[stat_key(f"rt_{k}_{prefix}", lvl)] = BASE_SIM_PARAMS["max_time"]
            else:
                rts = rt_df.rt.values / BASE_SIM_PARAMS["max_time"]  # normalise 0‑1
                for k, fn in STATS_RT_FUNCS.items():
                    out[stat_key(f"rt_{k}_{prefix}", lvl)] = float(fn(rts))
        fill(cor, "correct")
        fill(err, "error")

    return out

# ======= DISTANCE FUNCTION =====================================================

def distance(obs: dict, sim: dict) -> float:
    dist = 0.0
    for k in obs.keys():
        o = obs[k]
        s = sim.get(k, np.nan)
        if not np.isfinite(o) or not np.isfinite(s):
            dist += 25.0                # moderate penalty – not INF
        else:
            w = 1.0 if "error_rate" in k else 0.5
            dist += w * (o - s) ** 2
    return np.sqrt(dist)

# ======= SBC RUNNER ============================================================

def run_sbc(seed: int, iters: int, use_multi: bool):
    rng = np.random.default_rng(seed)

    # Pre‑compute observed summary stats once per iteration
    results = []
    ranks   = []

    for idx in range(iters):
        true_ws = WS_PRIOR.rvs()["w_s"]
        df_obs  = simulate_dataset(true_ws, seed + idx)
        obs_sum = compute_summary(df_obs)

        # Simulator for ABC (closure captures shared trial inputs)
        def simulator(params):
            ws = params["w_s"]
            df = simulate_dataset(ws, seed + idx + 10_000)  # different RNG stream
            return compute_summary(df)

        # ABC-SMC setup with early stopping
        eps = pyabc.epsilon.QuantileEpsilon(initial_epsilon=1.0, alpha=0.3)  # More lenient epsilon
        sampler = (MulticoreEvalParallelSampler() if use_multi else SingleCoreSampler())
        
        abc = ABCSMC(
            models=simulator,
            parameter_priors=WS_PRIOR,
            distance_function=distance,
            population_size=POP_SIZE,
            eps=eps,
            sampler=sampler
        )
        
        # Setup database
        db_path = f"sqlite:///{Path(tempfile_prefix(seed, idx)).with_suffix('.db')}"
        abc.new(db_path, obs_sum)
        
        # Run with early stopping
        hist = abc.run(
            max_nr_populations=MAX_GENS,
            max_walltime=3600  # 1 hour max per SBC iteration
        )

        post = hist.get_distribution(m=0, t=hist.max_t)
        samples = post["w_s"].to_numpy() if post is not None else np.array([])
        rank    = np.sum(samples < true_ws) if len(samples) else np.nan
        ranks.append(rank)
        results.append(dict(true_ws=true_ws, mean_post=samples.mean() if len(samples) else np.nan,
                            rank=rank, n=len(samples)))
        print(f"[Iter {idx+1}/{iters}] true={true_ws:.3f}  mean̂={samples.mean() if len(samples) else np.nan:.3f}  rank={rank}")

    res_df = pd.DataFrame(results)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    outdir = SCRIPT_DIR / f"ws_sbc_results_{ts}"
    outdir.mkdir(exist_ok=True)
    res_df.to_csv(outdir / "results.csv", index=False)
    plot_histogram(ranks, POP_SIZE, outdir / "rank_hist.png")
    print(f"Results saved in {outdir}")

# Utility helpers
import tempfile, math

def tempfile_prefix(seed, idx):
    return Path(tempfile.gettempdir()) / f"ws_sbc_{seed}_{idx}"


def plot_histogram(ranks, n_post, path):
    r = np.asarray([x for x in ranks if np.isfinite(x)])
    plt.figure(figsize=(8,4))
    bins = min(25, n_post+1)
    plt.hist(r, bins=bins, range=(-0.5, n_post+0.5), density=True, color="#6c5ce7", edgecolor="black")
    plt.axhline(1/(n_post+1), color="red", ls="--")
    plt.xlabel("Rank of true w_s")
    plt.ylabel("Density")
    plt.title("SBC rank histogram for w_s")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ======= ENTRY‑POINT ===========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simulation‑Based Calibration for w_s (NES‑DDM Stroop)")
    ap.add_argument("seed",   nargs="?", type=int, default=DEFAULT_SEED)
    ap.add_argument("iters",  nargs="?", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--multi", action="store_true", help="Use multicore sampler (Unix)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    run_sbc(args.seed, args.iters, use_multi=args.multi)
