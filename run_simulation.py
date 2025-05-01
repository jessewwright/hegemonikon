import argparse
import json
import os
import datetime
import pandas as pd
from nes.comparator import Comparator
from nes.assent_gate import AssentGate
from nes.raa import RAA
from nes.norm_repository import NormRepository

def load_params(params_file):
    with open(params_file, 'r') as f:
        return json.load(f)

def run_task(task, params, n_trials):
    """
    Dispatch to the appropriate NES task simulation.

    Args:
        task (str): one of 'stroop', 'gng', 'dd', 'moral'
        params (dict): parameter dictionary loaded from JSON
        n_trials (int): number of trials to simulate

    Returns:
        pd.DataFrame: raw trial data
    """
    # Example for Stroop; expand for other tasks
    if task == "stroop":
        comp = Comparator(**params)
        results = [comp.run_trial() for _ in range(n_trials)]
    else:
        raise ValueError(f"Unknown task: {task}")
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="NES simulation runner (hegemonikon project)"
    )
    parser.add_argument("task", help="Which task to run (stroop|gng|dd|moral)")
    parser.add_argument(
        "--params", "-p", required=True,
        help="Path to JSON file with default parameters"
    )
    parser.add_argument(
        "--n-trials", "-n", type=int, default=100,
        help="Number of trials to run (default: 100)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress messages"
    )

    args = parser.parse_args()
    params = load_params(args.params)

    if args.verbose:
        print(f"Running {args.task} for {args.n_trials} trials using {args.params}")

    df = run_task(args.task, params, args.n_trials)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("results", args.task, ts)
    os.makedirs(outdir, exist_ok=True)
    raw_fp = os.path.join(outdir, "raw.csv")
    summary_fp = os.path.join(outdir, "summary.csv")

    df.to_csv(raw_fp, index=False)
    summary = df.describe().transpose()
    summary.to_csv(summary_fp)

    if args.verbose:
        print(f"Saved raw to {raw_fp}, summary to {summary_fp}")

if __name__ == "__main__":
    main()
