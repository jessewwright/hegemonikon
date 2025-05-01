import argparse
import json
import os
import datetime
import numpy as np
import pandas as pd
from nes.comparator import Comparator
from nes.assent_gate import AssentGate
from nes.raa import RAA
from nes.norm_repository import NormRepository
from nes.sim.moral_dilemma import MoralDilemmaSimulator

def load_params(params_file):
    with open(params_file, 'r') as f:
        return json.load(f)

def run_task(task, params, n_trials):
    """
    Dispatch to the appropriate NES task simulation.

    Args:
        task (str): one of 'stroop', 'gng', 'dd', 'moral_dilemma'
        params (dict): parameter dictionary loaded from JSON
        n_trials (int): number of trials to simulate

    Returns:
        pd.DataFrame: raw trial data
    """
    if task == "stroop":
        comp = Comparator(**params)
        results = [comp.run_trial() for _ in range(n_trials)]
    elif task == "moral_dilemma":
        # Split params into moral dilemma and RAA params
        params_md = {k: v for k, v in params.items() if k != 'raa'}
        params_raa = params['raa']
        simulator = MoralDilemmaSimulator(params_md, params_raa)
        
        # Use fixed parameters for balanced conflict condition
        results = []
        fixed_params = {
            'serotonin_level': 0.0,
            'salience_lie': 0.1,
            'salience_truth': 0.1
        }
        
        for _ in range(n_trials):
            # Use norms from params
            norms = params_md.get('norms', {'lie': -0.7, 'truth': 0.7})
            
            result = simulator.run_trial(
                fixed_params['serotonin_level'],
                fixed_params['salience_lie'],
                fixed_params['salience_truth'],
                norms
            )
            results.append({
                'serotonin_level': fixed_params['serotonin_level'],
                'salience_lie': fixed_params['salience_lie'],
                'salience_truth': fixed_params['salience_truth'],
                **result
            })
    else:
        raise ValueError(f"Unknown task: {task}")
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(
        description="NES simulation runner (hegemonikon project)"
    )
    parser.add_argument("task", help="Which task to run (stroop|gng|dd|moral_dilemma)")
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
