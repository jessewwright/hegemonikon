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

def run_simulation(params):
    # Initialize components
    comparator = Comparator(
        drift_rate=params['drift_rate'],
        threshold=params['threshold'],
        noise_std=params['noise_std']
    )
    assent_gate = AssentGate()
    raa = RAA()
    norm_repo = NormRepository()
    
    # Example simulation loop
    results = []
    for trial in range(10):  # Example number of trials
        trial_result = {
            'trial': trial,
            'comparator_output': comparator.run_trial(),
            'assent_gate_output': assent_gate.process_input(1.0),
            'raa_output': raa.update(1.0)
        }
        results.append(trial_result)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Run NES simulation')
    parser.add_argument('--task', type=str, required=True, help='Task type (e.g., stroop, gng)')
    parser.add_argument('--params', type=str, required=True, help='Path to parameter JSON file')
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Run simulation
    df = run_simulation(params)
    
    # Create output directory
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("results", args.task, ts)
    os.makedirs(outdir, exist_ok=True)
    
    # Save results
    df.to_csv(os.path.join(outdir, "raw.csv"), index=False)
    summary = df.groupby("trial").agg(["mean", "std"])
    summary.to_csv(os.path.join(outdir, "summary.csv"))

if __name__ == "__main__":
    main()
