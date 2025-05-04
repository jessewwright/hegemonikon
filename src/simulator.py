# Filename: src/simulator.py
# Purpose: Runs the Go/No-Go simulation trial loop using MVNES agent.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import os
import sys
import random

# Add parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Try to import config and agent, fall back to default parameters if not available
try:
    from src.config import *
except ImportError:
    print("Warning: Could not import from config. Using default parameters.")
    # Default parameters
    W_S = 1.0
    W_N = 1.0
    THRESHOLD_A = 1.0
    T_NONDECISION = 0.1
    NOISE_STD_DEV = 1.0
    DT = 0.01
    MAX_TIME = 2.0
    N_TRIALS = 200
    P_GO_TRIAL = 0.5
    AFFECT_STRESS_THRESHOLD_REDUCTION = -0.3

try:
    from src.agent_mvnes import MVNESAgent
except ImportError:
    print("Warning: Could not import MVNESAgent from src.agent_mvnes. Defining dummy class.")
    class MVNESAgent:
        def __init__(self, config):
            self.config = config
        
        def run_mvnes_trial(self, salience_input, norm_input, params):
            return {'choice': 0, 'rt': 1.0, 'trace': [0.0], 'timeout': False}

# Constants for affect modulation (already defined in default parameters)

# --- Main Simulation Function ---
def run_simulation(n_trials=N_TRIALS, p_go=P_GO_TRIAL, custom_params=None):
    """
    Runs a simulation of Go/No-Go trials using the MVNES agent.
    
    Args:
        n_trials (int): Number of trials to simulate
        p_go (float): Probability of a trial being a Go trial (0-1)
        custom_params (dict): Optional dictionary of custom parameters to override defaults
    
    Returns:
        pd.DataFrame: DataFrame containing trial results
    """
    print("\nRunning MVNES Go/No-Go Simulation")
    print("-" * 50)
    
    # Initialize custom params if None
    if custom_params is None:
        custom_params = {}
    
    # Combine default and custom parameters
    sim_params = {
        'w_s': W_S, 'w_n': W_N, 'threshold_a': THRESHOLD_A, 't': T_NONDECISION,
        'noise_std_dev': NOISE_STD_DEV, 'dt': DT, 'max_time': MAX_TIME
    }
    sim_params.update(custom_params)

    # Check for stress condition and adjust threshold
    if 'affect_stress' in sim_params and sim_params['affect_stress']:
        sim_params['threshold_a'] += AFFECT_STRESS_THRESHOLD_REDUCTION

    print(f"\nRunning simulation with parameters: {sim_params}")
    print(f"Trials: {n_trials}, P(Go): {p_go}")

    # Initialize the agent
    agent = MVNESAgent(sim_params)
    
    # Initialize results list
    results = []
    start_time = time.time()
    
    # Run trials
    for trial_idx in range(n_trials):
        # Determine trial type
        is_go_trial = random.random() < p_go
        trial_type = 'go' if is_go_trial else 'nogo'
        
        # Set inputs based on trial type
        if is_go_trial:
            # Go trial: strong salience push, no norm inhibition
            salience_input = 1.0
            norm_input = 0.0
        else:
            # NoGo trial: weak salience push, strong norm inhibition
            salience_input = 0.1  # Small residual go impulse
            norm_input = 1.0
        
        # Run trial through agent
        trial_result = agent.run_mvnes_trial(salience_input, norm_input, sim_params)
        
        # Determine outcome
        if is_go_trial:
            if trial_result['choice'] == 1:
                outcome = 'Hit'
            else:
                outcome = 'Miss'
        else:
            if trial_result['choice'] == 1:
                outcome = 'False Alarm'
            else:
                outcome = 'Correct Rejection'
        
        # Store results
        results.append({
            'trial': trial_idx,
            'stimulus': trial_type,
            'outcome': outcome,
            'choice': trial_result['choice'],
            'rt': trial_result['rt'],
            'trace': trial_result['trace'],
            'timeout': trial_result['timeout'],
            **sim_params  # Include all parameters used
        })
    
    # Calculate simulation time
    end_time = time.time()
    sim_time = end_time - start_time
    print(f"\nSimulation finished in {sim_time:.2f} seconds.")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    print("\n--- Simulation Results Summary ---")
    print(f"Total Trials: {n_trials}")
    
    # Overall performance
    print("\nOverall Performance:")
    print(results_df['outcome'].value_counts(normalize=True))
    
    # Calculate rates
    go_trials = results_df[results_df['stimulus'] == 'go']
    nogo_trials = results_df[results_df['stimulus'] == 'nogo']
    
    hit_rate = (go_trials['outcome'] == 'Hit').mean()
    miss_rate = (go_trials['outcome'] == 'Miss').mean()
    fa_rate = (nogo_trials['outcome'] == 'False Alarm').mean()
    cr_rate = (nogo_trials['outcome'] == 'Correct Rejection').mean()
    
    print("\nPerformance Metrics:")
    print(f"Hit Rate (Go Trials): {hit_rate:.3f}")
    print(f"Miss Rate (Go Trials): {miss_rate:.3f}")
    print(f"False Alarm Rate (NoGo Trials): {fa_rate:.3f}")
    print(f"Correct Rejection Rate (NoGo Trials): {cr_rate:.3f}")
    
    # Calculate RT statistics
    hits = results_df[results_df['outcome'] == 'Hit']
    fas = results_df[results_df['outcome'] == 'False Alarm']
    
    # Calculate mean RTs with safe handling for empty dataframes
    mean_rt_hit = hits['rt'].mean() if not hits.empty else None
    mean_rt_fa = fas['rt'].mean() if not fas.empty else None
    
    print("\nRT Statistics:")
    if mean_rt_hit is not None:
        print(f"Mean RT Hits: {mean_rt_hit:.3f} s")
    else:
        print("Mean RT Hits: N/A")
    
    if mean_rt_fa is not None:
        print(f"Mean RT False Alarms: {mean_rt_fa:.3f} s")
    else:
        print("Mean RT False Alarms: N/A")
    
    # Save results with unique filename based on parameters
    param_str = f"wn_{sim_params['w_n']}_ws_{sim_params['w_s']}"
    results_dir = Path(__file__).parent.parent / 'data'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'mvnes_gng_results_{param_str}.csv'
    results_df.to_csv(results_file, index=False, float_format='%.4f')
    print(f"\nResults saved to {results_file}")
    
    return results_df

    results = []
    start_time = time.time()

    for i in range(n_trials):
        # Determine trial type
        is_go_trial = random.random() < p_go
        stimulus_type = 'go' if is_go_trial else 'nogo'

        # Set inputs based on trial type
        # Go trial: Salience drives action (S=1), No norm inhibition (N=0)
        # NoGo trial: Salience might still push for action (e.g., S=1 or S=0.5?),
        #             Norm provides inhibition (N=1) -> effective drift = w_s*S - w_n*N
        # Let's assume S=1 for Go cue presence, N=1 for NoGo cue presence
        salience_input = 1.0 if is_go_trial else 1.0 # Assume Go drive is present on both
        norm_input = 0.0 if is_go_trial else 1.0    # Norm inhibition only active on NoGo

        # Run the MVNES trial simulation
        trial_output = run_mvnes_trial(salience_input, norm_input, sim_params)

        # Determine outcome based on trial type and choice (0=Inhibit, 1=Go)
        outcome = "Unknown"
        if is_go_trial:
            if trial_output['choice'] == 1: outcome = "Hit"
            else: outcome = "Miss" # Timed out or inhibited Go
        else: # NoGo trial
            if trial_output['choice'] == 1: outcome = "False Alarm"
            else: outcome = "Correct Rejection" # Inhibited NoGo

        results.append({
            'trial': i + 1,
            'stimulus': stimulus_type,
            'choice': trial_output['choice'], # 0 or 1
            'outcome': outcome,
            'rt': trial_output['rt'],
            'timeout': trial_output['timeout']
            # Optionally log parameters used if they change dynamically later
            # 'threshold_a': sim_params['threshold_a'],
            # 'w_n': sim_params['w_n']
        })

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    return pd.DataFrame(results)

# --- Example Execution ---
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MVNES Go/No-Go Simulation')
    parser.add_argument('--normal', action='store_true', help='Run normal trials')
    parser.add_argument('--stress', action='store_true', help='Run stress trials')
    parser.add_argument('--w_n', type=float, help='Norm weight parameter')
    parser.add_argument('--w_s', type=float, help='Salience weight parameter')
    args = parser.parse_args()
    
    # Create custom parameters dictionary with default values
    custom_params = {
        'w_s': W_S,  # Use default value from config
        'w_n': W_N,  # Use default value from config
        'threshold_a': THRESHOLD_A,
        't': T_NONDECISION,
        'noise_std_dev': NOISE_STD_DEV,
        'dt': DT,
        'max_time': MAX_TIME,
        'affect_stress': args.stress,  # Add stress flag
        'stress_threshold_reduction': AFFECT_STRESS_THRESHOLD_REDUCTION  # Add stress threshold reduction
    }
    
    # Override parameters from command line if specified
    if args.w_n is not None:
        custom_params['w_n'] = args.w_n
    if args.w_s is not None:
        custom_params['w_s'] = args.w_s
    
    # Run the simulation with custom parameters
    results_df = run_simulation(custom_params=custom_params)

    print("\n--- Simulation Results Summary ---")
    print(f"Total Trials: {len(results_df)}")

    # Calculate overall performance metrics
    print("\nOverall Performance:")
    print(results_df['outcome'].value_counts(normalize=True).round(3))

    # Calculate specific rates
    go_trials = results_df[results_df['stimulus'] == 'go']
    nogo_trials = results_df[results_df['stimulus'] == 'nogo']

    hit_rate = (go_trials['outcome'] == 'Hit').mean() if not go_trials.empty else 0
    miss_rate = (go_trials['outcome'] == 'Miss').mean() if not go_trials.empty else 0
    fa_rate = (nogo_trials['outcome'] == 'False Alarm').mean() if not nogo_trials.empty else 0
    cr_rate = (nogo_trials['outcome'] == 'Correct Rejection').mean() if not nogo_trials.empty else 0

    print(f"\nHit Rate (Go Trials): {hit_rate:.3f}")
    print(f"Miss Rate (Go Trials): {miss_rate:.3f}")
    print(f"False Alarm Rate (NoGo Trials): {fa_rate:.3f}")
    print(f"Correct Rejection Rate (NoGo Trials): {cr_rate:.3f}")

    # Calculate mean RT for Hits and False Alarms
    mean_rt_hit = go_trials[go_trials['outcome'] == 'Hit']['rt'].mean()
    mean_rt_fa = nogo_trials[nogo_trials['outcome'] == 'False Alarm']['rt'].mean()

    print(f"\nMean RT Hits: {mean_rt_hit:.3f} s" if pd.notna(mean_rt_hit) else "Mean RT Hits: N/A")
    print(f"Mean RT False Alarms: {mean_rt_fa:.3f} s" if pd.notna(mean_rt_fa) else "Mean RT False Alarms: N/A")

    # Save results with unique filename based on parameters
    param_str = f"wn_{custom_params.get('w_n', 'default')}_ws_{custom_params.get('w_s', 'default')}"
    # Get absolute path to data directory in the project root
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_filename = os.path.join(script_dir, '..', 'data', f'mvnes_gng_results_{param_str}.csv')
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        results_df.to_csv(output_filename, index=False, float_format='%.4f')
        print(f"\nResults saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results: {e}")
