# Filename: validate_wn_recovery_stroop.py
# Purpose: Perform Simulation-Based Calibration (SBC) for the w_n parameter
#          using a simplified Stroop-like task with graded conflict,
#          ABC-SMC, and summary statistics focused on error rates and RT quantiles.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pyabc
from pyabc import Distribution, RV, ABCSMC
from pyabc.sampler import MulticoreEvalParallelSampler, SingleCoreSampler
import time
import os
import tempfile
import traceback
import sys
import random
import json
import argparse
from pathlib import Path

def save_config(args, timestamp):
    """Save configuration to JSON file."""
    config = {
        'sim': {
            'w_s': FIXED_W_S,
            'threshold': SIM_THRESHOLD,
            'noise': SIM_NOISE_STD,
            't0': SIM_T0,
            'max_time': SIM_MAX_TIME,
            'conflict_levels': CONFLICT_LEVELS,
            'conflict_props': CONFLICT_PROPS
        },
        'abc': {
            'pop_size': ABC_POPULATION_SIZE,
            'max_gens': ABC_MAX_GENERATIONS,
            'min_eps': ABC_MIN_EPSILON,
            'distance': 'weighted_l2_v2'
        },
        'trials_per_sim': N_TRIALS_PER_SIM,
        'prior': {
            'w_n': [0.0, 2.5]
        },
        'seed': GLOBAL_SEED,
        'timestamp': timestamp
    }
    
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f'run_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Run SBC validation for w_n recovery.')
    parser.add_argument('seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('iterations', type=int, help='Number of SBC iterations')
    parser.add_argument('--dump-config', action='store_true',
                       help='Dump current configuration to JSON and exit')
    
    args = parser.parse_args()
    
    # Get timestamp for unique filenames
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save configuration
    save_config(args, timestamp)
    
    if args.dump_config:
        sys.exit(0)

# Set random seed for reproducibility
GLOBAL_SEED = seed
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Set parameters after parsing seed & sbc_iterations
GLOBAL_SEED = seed
N_SBC_SIMULATIONS = N_SBC_ITERATIONS  # Use the parsed value
N_TRIALS_PER_SIM = 300

# Task parameters
CONFLICT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONFLICT_PROPS = [0.2] * 5
FIXED_W_S = 0.7
SIM_THRESHOLD = 0.6
SIM_NOISE_STD = 0.40
SIM_T0 = 0.2
SIM_MAX_TIME = 4.0

# ABC settings
ABC_POPULATION_SIZE = 150
ABC_MAX_GENERATIONS = 8
ABC_MIN_EPSILON = 0.06
WN_PRIOR = Distribution(w_n=RV("uniform", 0, 2.5))

def normalize_summary_stats(stats_dict):
    """
    Normalize summary statistics to a standard scale.
    Error rates (0-1) are already on a good scale.
    RTs (0-3s) are normalized to 0-1 range.
    Count-based statistics (n_correct_*, n_error_*) are excluded from distance calculation.
    """
    normalized_stats = {}
    
    # Handle error rates (0-1 scale)
    for key, value in stats_dict.items():
        if 'error_rate' in key:
            normalized_stats[key] = value  # Already 0-1
            continue
        
        # Skip count-based statistics
        if 'n_' in key:
            normalized_stats[key] = value
            continue
        
        # Handle RTs and RT statistics
        if 'rt' in key:
            # For RTs (0-4s), normalize to 0-1 range
            if '_mean_' in key or '_25_' in key or '_50_' in key or '_75_' in key:
                normalized_stats[key] = value / SIM_MAX_TIME if not np.isnan(value) else np.nan
            # For RT standard deviations, normalize to 0-1 range
            elif '_std_' in key:
                # Impute 0.0 for NaN standard deviations
                std_value = value if not np.isnan(value) else 0.0
                # Normalize to 0-1 range
                normalized_stats[key] = std_value / SIM_MAX_TIME
            else:
                normalized_stats[key] = value
    
    return normalized_stats

def custom_distance_function(x, x_0):
    """
    Custom distance function that normalizes RTs to 0-1 range and uses weighted Euclidean distance.
    Handles NaN values by skipping them in the distance calculation.
    Weights are adjusted to down-weight extreme conflict levels (0.0 and 1.0).
    """
    x = normalize_summary_stats(x)
    x_0 = normalize_summary_stats(x_0)
    
    # Initialize weights (default weight is 1.0)
    weights = {key: 1.0 for key in x.keys()}
    
    # Down-weight extreme conflict levels (0.0 and 1.0)
    extreme_levels = ['0.0', '1.0']
    for lvl in extreme_levels:
        weights[f'error_rate_{lvl}'] *= 0.3
        weights[f'rt_mean_{lvl}'] *= 0.3
        weights[f'rt_std_{lvl}'] *= 0.3
    
    # Calculate weighted distance with small noise term
    dist = 0
    count = 0
    total_weight = 0
    
    for key in x.keys():
        if not np.isnan(x[key]) and not np.isnan(x_0[key]):
            weight = weights[key]
            # Add small noise term (1e-3) to prevent division by zero
            dist += weight * ((x[key] - x_0[key]) / (1.0 + 1e-3)) ** 2
            count += 1
            total_weight += weight
    
    if count == 0:
        return np.inf
    
    # Normalize by total weight instead of count
    return np.sqrt(dist / total_weight)

def simulate_ddm_trial(drift, threshold, noise_std, dt, t0, max_time):
    """
    Simulates a single trial of a two-choice DDM using Euler-Maruyama.
    """
    t = t0
    x = 0.0
    
    while t < max_time:
        # Decision process
        dx = drift * dt + noise_std * np.sqrt(dt) * np.random.randn()
        x += dx
        t += dt
        
        # Check boundaries
        if x >= threshold:
            return 1, t  # Correct choice
        elif x <= -threshold:
            return 0, t  # Error choice
    
    return np.nan, np.nan  # No decision within max_time

def generate_stroop_like_dataset(w_n_true, n_trials, conflict_levels, conflict_proportions):
    """
    Generates a dataset for the Stroop-like task for a given true w_n.
    Uses fixed task parameters.
    """
    # Fixed task parameters
    DT = 0.02  # Time step for simulation
    T0 = 0.2   # Non-decision time
    MAX_TIME = 4.0  # Maximum time for decision
    
    # Generate trial types according to proportions
    n_levels = len(conflict_levels)
    trial_types = np.random.choice(
        np.arange(n_levels),
        size=n_trials,
        p=conflict_proportions
    )
    
    # Initialize lists to store results
    choices = []
    rts = []
    conflict_levels_used = []
    
    for trial_type in trial_types:
        # Calculate drift based on conflict level
        conflict_level = conflict_levels[trial_type]
        drift = FIXED_W_S * (1.0 - conflict_level) - w_n_true * conflict_level
        
        # Simulate trial
        choice, rt = simulate_ddm_trial(
            drift=drift,
            threshold=SIM_THRESHOLD,
            noise_std=SIM_NOISE_STD,
            dt=DT,
            t0=SIM_T0,
            max_time=SIM_MAX_TIME
        )
        
        if not np.isnan(choice):  # Only include valid trials
            choices.append(choice)
            rts.append(rt)
            conflict_levels_used.append(conflict_level)
    
    # Create DataFrame
    df = pd.DataFrame({
        'choice': choices,
        'rt': rts,
        'conflict_level': conflict_levels_used
    })
    
    return df

def calculate_summary_statistics(df_trials):
    """
    Computes summary statistics (error rates and RT quantiles per conflict level).
    """
    stats = {}
    
    # Fixed task parameters
    DT = 0.02  # Time step for simulation
    T0 = 0.2   # Non-decision time
    MAX_TIME = 4.0  # Maximum time for decision
    
    # Group by conflict level
    grouped = df_trials.groupby('conflict_level')
    
    # For each conflict level, calculate error rate and RT quantiles
    for level, group in grouped:
        # Error rate (proportion of incorrect choices)
        n_trials = len(group)
        n_errors = (group['choice'] == 0).sum()
        error_rate = n_errors / n_trials if n_trials > 0 else np.nan
        
        # RT statistics (only for correct trials)
        correct_rts = group[group['choice'] == 1]['rt']
        default_rt = SIM_MAX_TIME + SIM_T0
        if len(correct_rts) > 0:
            rt_mean = correct_rts.mean()
            if len(correct_rts) >= 2:
                rt_std = correct_rts.std(ddof=1)
            else:
                rt_std = 0.0
            # Calculate quantiles as a Series
            rt_quantiles = correct_rts.quantile([0.25, 0.5, 0.75])
        else:
            rt_mean = rt_std = default_rt
            for qk in ['rt25', 'rt50', 'rt75']:
                stats[f'{qk}_{level}'] = default_rt
        
        # Store statistics
        stats[f'error_rate_{level}'] = error_rate
        stats[f'rt_mean_{level}'] = rt_mean
        stats[f'rt_std_{level}'] = rt_std
        if len(correct_rts) > 0:
            stats[f'rt25_{level}'] = rt_quantiles[0.25]
            stats[f'rt50_{level}'] = rt_quantiles[0.5]
            stats[f'rt75_{level}'] = rt_quantiles[0.75]
        
        # Count statistics (for debugging)
        stats[f'n_correct_{level}'] = (group['choice'] == 1).sum()
        stats[f'n_error_{level}'] = n_errors
        stats[f'n_total_{level}'] = n_trials
    
    return stats

def model_for_abc(params_from_abc):
    """
    The model function pyabc calls. Takes parameters sampled from the prior,
    runs the simulation, and returns summary statistics.
    """
    w_n = params_from_abc['w_n']
    
    # Generate dataset
    df_sim = generate_stroop_like_dataset(
        w_n_true=w_n,
        n_trials=N_TRIALS_PER_SIM,
        conflict_levels=CONFLICT_LEVELS,
        conflict_proportions=CONFLICT_PROPS
    )
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(df_sim)
    
    return summary_stats
def calculate_sbc_rank(posterior_samples, true_value):
    """
    Calculates the rank of the true value within the posterior samples (0 to N).
    """
    if len(posterior_samples) == 0:
        return np.nan
    
    # Count how many posterior samples are less than or equal to the true value
    rank = np.sum(posterior_samples <= true_value)
    return rank

def plot_sbc_histogram(ranks, n_posterior_samples, parameter_name="w_n"):
    """
    Plots the SBC rank histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=range(0, n_posterior_samples + 2),
            edgecolor='black', alpha=0.7)
    plt.title('SBC Rank Histogram')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'sbc_rank_histogram_{parameter_name}.png')
    plt.close()

# --- Main SBC Validation Loop ---
def main():
    print("============================================================")
    print("Starting SBC Validation for w_n (Stroop-like Task)")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Fixed W_S: {FIXED_W_S}, Threshold: {SIM_THRESHOLD}, Noise: {SIM_NOISE_STD}")
    print(f"SBC Simulations: {N_SBC_SIMULATIONS}, Trials per Sim: {N_TRIALS_PER_SIM}")
    print(f"Conflict Levels: {CONFLICT_LEVELS}, Proportions: {CONFLICT_PROPS}")
    print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Gens={ABC_MAX_GENERATIONS}, Min Eps={ABC_MIN_EPSILON}")
    print(f"Prior for w_n: {WN_PRIOR}")
    print("============================================================")
    print(f"Starting {N_SBC_SIMULATIONS} simulations with seed {GLOBAL_SEED}...")
    print("")

    # Create single core sampler (more Windows-compatible)
    sampler = SingleCoreSampler()

    # List to store results
    sbc_results_list = []

    # Start timer for overall SBC process
    start_time = time.time()

    # --- Loop through each SBC iteration ---
    for i in range(N_SBC_SIMULATIONS):
        print(f"\n--------------------------------------------------")
        print(f"Running SBC Iteration {i+1}/{N_SBC_SIMULATIONS}")
        
        # Create iteration-specific seed
        sbc_iteration_seed = GLOBAL_SEED + i
        
        # 1. DRAW TRUE w_n FROM PRIOR
        true_wn = np.random.uniform(0, 2.0)  # Draw from same prior as ABC
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        # 2. GENERATE 'OBSERVED' DATA
        print(f"  Step 2: Generating 'observed' data (N={N_TRIALS_PER_SIM})...")
        observed_data = generate_stroop_like_dataset(
            w_n_true=true_wn,
            n_trials=N_TRIALS_PER_SIM,
            conflict_levels=CONFLICT_LEVELS,
            conflict_proportions=CONFLICT_PROPS
        )
        
        # Calculate summary statistics
        observed_summary_stats = calculate_summary_statistics(observed_data)
        
        # Print normalized summary stats for debugging only
        if i == 0:
            normalized_stats = normalize_summary_stats(observed_summary_stats)
            print("\nNormalized summary statistics (example from first iteration):")
            for key, value in normalized_stats.items():
                print(f"  {key}: {value:.4f}")
            print("\n")
            normalized_stats = normalize_summary_stats(observed_summary_stats)
            print("\nNormalized summary statistics (example from first iteration):")
            for key, value in normalized_stats.items():
                print(f"  {key}: {value:.4f}")
            print("\n")
        
        # Create a random number generator seeded with our iteration seed
        rng = np.random.default_rng(sbc_iteration_seed)
        
        # Set default values in case ABC blows up
        posterior_mean = np.nan
        sbc_rank = np.nan
        n_posterior = 0
        
        # Create a temporary directory for database files
        db_dir = os.path.join(os.getcwd(), f'tmp_sbc_{sbc_iteration_seed}')
        os.makedirs(db_dir, exist_ok=True)
        
        try:
            # Create database file path
            db_filename = f"sbc_iter_{sbc_iteration_seed}.db"
            db_path = os.path.join(db_dir, db_filename)
            
            # Initialize ABCSMC with the sampler
            abc = ABCSMC(
                models=model_for_abc,         # The simulator function
                parameter_priors=WN_PRIOR,    # Prior for w_n (MUST match step 1)
                distance_function=custom_distance_function,
                population_size=ABC_POPULATION_SIZE,
                sampler=sampler
            )
            
            # Initialize the database with proper SQLite path
            # Pass raw summary stats to ABC for consistent normalization
            abc.new(f'sqlite:///{db_path}', observed_summary_stats)
            
            # Run ABC-SMC
            print(f"\nStarting ABC-SMC with min_epsilon={ABC_MIN_EPSILON} and max_gens={ABC_MAX_GENERATIONS}")
            start_time_sbc = time.time()
            
            try:
                # Run ABC and get history
                abc.run(
                    minimum_epsilon=ABC_MIN_EPSILON,
                    max_nr_populations=ABC_MAX_GENERATIONS
                )
                history = abc.history
                
                # Always calculate rank, even if history is empty
                sbc_rank = np.nan
                n_posterior = 0
                
                # Extract posterior and calculate rank if history is valid
                if history and history.max_t >= 0:
                    df_posterior, _ = history.get_distribution(m=0, t=history.max_t)
                    if not df_posterior.empty and 'w_n' in df_posterior.columns:
                        posterior_samples = df_posterior['w_n'].values
                        n_posterior = len(posterior_samples)
                        posterior_mean = posterior_samples.mean()
                        sbc_rank = calculate_sbc_rank(posterior_samples, true_wn)
                        print(f"    True value: {true_wn:.4f}, Posterior mean: {posterior_mean:.4f}, Rank: {sbc_rank} (out of {n_posterior} samples)")
                    else:
                        print("    WARNING: Posterior distribution is empty or 'w_n' column missing!")
                        posterior_mean = np.nan
                        n_posterior = 0
                else:
                    print("    WARNING: ABC history is empty or invalid.")
                    posterior_mean = np.nan
                    n_posterior = 0

            except Exception as e:
                print(f"    ERROR during ABC run or result extraction for true_wn={true_wn:.4f}:")
                traceback.print_exc()
                sbc_rank = np.nan
                n_posterior = 0
            finally:
                # Force close any database connections
                try:
                    if hasattr(abc, 'history'):
                        abc.history._session.close()
                        abc.history._engine.dispose()
                except:
                    pass
                
                # Always append results
                sbc_results_list.append({
                    'true_w_n': true_wn,
                    'posterior_mean': posterior_mean,
                    'sbc_rank': sbc_rank,
                    'n_posterior_samples': n_posterior
                })
                end_time_sbc = time.time()
                print(f"  Finished SBC Iteration {i+1} in {end_time_sbc - start_time_sbc:.1f} sec.")

        finally:
            # Clean up database directory
            try:
                if os.path.exists(db_dir):
                    for file in os.listdir(db_dir):
                        os.remove(os.path.join(db_dir, file))
                    os.rmdir(db_dir)
            except Exception as e:
                print(f"    WARNING: Error cleaning up database directory: {str(e)}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(sbc_results_list)
    
    # Get timestamp for unique filenames
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Create results directory if it doesn't exist
    results_dir = Path('wn_sbc_results')
    results_dir.mkdir(exist_ok=True)
    
    # Save results with timestamp in results directory
    results_filename = results_dir / f"stroop_sbc_results_seed{GLOBAL_SEED}_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nSaved results to: {results_filename}")

    # Plot SBC histogram
    plot_sbc_histogram(
        ranks=results_df['sbc_rank'].values,
        n_posterior_samples=ABC_POPULATION_SIZE,
        parameter_name="w_n"
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean SBC rank: {results_df['sbc_rank'].mean():.2f}")
    print(f"Mean posterior mean: {results_df['posterior_mean'].mean():.4f}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("============================================================")

if __name__ == "__main__":
    main()
end_time = time.time()
print(f"\nSBC validation script (Stroop-like task) finished.")
print("============================================================")
