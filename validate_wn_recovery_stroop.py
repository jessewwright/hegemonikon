# Filename: validate_wn_recovery_stroop.py
# Purpose: Perform Simulation-Based Calibration (SBC) for the w_n parameter
#          using a simplified Stroop-like task with graded conflict,
#          ABC-SMC, and summary statistics focused on error rates and RT quantiles.

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pyabc # Ensure this is imported
from pyabc import Distribution, RV, ABCSMC, MulticoreEvalParallelSampler, SingleCoreSampler
import time
import os
import tempfile
import logging
import traceback
import multiprocessing

# Set multiprocessing context for Windows
multiprocessing.set_start_method('spawn')

# --- 1. Robust Imports & Dependency Checks ---
# No local 'src' imports needed for this standalone version

# Configure logging for pyabc (optional, reduces console noise)
logging.getLogger("pyabc").setLevel(logging.WARNING)

# --- 2. Configuration ---

# Reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED) # For python's random if used elsewhere

# Task / Simulation Parameters
N_TRIALS_PER_DATASET = 300      # Trials per synthetic dataset in each SBC iteration
# Conflict Levels: 0=Congruent (easy positive drift), 0.5=Neutral (moderate positive drift), 1=Incongruent (conflict, drift depends on w_n)
CONFLICT_LEVELS = [0.0, 0.5, 1.0]
CONFLICT_PROPORTIONS = [0.3333, 0.3333, 0.3334]  # Balanced proportions that sum to 1.0
if not np.isclose(sum(CONFLICT_PROPORTIONS), 1.0):
    raise ValueError("Conflict proportions must sum to 1.0")

# DDM Parameters for the Simulator
SIM_THRESHOLD  = 1.0      # DDM boundary separation 'a'
SIM_NOISE_STD  = 0.25     # Standard deviation of Wiener process noise
SIM_DT         = 0.01     # Time step for Euler-Maruyama
SIM_T0         = 0.1      # Non-decision time
SIM_MAX_TIME   = 3.0      # Max time for a trial before timeout

# Fixed Model Parameters (for NES-like interpretation)
# W_S will determine the base positive drift on congruent/neutral trials
FIXED_W_S = 0.7           # Salience weight (effective positive drift component)
# We will recover W_N only (norm weight / conflict modulation strength)

# SBC / ABC Settings
N_SBC_ITERATIONS = 5    # Running 5 iterations for testing
ABC_POPULATION_SIZE = 100 # Number of particles per ABC generation
ABC_MAX_GENERATIONS = 4   # Number of generations to run
ABC_MIN_EPSILON = 0.05    # Minimum epsilon (can be lowered to 0.02 if needed)

# Prior for w_n (the parameter to be recovered)
# This prior MUST be used for BOTH drawing true values AND in the ABCSMC setup
WN_PRIOR_DIST = RV("uniform", 0, 2.0) # Example: Uniform between 0 and 2
WN_PRIOR = Distribution(w_n=WN_PRIOR_DIST)


# --- 3. Core Functions ---

def normalize_summary_stats(stats_dict):
    """
    Normalize summary statistics to a standard scale.
    Error rates (0-1) are already on a good scale.
    RTs (0-3s) are normalized to 0-1 range.
    Count-based statistics (n_correct_*, n_error_*) are excluded from distance calculation.
    """
    normalized_stats = {}
    for key, value in stats_dict.items():
        if 'err_' in key:  # Error rates are already on 0-1 scale
            normalized_stats[key] = value
        elif 'rt' in key:  # RTs are in seconds (0-3s)
            if value is not None and not np.isnan(value):
                normalized_stats[key] = value / SIM_MAX_TIME  # Normalize to 0-1 range
            else:
                normalized_stats[key] = value  # Keep NaN values as is
        elif 'n_' in key:  # Skip count-based statistics
            continue
    return normalized_stats

def custom_distance_function(x, x_0):
    """
    Custom distance function that normalizes RTs to 0-1 range and uses Euclidean distance.
    Handles NaN values by skipping them in the distance calculation.
    """
    # Normalize both summaries
    x_norm = normalize_summary_stats(x)
    x_0_norm = normalize_summary_stats(x_0)
    
    # Calculate Euclidean distance between normalized summaries
    dist = 0.0
    n_valid_pairs = 0
    
    # Iterate over normalized items directly
    for key, val in x_norm.items():
        if not (np.isnan(val) or np.isnan(x_0_norm[key])):
            dist += (val - x_0_norm[key]) ** 2
            n_valid_pairs += 1
    
    if n_valid_pairs == 0:
        return float('inf')  # If no valid pairs, return infinite distance
    
    return np.sqrt(dist)

def simulate_ddm_trial(drift, threshold, noise_std, dt, t0, max_time):
    """Simulates a single trial of a two-choice DDM using Euler-Maruyama."""
    evidence = 0.0
    current_time = 0.0
    # Max decision time
    max_decision_time = max_time - t0
    if max_decision_time <= 0: # Should not happen with reasonable t0
        return 1 if drift > 0 else 0, t0 # Minimal choice based on drift

    noise_scaling_factor = noise_std * np.sqrt(dt)

    while current_time < max_decision_time:
        noise_sample = np.random.randn() # Standard normal
        evidence += drift * dt + noise_scaling_factor * noise_sample
        current_time += dt

        if evidence >= threshold:
            return 1, t0 + current_time   # Upper boundary (e.g., "correct" or "dominant")
        elif evidence <= -threshold:
            return 0, t0 + current_time   # Lower boundary (e.g., "error" or "alternative")

    # Timeout: assign choice based on sign of final evidence
    # This is a common way to handle timeouts in DDM
    choice = 1 if evidence >= 0 else 0
    return choice, max_time # Return max_time as RT for timeouts

def generate_stroop_like_dataset(w_n_true, n_trials, conflict_levels, conflict_proportions):
    """
    Generates a dataset for the Stroop-like task for a given true w_n.
    Uses fixed W_S, THRESHOLD, NOISE_STD, DT, T0, MAX_TIME.
    """
    trial_data_list = []
    for _i in range(n_trials):
        # Determine conflict level for this trial based on proportions
        # Note: np.random.choice can also be used here for efficiency if many levels
        rand_val = np.random.rand()
        cumulative_prop = 0.0
        current_conflict_level = conflict_levels[-1] # Default to last if something goes wrong
        for level, prop in zip(conflict_levels, conflict_proportions):
            cumulative_prop += prop
            if rand_val < cumulative_prop:
                current_conflict_level = level
                break
        
        # Calculate drift rate for this trial
        # Congruent (level=0):   drift = W_S * 1.0 - w_n_true * 0.0 = W_S (strong positive)
        # Neutral (level=0.5):   drift = W_S * 0.5 - w_n_true * 0.5 (moderate positive or conflict)
        # Incongruent (level=1): drift = W_S * 0.0 - w_n_true * 1.0 = -w_n_true (strong negative, assuming correct is 'Go')
        # OR, simpler: let 'conflict_level' directly scale the opposing force
        # drift = FIXED_W_S - w_n_true * current_conflict_level
        # Let's use ChatGPT's suggestion: drift = W_S * (1.0 - lvl) - w_n * lvl
        # This makes:
        # lvl=0 (Congruent): drift = FIXED_W_S
        # lvl=0.5 (Neutral): drift = FIXED_W_S * 0.5 - w_n_true * 0.5 = 0.5 * (FIXED_W_S - w_n_true)
        # lvl=1 (Incongruent): drift = FIXED_W_S * 0.0 - w_n_true * 1.0 = -w_n_true
        # This seems reasonable. For "correct" = upper boundary (choice 1)
        
        drift = FIXED_W_S * (1.0 - current_conflict_level) - w_n_true * current_conflict_level

        choice, rt = simulate_ddm_trial(
            drift, SIM_THRESHOLD, SIM_NOISE_STD, SIM_DT, SIM_T0, SIM_MAX_TIME
        )
        trial_data_list.append({
            'conflict_level': current_conflict_level,
            'choice': choice, # 1 for upper, 0 for lower
            'rt': rt
        })
    return pd.DataFrame(trial_data_list)

def calculate_summary_statistics(df_trials):
    """
    Computes summary statistics (error rates and RT quantiles per conflict level).
    """
    stats = {}
    # Define default NaN values for all expected keys first
    for lvl_val in CONFLICT_LEVELS:
        lvl_key_suffix = int(lvl_val * 100) # e.g., 0, 50, 100
        stats[f'err_{lvl_key_suffix}'] = np.nan
        stats[f'rt25c_{lvl_key_suffix}'] = np.nan # 'c' for correct
        stats[f'rt50c_{lvl_key_suffix}'] = np.nan
        stats[f'rt75c_{lvl_key_suffix}'] = np.nan
        stats[f'rt25e_{lvl_key_suffix}'] = np.nan # 'e' for error
        stats[f'rt50e_{lvl_key_suffix}'] = np.nan
        stats[f'rt75e_{lvl_key_suffix}'] = np.nan
        stats[f'n_correct_{lvl_key_suffix}'] = 0
        stats[f'n_error_{lvl_key_suffix}'] = 0


    for lvl_val in CONFLICT_LEVELS:
        lvl_key_suffix = int(lvl_val * 100)
        subset_df = df_trials[df_trials['conflict_level'] == lvl_val]

        if subset_df.empty:
            continue # Leave stats as NaN if no trials for this level

        choices = subset_df['choice'].values
        rts = subset_df['rt'].values

        # Error rate (assuming choice=1 is "correct" for DDM upper boundary)
        error_rate = 1.0 - np.mean(choices) if len(choices) > 0 else np.nan
        stats[f'err_{lvl_key_suffix}'] = error_rate

        correct_rts = rts[choices == 1]
        error_rts = rts[choices == 0]
        
        stats[f'n_correct_{lvl_key_suffix}'] = len(correct_rts)
        stats[f'n_error_{lvl_key_suffix}'] = len(error_rts)

        if len(correct_rts) > 0:
            stats[f'rt25c_{lvl_key_suffix}'] = np.nanpercentile(correct_rts, 25)
            stats[f'rt50c_{lvl_key_suffix}'] = np.nanpercentile(correct_rts, 50)
            stats[f'rt75c_{lvl_key_suffix}'] = np.nanpercentile(correct_rts, 75)
        # else already NaN

        if len(error_rts) > 0:
            stats[f'rt25e_{lvl_key_suffix}'] = np.nanpercentile(error_rts, 25)
            stats[f'rt50e_{lvl_key_suffix}'] = np.nanpercentile(error_rts, 50)
            stats[f'rt75e_{lvl_key_suffix}'] = np.nanpercentile(error_rts, 75)
        # else already NaN
    return stats

def model_for_abc(params_from_abc):
    """
    The model function pyabc calls. Takes parameters sampled from the prior,
    runs the simulation, and returns summary statistics.
    """
    # params_from_abc will be a dict like {'w_n': value}
    w_n_current = params_from_abc['w_n']
    df_simulated = generate_stroop_like_dataset(
        w_n_current, N_TRIALS_PER_DATASET, CONFLICT_LEVELS, CONFLICT_PROPORTIONS
    )
    return calculate_summary_statistics(df_simulated)

def calculate_sbc_rank(posterior_samples, true_value):
    """Calculates the rank of the true value within the posterior samples (0 to N)."""
    samples = np.asarray(posterior_samples)
    valid_samples = samples[~np.isnan(samples)]
    if len(valid_samples) == 0: return np.nan
    rank = np.sum(valid_samples < true_value)
    return rank

def plot_sbc_histogram(ranks, n_posterior_samples, parameter_name="w_n"):
    """Plots the SBC rank histogram."""
    valid_ranks = np.asarray([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print("No valid SBC ranks found to plot.")
        return

    n_sbc_runs = len(valid_ranks)
    n_outcomes = n_posterior_samples + 1 # Ranks can be 0 to n_posterior_samples

    plt.figure(figsize=(10, 6))
    # Determine number of bins, e.g., n_outcomes or fewer for clarity
    n_bins_plot = min(n_outcomes, 25) # Cap at 25 bins for visual clarity

    plt.hist(valid_ranks, bins=n_bins_plot, range=(-0.5, n_posterior_samples + 0.5),
             density=True, alpha=0.75, color='skyblue', edgecolor='black',
             label=f'Observed Ranks (N={n_sbc_runs})')

    expected_density = 1.0 / n_outcomes
    plt.axhline(expected_density, color='red', linestyle='--',
                label=f'Uniform (Exp. Density ≈ {expected_density:.3f})')

    plt.xlabel(f"Rank of True {parameter_name} (0-{n_posterior_samples})", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"SBC Rank Histogram for {parameter_name} (Stroop-like Task)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.xlim(-0.5, n_posterior_samples + 0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    # Ensure 'plots' directory exists
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / f"sbc_hist_{parameter_name}_stroop.png")
    print(f"SBC rank histogram saved to plots/sbc_hist_{parameter_name}_stroop.png")
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    print("="*60)
    print("Starting SBC Validation for w_n (Stroop-like Task)")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Fixed W_S: {FIXED_W_S}, Threshold: {SIM_THRESHOLD}, Noise: {SIM_NOISE_STD}")
    print(f"SBC Iterations: {N_SBC_ITERATIONS}, Trials per Dataset: {N_TRIALS_PER_DATASET}")
    print(f"Conflict Levels: {CONFLICT_LEVELS}, Proportions: {CONFLICT_PROPORTIONS}")
    print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Gens={ABC_MAX_GENERATIONS}, Min Eps={ABC_MIN_EPSILON}")
    print(f"Prior for w_n: {WN_PRIOR}")
    print("="*60)

    sbc_results_list = []
    # Distance function: Using custom weighted_distance.
    distance_function = custom_distance_function

    # --- Loop through each SBC iteration ---
    for i in range(N_SBC_ITERATIONS):
        print("\n" + "-" * 50)
        print(f"Running SBC Iteration {i+1}/{N_SBC_ITERATIONS}")
        sbc_iteration_seed = GLOBAL_SEED + i + 1 # Ensure each SBC iteration is reproducible
        np.random.seed(sbc_iteration_seed) # Reseed for drawing true_wn and for DDM noise

        # 1. DRAW TRUE PARAMETER FROM PRIOR
        true_wn_dict = WN_PRIOR.rvs() # pyabc returns a dict
        true_wn = true_wn_dict['w_n']
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        # 2. GENERATE "OBSERVED" DATA using true_wn
        print(f"  Step 2: Generating 'observed' data (N={N_TRIALS_PER_DATASET})...")
        df_observed_data = generate_stroop_like_dataset(
            true_wn, N_TRIALS_PER_DATASET, CONFLICT_LEVELS, CONFLICT_PROPORTIONS
        )
        
        # Debug: Print some basic stats about the generated data
        print(f"    Generated data summary:")
        print(f"    Total trials: {len(df_observed_data)}")
        for level in CONFLICT_LEVELS:
            lvl_df = df_observed_data[df_observed_data['conflict_level'] == level]
            print(f"    Level {level}: {len(lvl_df)} trials, {lvl_df['choice'].mean():.2f} mean choice")
            print(f"    Level {level}: {lvl_df['rt'].mean():.2f} mean RT")
            print(f"    Level {level}: {lvl_df['rt'].std():.2f} RT std")
        
        observed_summary_stats = calculate_summary_statistics(df_observed_data)
        
        # Debug: Print summary stats before normalization
        print("\n    Raw summary statistics:")
        for key, value in observed_summary_stats.items():
            if np.isnan(value):
                print(f"    {key}: NaN")
            else:
                print(f"    {key}: {value:.4f}")
        
        # Check if essential observed stats are NaN, skip if so
        if np.isnan(observed_summary_stats.get(f'err_{int(CONFLICT_LEVELS[0]*100)}', np.nan)): # Check err_0
             print("\n    ERROR: Essential summary stat (e.g., error rate for congruent) is NaN. Skipping SBC iteration.")
             sbc_results_list.append({'true_w_n': true_wn, 'sbc_rank': np.nan, 'n_posterior_samples': 0})
             continue
        
        if i == 0:
            normalized_stats = normalize_summary_stats(observed_summary_stats)
            print("\nNormalized summary statistics (example from first iteration):")
            for key, value in normalized_stats.items():
                print(f"  {key}: {value:.4f}")
            print("\n")
            
        # Initialize the database with proper SQLite path
        # Pass raw summary stats to ABC for consistent normalization
        abc.new(f'sqlite:///{db_path}', observed_summary_stats)
        
        # Create a random number generator seeded with our iteration seed
        rng = np.random.default_rng(sbc_iteration_seed)
        
        # Use single-core processing to avoid multiprocessing issues
        n_procs = 1
        
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
                sampler=SingleCoreSampler(rng=np.random.default_rng(sbc_iteration_seed))
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
            except Exception as e:
                print(f"    ERROR during ABC run: {str(e)}")
                raise
            
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

                # Force close any database connections
                try:
                    if hasattr(abc, 'history'):
                        abc.history._session.close()
                        abc.history._engine.dispose()
                except:
                    pass
                
                # Always calculate rank, even if history is empty
                sbc_rank = np.nan
                n_posterior = 0
                
                # Extract posterior and calculate rank if history is valid
                if history and history.max_t >= 0:
                    df_posterior, _ = history.get_distribution(m=0, t=history.max_t)
                    if not df_posterior.empty and 'w_n' in df_posterior.columns:
                        posterior_samples = df_posterior['w_n'].values
                        n_posterior = len(posterior_samples)
                        sbc_rank = calculate_sbc_rank(posterior_samples, true_wn)
                        print(f"    True value: {true_wn:.4f}, Rank: {sbc_rank} (out of {n_posterior} samples)")
                    else:
                        print("    WARNING: Posterior distribution is empty or 'w_n' column missing!")
                else:
                    print("    WARNING: ABC history is empty or invalid.")

            except Exception as e:
                print(f"    ERROR during ABC run or result extraction for true_wn={true_wn:.4f}:")
                traceback.print_exc()
                sbc_rank = np.nan
                n_posterior = 0
            sbc_results_list.append({'true_w_n': true_wn, 'sbc_rank': sbc_rank, 'n_posterior_samples': n_posterior})
        finally:
            # Clean up database directory
            try:
                if os.path.exists(db_dir):
                    for file in os.listdir(db_dir):
                        file_path = os.path.join(db_dir, file)
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    os.rmdir(db_dir)
            except:
                pass
    print("="*60)

    if sbc_results_list:
        results_df = pd.DataFrame(sbc_results_list)
        # Create timestamped filename using global seed
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f"stroop_sbc_results_seed{GLOBAL_SEED}_{timestamp}.csv"
    # Save results to CSV with versioning
    results_df.to_csv(results_filename, index=False)
    print(f"\nSaved results to: {results_filename}")

    print(f"\nSBC Results Summary (Sample from {results_filename}):")
    print(pd.concat([results_df.head(), results_df.tail()]).to_string(float_format='%.4f'))

    valid_ranks = results_df['sbc_rank'].values[~np.isnan(results_df['sbc_rank'].values)]
    valid_ranks_count = len(valid_ranks)
    if valid_ranks_count > 0:
        # Use the target population size for N samples in plot title,
        # assuming most runs achieved this.
        plot_sbc_histogram(results_df['sbc_rank'].tolist(), ABC_POPULATION_SIZE)
    else:
        print("No valid ranks to plot.")

    if not sbc_results_list:
        print("No SBC results were obtained.")

    print("\nSBC validation script (Stroop-like task) finished.")
    print("="*60)