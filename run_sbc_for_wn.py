# Filename: run_sbc_for_wn.py
# Purpose: Perform Simulation-Based Calibration (SBC) for the w_n parameter
#          using the simplified task, ABC-SMC, and rich summary statistics.
#          This version incorporates best practices for robustness and reproducibility.

import sys
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import time
import os
import tempfile
from pathlib import Path
import logging
import traceback # For detailed error printing

# --- 1. Robust Imports & Dependency Checks ---
try:
    # Dynamically add 'src' to path based on script location
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    # Ensure the path is added only once and at the beginning
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agent_mvnes import MVNESAgent
    # Import necessary fixed parameters (or define defaults)
    try:
        # Use uppercase consistent with original config files
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2 # Using a more realistic noise value
        DT = 0.01
        MAX_TIME = 2.0 # Reduced max_time might speed up sims if timeouts are common

except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    print("Ensure this script is in the project root or adjust paths.")
    print("Make sure 'src/agent_mvnes.py' and 'src/agent_config.py' exist.")
    sys.exit(1)

try:
    import pyabc
    from pyabc import Distribution, RV, ABCSMC
    # Optional: Check pyabc version if specific features are needed
    # print(f"Using pyabc version: {pyabc.__version__}")
except ImportError:
    print("ERROR: pyabc library not found.")
    print("Please install it: pip install pyabc")
    sys.exit(1)

# Configure logging for pyabc (optional, reduces console noise)
logging.getLogger("pyabc").setLevel(logging.WARNING)

# --- 2. Configuration & Constants ---

# Reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# SBC Parameters
N_SBC_SIMULATIONS = 100   # Number of SBC iterations (e.g., 100 for initial check, 500+ for smooth plot)

# Simulation Parameters (per SBC iteration)
N_TRIALS_PER_SIM = 300    # Trials per synthetic dataset (Increase for more stable summaries)

# Fixed Model Parameters (MUST BE CONSISTENT with agent_mvnes internals)
# *** PLEASE DOUBLE-CHECK THESE VALUES MATCH YOUR INTENDED FIXED STATE ***
TRUE_A = 1.0            # Fixed threshold value
TRUE_W_S = 0.7          # Fixed salience weight
BASE_SIM_PARAMS = {
    't': T_NONDECISION,
    'noise_std_dev': NOISE_STD_DEV,
    'dt': DT,
    'max_time': MAX_TIME,
    # Add other params if run_mvnes_trial expects them
    'affect_stress_threshold_reduction': -0.3, # Default, not used unless affect_stress=True in params
    'veto_flag': False # Assuming default veto is off
}

# Simplified Task Parameters
P_HIGH_CONFLICT = 0.5   # Probability of a high-conflict trial
NEUTRAL_SALIENCE = 1.0
NEUTRAL_NORM = 0.0
CONFLICT_SALIENCE = 1.0 # Assume Go drive is present
CONFLICT_NORM = 1.0     # Norm opposes Go drive

# ABC Configuration
ABC_POPULATION_SIZE = 100 # Smaller for SBC checks
ABC_MAX_NR_POPULATIONS = 8 # Max generations
ABC_MIN_EPSILON = 0.005 # Target epsilon (might need tuning based on distances)

# Define PRIOR for w_n (MUST match the prior used IN the ABC run)
WN_PRIOR_DIST = RV("uniform", 0, 2.0) # Uniform prior for SBC check
WN_PRIOR = Distribution(w_n=WN_PRIOR_DIST)

# --- 3. Core Functions (Modular Structure) ---

def generate_trial_inputs(n_trials, p_conflict, seed=None):
    """Generates fixed arrays of salience and norm inputs for the simplified task."""
    rng = np.random.default_rng(seed) # Use separate generator for inputs
    salience_inputs = np.zeros(n_trials)
    norm_inputs = np.zeros(n_trials)
    for i in range(n_trials):
        if rng.random() < p_conflict:
            salience_inputs[i] = CONFLICT_SALIENCE
            norm_inputs[i] = CONFLICT_NORM
        else:
            salience_inputs[i] = NEUTRAL_SALIENCE
            norm_inputs[i] = NEUTRAL_NORM
    return salience_inputs, norm_inputs

def simulate_trials(params_dict, salience_inputs, norm_inputs):
    """
    Simulates N trials for a given parameter set using fixed inputs.
    Returns a DataFrame of trial results.
    Ensures all required parameters for agent.run_mvnes_trial are present.
    """
    n_sim_trials = len(salience_inputs)
    results_list = []
    # Agent should ideally be stateless for this function call
    # If agent has internal state that persists, need to reinstantiate per simulation
    agent = MVNESAgent(config={}) # Assuming stateless for run_mvnes_trial

    # Combine base params with current sampled w_n and fixed a, w_s
    # Ensure all necessary parameters for run_mvnes_trial are present
    params_for_agent = {
        'w_n': params_dict['w_n'],
        'threshold_a': params_dict['a'], # Pass 'a' which holds fixed TRUE_A
        'w_s': params_dict['w_s'],       # Pass 'w_s' which holds fixed TRUE_W_S
        **BASE_SIM_PARAMS                # Add other fixed base parameters
    }

    for i in range(n_sim_trials):
        # Catch potential errors during simulation of a single trial
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_inputs[i],
                norm_input=norm_inputs[i],
                params=params_for_agent # Pass the complete parameter set
            )
            results_list.append({
                'rt': trial_result.get('rt', np.nan), # Use .get for safety
                'choice': trial_result.get('choice', np.nan)
            })
        except Exception as e:
            print(f"Warning: Error in run_mvnes_trial (params={params_for_agent}): {e}")
            results_list.append({'rt': np.nan, 'choice': np.nan})

    return pd.DataFrame(results_list)

def get_summary_stat_keys():
    """Helper function to define the expected keys returned by calculate_summary_stats."""
    keys = ["n_choice_1", "n_choice_0", "choice_rate"]
    stat_names = ["rt_mean", "rt_median", "rt_var", "rt_skew", "rt_q10",
                  "rt_q30", "rt_q50", "rt_q70", "rt_q90", "rt_min",
                  "rt_max", "rt_range"]
    keys.extend(stat_names) # Overall stats
    keys.extend([f"choice_1_{s}" for s in stat_names]) # Choice 1 stats
    keys.extend([f"choice_0_{s}" for s in stat_names]) # Choice 0 stats
    keys.extend([f"rt_bin_{i}" for i in range(10)])    # Histogram bins
    return keys

def calculate_summary_stats(df_results):
    """
    Calculates a rich set of summary statistics from trial data (DataFrame).
    Handles cases with no trials or only one choice type robustly.
    Returns a dictionary with predefined keys, using NaN for impossible calculations.
    """
    all_keys = get_summary_stat_keys()
    summaries = {k: np.nan for k in all_keys} # Initialize with NaNs

    # Drop rows with NaN rt or choice if they occurred during simulation error
    df_results = df_results.dropna(subset=['rt', 'choice'])
    n_sim_trials = len(df_results)

    if n_sim_trials == 0: return summaries # Return NaNs if no valid data

    rts = df_results['rt'].values
    choices = df_results['choice'].values
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)

    summaries["n_choice_1"] = n_choice_1
    summaries["n_choice_0"] = n_choice_0
    summaries["choice_rate"] = n_choice_1 / n_sim_trials if n_sim_trials > 0 else np.nan

    def safe_stat(data, func, min_len=1, check_std=False):
        """Safely calculate stat, returning NaN on error or insufficient data."""
        data = np.asarray(data) # Ensure numpy array for nan functions
        valid_data = data[~np.isnan(data)] # Filter NaNs before calculation
        if len(valid_data) < min_len: return np.nan

        # Handle potential RuntimeWarning for std of empty or constant array
        std_val = np.std(valid_data) if len(valid_data) > 0 else 0
        # Check for NaN std dev or zero std dev when calculating skew
        if check_std and (np.isnan(std_val) or std_val == 0): return np.nan
        try:
            # Use nan-aware functions where possible
            if func.__name__.startswith("nan"):
                 result = func(data) # Use original data with nan-aware func
            elif func.__name__ == "<lambda>" or func.__name__ == "mean" or func.__name__ == "var" or func.__name__ == "median":
                 # Use nan-aware equivalents for common functions
                 nan_func = getattr(np, f"nan{func.__name__}", None)
                 result = nan_func(data) if nan_func else func(valid_data) # Fallback to valid data
            else:
                 result = func(valid_data) # Use filtered data for other funcs

            # Ensure result is finite (not inf or -inf)
            return result if np.isfinite(result) else np.nan
        except (ValueError, FloatingPointError, RuntimeWarning, ZeroDivisionError, IndexError):
             # Catch potential errors during calculation
            # print(f"Warning: safe_stat error for {func.__name__}") # Debug
            return np.nan

    stat_funcs = {
        "rt_mean": partial(safe_stat, func=np.nanmean),
        "rt_median": partial(safe_stat, func=np.nanmedian),
        "rt_var": partial(safe_stat, func=np.nanvar),
        "rt_skew": partial(safe_stat, func=lambda x: np.nanmean(((x - np.nanmean(x))/np.nanstd(x))**3), min_len=3, check_std=True),
        "rt_q10": partial(safe_stat, func=lambda x: np.nanpercentile(x, 10)),
        "rt_q30": partial(safe_stat, func=lambda x: np.nanpercentile(x, 30)),
        "rt_q50": partial(safe_stat, func=lambda x: np.nanpercentile(x, 50)),
        "rt_q70": partial(safe_stat, func=lambda x: np.nanpercentile(x, 70)),
        "rt_q90": partial(safe_stat, func=lambda x: np.nanpercentile(x, 90)),
        "rt_min": partial(safe_stat, func=np.nanmin),
        "rt_max": partial(safe_stat, func=np.nanmax),
        "rt_range": partial(safe_stat, func=lambda x: np.nanmax(x) - np.nanmin(x))
    }

    # Calculate overall stats
    for name, func in stat_funcs.items():
        summaries[name] = func(rts)

    # Calculate choice=1 stats
    for name, func in stat_funcs.items():
        summaries[f"choice_1_{name}"] = func(choice_1_rts)

    # Calculate choice=0 stats
    for name, func in stat_funcs.items():
        summaries[f"choice_0_{name}"] = func(choice_0_rts)

    # Add RT histogram bins
    try:
        valid_rts = rts[~np.isnan(rts)]
        if len(valid_rts) > 0:
            rt_min_val, rt_max_val = np.nanmin(valid_rts), np.nanmax(valid_rts)
            if np.isfinite(rt_min_val) and np.isfinite(rt_max_val): # Check min/max are valid
                hist_range = (rt_min_val, rt_max_val) if rt_max_val > rt_min_val else (rt_min_val - 0.1, rt_max_val + 0.1)
                hist, _ = np.histogram(valid_rts, bins=10, range=hist_range, density=True)
                summaries.update({f"rt_bin_{i}": hist[i] for i in range(10)})
            # else: leaves hist bins as NaN if min/max invalid
        # else: leaves hist bins as NaN if no valid RTs
    except Exception as e:
        print(f"Warning: Error calculating histogram bins: {e}")
        # Ensure keys exist even on error
        summaries.update({f"rt_bin_{i}": np.nan for i in range(10)})

    # Final check to ensure all keys present
    final_summaries = {k: summaries.get(k, np.nan) for k in all_keys}
    return final_summaries

def weighted_distance(sim_summary, obs_summary):
    """Calculate weighted Euclidean distance, handling NaNs robustly."""
    # --- Weights Dictionary ---
    # Tuning these weights is important! Assign higher weight to stats
    # believed to be more informative or reliable for the parameter.
    weights = {
        # Choice Rate: Often highly informative
        "choice_rate": 3.0,

        # Overall RT Moments (Lower weight if conditional stats are used)
        "rt_mean": 0.5, "rt_median": 0.5, "rt_var": 0.5, "rt_skew": 0.2,

        # Overall RT Quantiles (Can be useful)
        "rt_q10": 0.5, "rt_q30": 0.5, "rt_q50": 0.5, "rt_q70": 0.5, "rt_q90": 0.5,
        "rt_min": 0.1, "rt_max": 0.1, "rt_range": 0.1, # Range less informative

        # Conditional RT Stats (Potentially very informative)
        "choice_1_rt_mean": 2.0, "choice_1_rt_median": 2.0, "choice_1_rt_var": 1.5,
        "choice_1_rt_skew": 0.5, "choice_1_rt_q10": 1.5, "choice_1_rt_q30": 1.5,
        "choice_1_rt_q50": 1.5, "choice_1_rt_q70": 1.5, "choice_1_rt_q90": 1.5,
        "choice_1_rt_min": 0.2, "choice_1_rt_max": 0.2, "choice_1_rt_range": 0.2,

        "choice_0_rt_mean": 2.0, "choice_0_rt_median": 2.0, "choice_0_rt_var": 1.5,
        "choice_0_rt_skew": 0.5, "choice_0_rt_q10": 1.5, "choice_0_rt_q30": 1.5,
        "choice_0_rt_q50": 1.5, "choice_0_rt_q70": 1.5, "choice_0_rt_q90": 1.5,
        "choice_0_rt_min": 0.2, "choice_0_rt_max": 0.2, "choice_0_rt_range": 0.2,

        # Counts (Less informative than rate, but check existence)
        "n_choice_1": 0.1, "n_choice_0": 0.1,

        # Histogram Bins (Generally low weight unless specific shape focus)
        **{f"rt_bin_{i}": 0.05 for i in range(10)}
    }
    # ---------------------------

    nan_penalty_squared = 1000.0**2 # Penalize NaNs heavily
    total_distance_sq = 0.0
    keys_used_count = 0
    all_stat_keys = get_summary_stat_keys() # Get all defined statistic keys

    for key in all_stat_keys:
        weight = weights.get(key, 0) # Get weight, default to 0 if not specified
        if weight <= 0: continue    # Skip stats with zero or negative weight

        obs_val = obs_summary.get(key, np.nan)
        sim_val = sim_summary.get(key, np.nan)

        # Cannot compare if observed value itself is NaN
        if np.isnan(obs_val):
            # print(f"Skipping key '{key}': Observed value is NaN") # Debug
            continue

        if np.isnan(sim_val):
            # Apply large penalty if simulation failed to produce a required stat
            dist_sq = weight * nan_penalty_squared
            # print(f"NaN penalty applied for key '{key}'") # Debug
        else:
            # Calculate weighted squared difference
            diff = obs_val - sim_val
            # --- Normalization Option (Optional but often recommended) ---
            # Normalize difference by scale of observed stats to make weights more comparable
            # Example: Use interquartile range or std dev *across SBC simulations* if available
            # For simplicity now, using non-normalized weighted squared diff:
            dist_sq = weight * (diff**2)
            # -------------------------------------------------------------

        total_distance_sq += dist_sq
        keys_used_count += 1

    # If no valid statistics could be compared, return infinity
    if keys_used_count == 0:
        # print("Warning: No valid keys found for distance calculation.") # Debug
        return np.inf

    # Return Euclidean distance (sqrt of sum of squares)
    return np.sqrt(total_distance_sq)

def simulate_and_summarize_for_abc(params_abc, fixed_params, salience_inputs, norm_inputs):
    """ Helper function combining simulation and summary calculation for pyabc models arg. """
    # params_abc contains {'w_n': value} from pyabc sampler
    df_simulated = simulate_trials(
        # Combine sampled w_n with fixed a, w_s
        {'w_n': params_abc['w_n'], 'a': fixed_params['a'], 'w_s': fixed_params['w_s']},
        salience_inputs,
        norm_inputs
    )
    return calculate_summary_stats(df_simulated)

def calculate_sbc_rank(posterior_samples, true_value):
    """Calculates the rank of the true value within the posterior samples (0 to N)."""
    samples = np.asarray(posterior_samples)
    # Filter out NaNs before calculating rank
    valid_samples = samples[~np.isnan(samples)]
    if len(valid_samples) == 0:
        return np.nan # Cannot calculate rank if no valid samples
    # Rank = number of samples strictly less than true value
    rank = np.sum(valid_samples < true_value)
    # Result is rank L in range [0, N], where N = len(valid_samples)
    return rank

def plot_sbc_ranks(ranks, n_posterior_samples_per_run, n_bins=20):
    """Plots the histogram of SBC ranks vs uniform."""
    valid_ranks = np.asarray([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print("No valid SBC ranks found to plot.")
        return

    n_sims = len(valid_ranks)
    # Number of possible rank outcomes = N_samples + 1 (ranks 0 to N_samples)
    n_outcomes = n_posterior_samples_per_run + 1

    plt.figure(figsize=(10, 6))
    # Adjust bins to match the number of possible outcomes for discrete ranks
    # Use n_outcomes bins centered correctly if possible, or fall back
    actual_n_bins = min(n_bins, n_outcomes)
    if actual_n_bins <= 1: actual_n_bins = 10 # Fallback

    # Calculate histogram counts and density
    counts, bin_edges = np.histogram(valid_ranks, bins=actual_n_bins, range=(-0.5, n_posterior_samples_per_run + 0.5))
    # Normalize counts to get density (sum of bar areas = 1)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = counts / n_sims / bin_widths

    plt.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='green',
            edgecolor='black', align='edge', label=f'Observed Ranks (N={n_sims})')

    # Add uniform line for comparison
    expected_density = 1.0 / n_outcomes
    plt.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Expected Density ≈ {expected_density:.3f})')

    plt.xlabel(f"Rank of True Value in Posterior Samples (Range 0-{n_posterior_samples_per_run})", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Simulation-Based Calibration (SBC) Ranks for w_n", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, n_posterior_samples_per_run + 0.5) # Ensure x-axis covers all possible ranks
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':

    print("="*60)
    print("Starting SBC Validation Script for w_n")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Fixed Parameters: a={TRUE_A}, w_s={TRUE_W_S}")
    print(f"Base Sim Params: {BASE_SIM_PARAMS}")
    print(f"SBC Simulations: {N_SBC_SIMULATIONS}")
    print(f"Trials per Sim: {N_TRIALS_PER_SIM}")
    print(f"Task: {P_HIGH_CONFLICT*100}% High-Conflict Trials")
    print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Pops={ABC_MAX_NR_POPULATIONS}, Min Eps={ABC_MIN_EPSILON}")
    print(f"Prior for w_n: {WN_PRIOR}")
    print("="*60)

    # Store results
    sbc_results = []

    # Generate the fixed trial inputs ONCE
    print("Generating shared trial inputs...")
    salience_inputs, norm_inputs = generate_trial_inputs(N_TRIALS_PER_SIM, P_HIGH_CONFLICT, seed=GLOBAL_SEED)
    print(f"Generated {len(salience_inputs)} trial inputs.")

    # Define fixed parameters dictionary used in ABC simulator
    fixed_params_for_abc = {'a': TRUE_A, 'w_s': TRUE_W_S} # Intentionally only includes params *not* sampled from prior

    # --- Loop through each SBC simulation ---
    for i in range(N_SBC_SIMULATIONS):
        print("\n" + "-" * 50)
        print(f"Running SBC Simulation {i+1}/{N_SBC_SIMULATIONS}")
        sbc_seed = GLOBAL_SEED + i + 1 # Unique seed for reproducibility WITHIN this SBC run
        start_time_sbc = time.time()

        # 1. DRAW TRUE PARAMETER FROM PRIOR
        true_wn_dict = WN_PRIOR.rvs()
        true_wn = true_wn_dict['w_n']
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        # 2. GENERATE "OBSERVED" DATA using true_wn
        print(f"  Step 2: Generating 'observed' data (N={N_TRIALS_PER_SIM})...")
        # Combine true_wn with fixed params for this simulation run
        current_true_params_for_sim = {'w_n': true_wn, **fixed_params_for_abc}
        df_obs = simulate_trials(current_true_params_for_sim, salience_inputs, norm_inputs)
        observed_summaries = calculate_summary_stats(df_obs)
        print(f"    Obs Choice Rate: {observed_summaries.get('choice_rate', 'NaN'):.3f}")
        # Check for critical NaNs that would prevent comparison
        if np.isnan(observed_summaries.get('choice_rate', np.nan)):
             print("    ERROR: Essential summary stat (choice_rate) is NaN. Skipping SBC iteration.")
             sbc_results.append({'true_wn': true_wn, 'sbc_rank': np.nan, 'n_posterior_samples': 0})
             continue # Skip to next SBC iteration
        if any(np.isnan(v) for k,v in observed_summaries.items() if 'skew' not in k and 'rt_bin' not in k and k not in ['n_choice_1', 'n_choice_0']):
             print("    WARNING: NaNs generated in important observed summary statistics!")
             # Potentially skip or handle carefully depending on NaN frequency


        # 3. RUN ABC-SMC TO GET POSTERIOR
        print(f"  Step 3: Running ABC-SMC (Seed: {sbc_seed})...")
        history = None
        sbc_rank = np.nan
        n_posterior = 0
        try:
            # Define the simulator partial function for this run
            # It binds the *fixed* parameters (a, w_s) and the *fixed* inputs
            simulator_for_this_abc = partial(simulate_and_summarize_for_abc,
                                             fixed_params=fixed_params_for_abc,
                                             salience_inputs=salience_inputs,
                                             norm_inputs=norm_inputs)

            # Setup single-core sampler to avoid multiprocessing issues
            sampler = pyabc.sampler.SingleCoreSampler()

            abc = ABCSMC(
                models=simulator_for_this_abc,
                parameter_priors=WN_PRIOR, # Use the SAME prior used for drawing true_wn
                distance_function=weighted_distance,
                population_size=ABC_POPULATION_SIZE,
                sampler=sampler,
                eps=pyabc.epsilon.QuantileEpsilon(alpha=0.5) # Adaptive quantile-based epsilon
            )
            temp_dir = tempfile.gettempdir()
            # Add seed to db name for parallel safety if needed, though single run here
            db_file = f"abc_sbc_run_{i+1}_seed_{sbc_seed}.db"
            db_path = f"sqlite:///{os.path.join(temp_dir, db_file)}"
            if os.path.exists(db_path.replace("sqlite:///", "")):
                 os.remove(db_path.replace("sqlite:///", ""))

            abc.new(db_path, observed_summaries)
            # Add seeding for the run method if needed - check pyabc docs
            # Using seed on the RNG used by the prior is the primary SBC requirement
            history = abc.run(minimum_epsilon=ABC_MIN_EPSILON, max_nr_populations=ABC_MAX_NR_POPULATIONS)


            # 4. EXTRACT POSTERIOR & CALCULATE RANK
            print("  Step 4: Extracting posterior and calculating rank...")
            # Ensure history object exists and ABC run completed reasonably
            if history and history.max_t >= 0:
                df_posterior, w = history.get_distribution(m=0, t=history.max_t)

                if not df_posterior.empty:
                    posterior_samples = df_posterior['w_n'].values
                    n_posterior = len(posterior_samples)
                    # Use UNWEIGHTED samples for standard SBC rank calculation
                    sbc_rank = calculate_sbc_rank(posterior_samples, true_wn)
                    print(f"    True value: {true_wn:.4f}, Rank: {sbc_rank} (out of {n_posterior} samples)")
                else:
                    print("    WARNING: Posterior distribution is empty! ABC failed to accept samples.")
                    sbc_rank = np.nan # Mark as NaN if no samples
            else:
                 print("    WARNING: ABC history is empty or invalid.")
                 sbc_rank = np.nan


        except Exception as e:
            print(f"    ERROR during ABC run or result extraction for true_wn={true_wn:.4f}:")
            traceback.print_exc() # Print full traceback for debugging
            sbc_rank = np.nan # Mark as NaN on error
        finally:
            # Clean up temporary database file
            db_full_path = db_path.replace("sqlite:///", "")
            if os.path.exists(db_full_path):
                 try: os.remove(db_full_path)
                 except OSError as e: print(f"    Warning: Error removing db {db_full_path}: {e}")

        # Store result for this iteration
        sbc_results.append({'true_wn': true_wn, 'sbc_rank': sbc_rank, 'n_posterior_samples': n_posterior})
        end_time_sbc = time.time()
        print(f"  Finished SBC Iteration {i+1} in {end_time_sbc - start_time_sbc:.1f} sec.")


    # --- 5. FINAL ANALYSIS & PLOTTING ---
    print("\n" + "="*60)
    print("Finished all SBC simulations. Generating final plot...")

    if sbc_results:
        results_df = pd.DataFrame(sbc_results)
        print("\nSBC Results Summary (First 5 & Last 5 Rows):")
        print(pd.concat([results_df.head(), results_df.tail()]))
        valid_ranks_count = results_df['sbc_rank'].notna().sum()
        print(f"\nNumber of valid ranks obtained: {valid_ranks_count} / {N_SBC_SIMULATIONS}")

        if valid_ranks_count > 0:
             # Use the target population size for N samples in plot title
             # This assumes most runs that yielded samples got ABC_POPULATION_SIZE
             plot_sbc_ranks(results_df['sbc_rank'].tolist(), ABC_POPULATION_SIZE)
        else:
             print("No valid ranks to plot.")
    else:
        print("No valid SBC results were obtained.")

    print("\nSBC validation script finished.")
    print("="*60)