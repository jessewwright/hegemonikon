# Filename: validate_wn_recovery_multilevel.py
# Purpose: Test parameter recovery for w_n using a multi-level conflict task,
#          ABC-SMC, and rich summary statistics.
#          MODIFIED FOR FASTER (BUT LESS PRECISE) ITERATION.

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
import traceback
import random

# --- 1. Robust Imports & Dependency Checks ---
# (Keep this section as is from your last version)
try:
    # Dynamically add 'src' to path based on script location
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2
        DT = 0.01
        MAX_TIME = 3.0
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    sys.exit(1)
try:
    import pyabc
    from pyabc import Distribution, RV, ABCSMC
except ImportError:
    print("ERROR: pyabc library not found. Please install it: pip install pyabc")
    sys.exit(1)

logging.getLogger("pyabc").setLevel(logging.WARNING)

# --- 2. Configuration & Constants ---
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# *** SPEEDUP MODIFICATIONS ***
N_SBC_SIMULATIONS = 75  # Reduced from 100 for faster testing
N_TRIALS_PER_SIM = 400   # Increased for more stable statistics
ABC_POPULATION_SIZE = 50  # Smaller population for faster testing stable stats

TRUE_A = 1.0
TRUE_W_S = 0.7
# Define grid that samples both below and above the salience weight
WN_GRID = np.linspace(0.1, 2.0, 20)  # More points for better coverage
BASE_SIM_PARAMS = {
    't': T_NONDECISION, 'noise_std_dev': 0.3, 'dt': DT, 'max_time': 4.0,  # Increased noise_std_dev to 0.3
    'affect_stress_threshold_reduction': -0.3, 'veto_flag': False,
    'timeout_penalty': -0.5, 'timeout_penalty_scale': 1.0  
}

# Define trial types with more balanced distribution and amplified conflict levels
NEUTRAL_PROB = 0.4  # 40% neutral trials (no conflict)
CONFLICT_LEVELS = [0.1, 0.4, 1.0, 1.6]    # target drift offsets - amplified to ensure suppression
CONFLICT_PROPS  = np.array([0.15, 0.15, 0.15, 0.15]) / 0.6  # Normalize to sum to 1 after neutral trials
# Total: 40% neutral + 60% conflict (15% each level)
# Note: Even at w_n=0.2, the lowest conflict (0.1) will create negative drift
# Using Go (S=1,N=0) and NoGo/Conflict (S=1,N=1) as in previous SBC script
# TRIAL_TYPES definition is not directly used for input generation in this version

ABC_POPULATION_SIZE = 150  # Increased for better sampling
ABC_MAX_NR_POPULATIONS = 8 # Kept the same
ABC_MIN_EPSILON = 0.15  # Slightly lower minimum epsilon for more exploration

WN_PRIOR_DIST = RV("uniform", 0, 2.0)
WN_PRIOR = Distribution(w_n=WN_PRIOR_DIST)

# --- 3. Core Functions (Keep generate_multilevel_trial_inputs, simulate_trials, get_summary_stat_keys, calculate_summary_stats, weighted_distance, simulate_and_summarize_for_abc, calculate_sbc_rank, plot_sbc_ranks AS THEY WERE IN YOUR LAST WORKING VERSION) ---
# For brevity, I'll assume these functions are copied from your last complete script.
# The key is that they work with the parameters below.

def generate_multilevel_trial_inputs(n_trials, seed=None):
    """Generates trial inputs with multi-level conflict."""
    rng = np.random.default_rng(seed)
    salience_inputs = np.zeros(n_trials)
    norm_inputs = np.zeros(n_trials)
    trial_labels = []
    
    for i in range(n_trials):
        if rng.random() < NEUTRAL_PROB:  # 50% chance of neutral trial
            salience_inputs[i] = TRUE_W_S  # Base salience
            norm_inputs[i] = 0.0  # No conflict
            trial_labels.append("Go")  # Neutral trials are Go trials
        else:  # Conflict trial
            level_idx = rng.choice(len(CONFLICT_LEVELS), p=CONFLICT_PROPS)
            salience_inputs[i] = TRUE_W_S  # Base salience
            norm_inputs[i] = CONFLICT_LEVELS[level_idx]
            trial_labels.append("NoGo")  # Conflict trials are NoGo trials
    
    return salience_inputs, norm_inputs, np.array(trial_labels)


def simulate_trials(params_dict, salience_inputs, norm_inputs, trial_labels):
    n_sim_trials = len(salience_inputs)
    results_list = []
    agent = MVNESAgent(config={})
    params_for_agent = {
        'w_n': params_dict['w_n'],
        'threshold_a': params_dict['a'],
        'w_s': params_dict['w_s'],
        **BASE_SIM_PARAMS
    }
    
    # Calculate drift magnitude for timeout penalty scaling
    drift_magnitudes = np.abs(params_dict['w_s'] * salience_inputs + params_dict['w_n'] * norm_inputs)
    
    for i in range(n_sim_trials):
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_inputs[i],
                norm_input=norm_inputs[i],
                params=params_for_agent
            )
            
            # Calculate timeout penalty based on drift magnitude
            timeout_penalty = params_for_agent['timeout_penalty']
            if trial_result['timeout']:
                timeout_penalty *= (1 + params_for_agent['timeout_penalty_scale'] * drift_magnitudes[i])
            
            results_list.append({
                'choice': trial_result['choice'],
                'rt': trial_result['rt'],
                'timeout': trial_result['timeout'],
                'timeout_penalty': timeout_penalty,
                'trial_label': trial_labels[i]
            })
        except Exception as e:
            print(f"Error in trial {i}: {e}")
            continue
    
    return pd.DataFrame(results_list)


def get_summary_stat_keys():
    """Get all summary statistic keys."""
    # Basic statistics
    stat_types = ['rt_mean', 'rt_median', 'rt_var', 'rt_skew',
                 'rt_q10', 'rt_q30', 'rt_q50', 'rt_q70', 'rt_q90',
                 'rt_min', 'rt_max', 'rt_range']
    
    # Survival curve deciles
    survival_percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]  # More granular deciles
    
    keys = ['n_choice_1', 'n_choice_0', 'choice_rate_overall']
    
    # Overall statistics
    keys.extend([f"overall_{s}" for s in stat_types])
    
    # Conflict-level specific statistics
    keys.extend([f"neutral_choice1_{s}" for s in stat_types])
    keys.extend([f"neutral_choice0_{s}" for s in stat_types])
    keys.extend([f"conflict_choice1_{s}" for s in stat_types])
    keys.extend([f"conflict_choice0_{s}" for s in stat_types])
    
    # RT bins
    keys.extend([f"overall_rt_bin_{i}" for i in range(10)])
    
    # Survival curve features - more granular deciles
    for p in survival_percentiles:
        keys.append(f"surv_p{int(p*100)}")  # Overall survival
        # Conflict-level specific survival
        keys.extend([f"{ctype}_choice1_surv_p{int(p*100)}" for ctype in ['neutral', 'conflict']])
        keys.extend([f"{ctype}_choice0_surv_p{int(p*100)}" for ctype in ['neutral', 'conflict']])
    
    return keys

def calculate_summary_stats(df_results):
    """Calculate summary statistics for SBC."""
    # Define statistical functions
    stat_funcs_def = {
        "rt_mean": np.mean, "rt_median": np.median, "rt_var": np.var,
        "rt_skew": lambda x: np.mean(((x - np.nanmean(x))/np.nanstd(x))**3),
        "rt_min": np.min, "rt_max": np.max, "rt_range": lambda x: np.nanmax(x) - np.nanmin(x)
    }
    
    # Add percentile functions dynamically
    for q in [10, 30, 50, 70, 90]:  # Basic percentiles
        stat_funcs_def[f"rt_q{q}"] = partial(np.nanpercentile, q=q)
    
    # Add survival curve deciles
    survival_percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    for p in survival_percentiles:
        stat_funcs_def[f"surv_p{int(p*100)}"] = partial(np.nanpercentile, q=p*100)

    all_keys = get_summary_stat_keys()
    summaries = {k: np.nan for k in all_keys}
    df_results = df_results.dropna(subset=['rt', 'choice'])
    n_total_trials = len(df_results)

    if n_total_trials == 0: return summaries

    # Get max_time from base parameters for imputation
    max_time = BASE_SIM_PARAMS['max_time']

    overall_rts = df_results['rt'].values
    overall_choices = df_results['choice'].values
    n_choice_1_overall = np.sum(overall_choices == 1)
    n_choice_0_overall = np.sum(overall_choices == 0)
    
    # Handle empty choice groups
    if n_choice_1_overall == 0:
        summaries["n_choice_1"] = 0
        summaries["choice_rate_overall"] = 0.0
        for name in stat_funcs_def.keys():
            summaries[f"choice_1_{name}"] = max_time if "rt" in name else 0.0
            # Add survival curve deciles for empty choice groups
            for p in survival_percentiles:
                summaries[f"choice_1_surv_p{int(p*100)}"] = max_time
    else:
        summaries["n_choice_1"] = n_choice_1_overall
        summaries["choice_rate_overall"] = n_choice_1_overall / n_total_trials

    if n_choice_0_overall == 0:
        summaries["n_choice_0"] = 0
        for name in stat_funcs_def.keys():
            summaries[f"choice_0_{name}"] = max_time if "rt" in name else 0.0
            # Add survival curve deciles for empty choice groups
            for p in survival_percentiles:
                summaries[f"choice_0_surv_p{int(p*100)}"] = max_time
    else:
        summaries["n_choice_0"] = n_choice_0_overall

    def safe_stat(data, func, min_len=1, check_std=False):
        data = np.asarray(data)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < min_len: return np.nan
        std_val = np.std(valid_data) if len(valid_data) > 0 else 0
        if check_std and (np.isnan(std_val) or std_val == 0): return np.nan
        try:
            if func in [np.mean, np.median, np.var, np.std, np.min, np.max, np.percentile]:
                nan_func_name = f"nan{func.__name__}"
                if hasattr(np, nan_func_name):
                    nan_func = getattr(np, nan_func_name)
                    if func == np.percentile:
                        # This part was tricky for partial, handle percentile q directly in lambda
                        return func(data) # Relies on lambda to call np.nanpercentile
                    return nan_func(data)
            if func.__name__ == "<lambda>" and "percentile" in str(func):
                 return func(data)
            result = func(valid_data)
            return result if np.isfinite(result) else np.nan
        except Exception: return np.nan

    stat_funcs_def = {
        "rt_mean": np.mean, "rt_median": np.median, "rt_var": np.var,
        "rt_skew": lambda x: np.mean(((x - np.nanmean(x))/np.nanstd(x))**3),
        "rt_q10": partial(np.nanpercentile, q=10), "rt_q30": partial(np.nanpercentile, q=30),
        "rt_q50": partial(np.nanpercentile, q=50), "rt_q70": partial(np.nanpercentile, q=70),
        "rt_q90": partial(np.nanpercentile, q=90), "rt_min": np.min,
        "rt_max": np.max, "rt_range": lambda x: np.nanmax(x) - np.nanmin(x)
    }

    for name, func in stat_funcs_def.items():
        summaries[f"overall_{name}"] = safe_stat(overall_rts, func, min_len=3 if name=="rt_skew" else 1, check_std=(name=="rt_skew"))

    conditions = [
        ("go_choice1", (df_results['trial_label'] == "Go") & (df_results['choice'] == 1)),
        ("nogo_choice1", (df_results['trial_label'] == "NoGo") & (df_results['choice'] == 1)),
        ("nogo_choice0", (df_results['trial_label'] == "NoGo") & (df_results['choice'] == 0)),
        ("go_choice0", (df_results['trial_label'] == "Go") & (df_results['choice'] == 0)),
    ]
    for prefix, mask in conditions:
        conditional_rts = df_results.loc[mask, 'rt'].values
        for name, func in stat_funcs_def.items():
            summaries[f"{prefix}_{name}"] = safe_stat(conditional_rts, func, min_len=3 if name=="rt_skew" else 1, check_std=(name=="rt_skew"))

    try:
        valid_rts = overall_rts[~np.isnan(overall_rts)]
        if len(valid_rts) > 0:
            rt_min_val, rt_max_val = np.nanmin(valid_rts), np.nanmax(valid_rts)
            if np.isfinite(rt_min_val) and np.isfinite(rt_max_val):
                hist_range = (rt_min_val, rt_max_val) if rt_max_val > rt_min_val else (rt_min_val - 0.1, rt_max_val + 0.1)
                hist, _ = np.histogram(valid_rts, bins=10, range=hist_range, density=True)
                summaries.update({f"overall_rt_bin_{i}": hist[i] for i in range(10)})
    except Exception as e: print(f"Warning: Hist error: {e}")

    # Calculate survival curve features
    max_time = BASE_SIM_PARAMS['max_time']
    survival_percentiles = [0.25, 0.50, 0.75]
    
    # Overall survival
    if len(valid_rts) > 0:
        for p in survival_percentiles:
            threshold = p * max_time
            prop = np.mean(valid_rts < threshold)
            summaries[f"surv_p{int(p*100)}"] = prop
    
    # Condition-specific survival
    for prefix, mask in conditions:
        conditional_rts = df_results.loc[mask, 'rt'].values
        if len(conditional_rts) > 0:
            for p in survival_percentiles:
                threshold = p * max_time
                prop = np.mean(conditional_rts < threshold)
                summaries[f"{prefix}_surv_p{int(p*100)}"] = prop
    
    return {k: summaries.get(k, np.nan) for k in all_keys}

def weighted_distance(sim_summary, obs_summary):
    """Calculate weighted distance between simulation and observation summaries."""
    # Base weights for existing statistics
    weights = {
        "choice_rate_overall": 5.0,
        "overall_rt_mean": 3.0,
        "overall_rt_median": 3.0,
        "overall_rt_var": 2.0
    }
    
    # Add weights for survival curve features
    survival_percentiles = [0.25, 0.50, 0.75]
    for p in survival_percentiles:
        weights[f"surv_p{int(p*100)}"] = 2.0  # Weight survival features
        for prefix in ["go_choice1", "nogo_choice1", "nogo_choice0", "go_choice0"]:
            weights[f"{prefix}_surv_p{int(p*100)}"] = 1.5  # Slightly less weight for condition-specific
    
    # Default weight of 1.0 for all other statistics
    for key in get_summary_stat_keys():
        if key not in weights:
            weights[key] = 1.0
    
    # Get common keys between simulation and observation
    common_keys = set(sim_summary.keys()).intersection(obs_summary.keys())
    
    dist = 0
    n_valid = 0
    
    for key in common_keys:
        if np.isnan(obs_summary[key]) or np.isnan(sim_summary[key]):
            continue
        dist += weights[key] * (sim_summary[key] - obs_summary[key])**2
        n_valid += 1
    
    if n_valid == 0: return float('inf')
    return np.sqrt(dist / n_valid)

def simulate_and_summarize_for_abc(params_abc, fixed_params, salience_inputs, norm_inputs, trial_labels):
    df_simulated = simulate_trials(
        {'w_n': params_abc['w_n'], **fixed_params},
        salience_inputs, norm_inputs, trial_labels
    )
    return calculate_summary_stats(df_simulated)

def calculate_sbc_rank(posterior_samples, true_value):
    samples = np.asarray(posterior_samples)
    valid_samples = samples[~np.isnan(samples)]
    if len(valid_samples) == 0: return np.nan
    rank = np.sum(valid_samples < true_value)
    return rank

def plot_sbc_ranks(ranks, n_posterior_samples_per_run, n_bins=20):
    valid_ranks = np.asarray([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print("No valid SBC ranks found to plot.")
        return
    n_sims = len(valid_ranks)
    n_outcomes = n_posterior_samples_per_run + 1
    plt.figure(figsize=(10, 6))
    actual_n_bins = min(n_bins, n_outcomes)
    if actual_n_bins <= 1: actual_n_bins = max(10, int(np.sqrt(n_sims))) if n_sims > 0 else 10
    counts, bin_edges = np.histogram(valid_ranks, bins=actual_n_bins, range=(-0.5, n_posterior_samples_per_run + 0.5))
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    density = counts / n_sims / (bin_widths[0] if len(bin_widths)>0 else 1) # Handle case of single bin or no width
    plt.bar(bin_edges[:-1], density, width=bin_widths, alpha=0.7, color='green',
            edgecolor='black', align='edge', label=f'Observed Ranks (N={n_sims})')
    expected_density = 1.0 / n_outcomes
    plt.axhline(expected_density, color='red', linestyle='--', label=f'Uniform (Exp. Density ≈ {expected_density:.3f})')
    plt.xlabel(f"Rank of True Value (0-{n_posterior_samples_per_run})", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"SBC Ranks for w_n (Multi-Level Task - Faster Settings)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, n_posterior_samples_per_run + 0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block (Copied and modified from your last version) ---
if __name__ == '__main__':
    print("="*60)
    print("Starting SBC Validation Script for w_n (Multi-Level Task - FASTER ITERATION)")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Fixed Parameters: a={TRUE_A}, w_s={TRUE_W_S}")
    print(f"Base Sim Params: {BASE_SIM_PARAMS}")
    print(f"SBC Simulations: {N_SBC_SIMULATIONS}")
    print(f"Trials per Sim: {N_TRIALS_PER_SIM}")
    # P_HIGH_CONFLICT is not used by generate_multilevel_trial_inputs, but was printed before
    # The task is now 70% Go, 30% NoGo as per generate_multilevel_trial_inputs
    print(f"Task: 70% Go / 30% NoGo Trials")
    print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Pops={ABC_MAX_NR_POPULATIONS}, Min Eps={ABC_MIN_EPSILON}")
    print(f"Prior for w_n: {WN_PRIOR}")
    print("="*60)

    sbc_results = []
    print("Generating shared trial inputs for multi-level task...")
    salience_inputs, norm_inputs, trial_labels = generate_multilevel_trial_inputs(N_TRIALS_PER_SIM, seed=GLOBAL_SEED)
    print(f"Generated {len(salience_inputs)} trial inputs.")

    fixed_params_for_abc = {'a': TRUE_A, 'w_s': TRUE_W_S}

    for i in range(N_SBC_SIMULATIONS):
        print("\n" + "-" * 50)
        print(f"Running SBC Simulation {i+1}/{N_SBC_SIMULATIONS}")
        sbc_seed = GLOBAL_SEED + i + 1
        np.random.seed(sbc_seed)
        random.seed(sbc_seed)
        start_time_sbc = time.time()

        true_wn_dict = WN_PRIOR.rvs()
        true_wn = true_wn_dict['w_n']
        print(f"  Step 1: Drawn True w_n = {true_wn:.4f}")

        print(f"  Step 2: Generating 'observed' data (N={N_TRIALS_PER_SIM})...")
        current_true_params_for_sim = {'w_n': true_wn, **fixed_params_for_abc}
        df_obs = simulate_trials(current_true_params_for_sim, salience_inputs, norm_inputs, trial_labels)
        observed_summaries = calculate_summary_stats(df_obs)
        print(f"    Obs Choice Rate (Overall): {observed_summaries.get('choice_rate_overall', 'NaN'):.3f}")
        if np.isnan(observed_summaries.get('choice_rate_overall', np.nan)):
             print("    ERROR: Essential summary stat (choice_rate_overall) is NaN. Skipping SBC iteration.")
             sbc_results.append({'true_wn': true_wn, 'sbc_rank': np.nan, 'n_posterior_samples': 0})
             continue

        print(f"  Step 3: Running ABC-SMC (Seed: {sbc_seed})...")
        history = None; sbc_rank = np.nan; n_posterior = 0
        try:
            simulator_for_this_abc = partial(simulate_and_summarize_for_abc,
                                             fixed_params=fixed_params_for_abc,
                                             salience_inputs=salience_inputs,
                                             norm_inputs=norm_inputs,
                                             trial_labels=trial_labels)
            # *** USE SINGLE CORE SAMPLER FOR NOW TO AVOID WINDOWS MULTIPROCESSING ISSUES ***
            sampler = pyabc.sampler.SingleCoreSampler()

            abc = ABCSMC(
                models=simulator_for_this_abc,
                parameter_priors=WN_PRIOR,
                distance_function=weighted_distance,
                population_size=ABC_POPULATION_SIZE,
                sampler=sampler,
                eps=pyabc.epsilon.QuantileEpsilon(alpha=0.5) # More aggressive start
            )
            temp_dir = tempfile.gettempdir()
            db_file = f"abc_sbc_multilevel_run_{i+1}_seed_{sbc_seed}.db" # Ensure unique DB name
            db_path = f"sqlite:///{os.path.join(temp_dir, db_file)}"
            if os.path.exists(db_path.replace("sqlite:///", "")):
                 os.remove(db_path.replace("sqlite:///", ""))

            abc.new(db_path, observed_summaries)
            history = abc.run(minimum_epsilon=ABC_MIN_EPSILON, max_nr_populations=ABC_MAX_NR_POPULATIONS)

            print("  Step 4: Extracting posterior and calculating rank...")
            if history and history.max_t >= 0:
                df_posterior, w = history.get_distribution(t=history.max_t) # m=0 is default for last gen
                if not df_posterior.empty:
                    posterior_samples = df_posterior['w_n'].values
                    n_posterior = len(posterior_samples)
                    sbc_rank = calculate_sbc_rank(posterior_samples, true_wn)
                    print(f"    True value: {true_wn:.4f}, Rank: {sbc_rank} (out of {n_posterior} samples)")
                else: print("    WARNING: Posterior distribution is empty!")
            else: print("    WARNING: ABC history is empty or invalid.")
        except Exception as e:
            print(f"    ERROR during ABC for true_wn={true_wn:.4f}:")
            traceback.print_exc()
        finally:
            db_full_path = db_path.replace("sqlite:///", "")
            if os.path.exists(db_full_path):
                 try: os.remove(db_full_path)
                 except OSError as e: print(f"    Warning: Error removing db {db_full_path}: {e}")

        sbc_results.append({'true_wn': true_wn, 'sbc_rank': sbc_rank, 'n_posterior_samples': n_posterior})
        end_time_sbc = time.time()
        print(f"  Finished SBC Iteration {i+1} in {end_time_sbc - start_time_sbc:.1f} sec.")

    # --- 5. FINAL ANALYSIS & PLOTTING ---
    print("\n" + "="*60)
    print("Finished all SBC simulations. Generating final plot...")
    if sbc_results:
        results_df = pd.DataFrame(sbc_results)
        print("\nSBC Results Summary (Sample):")
        print(pd.concat([results_df.head(), results_df.tail()]))
        valid_ranks_count = results_df['sbc_rank'].notna().sum()
        print(f"\nNumber of valid ranks obtained: {valid_ranks_count} / {N_SBC_SIMULATIONS}")
        if valid_ranks_count > 0:
             avg_n_posterior = results_df.loc[results_df['n_posterior_samples'] > 0, 'n_posterior_samples'].mean()
             n_samples_for_plot = int(avg_n_posterior) if not np.isnan(avg_n_posterior) and avg_n_posterior > 0 else ABC_POPULATION_SIZE
             plot_sbc_ranks(results_df['sbc_rank'].tolist(), n_samples_for_plot)
        else: print("No valid ranks to plot.")
    else: print("No valid SBC results were obtained.")
    print("\nSBC validation script (multi-level - FASTER) finished.")
    print("="*60)