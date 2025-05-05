# Filename: validate_wn_recovery.py
# Purpose: Test parameter recovery for w_n under simplified, controlled conditions
#          using ABC-SMC and rich summary statistics.

import sys
# Assuming the script is in the root project directory
# Adjust if placed elsewhere (e.g., scripts/)
sys.path.append('src')

import numpy as np
import pandas as pd
from functools import partial
import pyabc
from pyabc import Distribution, RV, ABCSMC, PNormDistance # Ensure PNormDistance is imported if used directly
import tempfile
import matplotlib.pyplot as plt
import time
import os

# --- Local Imports ---
try:
    from agent_mvnes import MVNESAgent
except ImportError:
    print("ERROR: Could not import MVNESAgent from src.agent_mvnes. Ensure script is run from project root or adjust PYTHONPATH.")
    sys.exit(1)
# Import necessary fixed parameters (or define defaults)
try:
    from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
except ImportError:
    print("Warning: Could not import agent_config. Using default simulation parameters.")
    T_NONDECISION = 0.1
    NOISE_STD_DEV = 0.2 # Using a more realistic noise value than 1.0 might be better
    DT = 0.01
    MAX_TIME = 2.0

# --- Configuration ---
N_TRIALS_PER_WN = 300   # Number of trials per w_n value (Increase for better stats)
WN_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] # Grid of true w_n values to test
TRUE_A = 1.0            # Fixed threshold value (Adjust as needed)
TRUE_W_S = 0.7          # Fixed salience weight (Adjust as needed)
P_HIGH_CONFLICT = 0.5   # Probability of a high-conflict trial

# ABC Configuration
ABC_POPULATION_SIZE = 200 # Smaller for faster testing, increase for accuracy
ABC_MAX_NR_POPULATIONS = 8
ABC_MIN_EPSILON = 0.01 # Adjust based on observed distances

# Define base simulation parameters (excluding w_n, a, w_s)
BASE_SIM_PARAMS = {
    't': T_NONDECISION,
    'noise_std_dev': NOISE_STD_DEV,
    'dt': DT,
    'max_time': MAX_TIME,
    # Add other necessary params from agent_mvnes if they aren't defaulted
    'affect_stress_threshold_reduction': -0.3 # Example, ensure it's used if needed
}

# --- Helper Functions ---

def simulate_summary(params_abc, fixed_params, salience_inputs, norm_inputs):
    """
    Simulates N trials using fixed inputs and returns rich summary statistics.
    'params_abc' contains the parameter(s) being estimated by ABC (here, just w_n).
    'fixed_params' contains other fixed model parameters (a, w_s, t, noise, etc.).
    """
    # Combine parameters: the ones from ABC + the fixed ones
    full_params = {**fixed_params, **params_abc}

    # Initialize arrays
    n_sim_trials = len(salience_inputs)
    rts = np.zeros(n_sim_trials)
    choices = np.zeros(n_sim_trials, dtype=int)

    # Simulate trials
    # Creating a new agent instance each time *can* be slow.
    # If performance is critical, initialize agent outside and pass as arg,
    # but ensure no state leaks between simulations if agent has internal state.
    agent = MVNESAgent(config={}) # Assuming agent is stateless per trial run
    for i in range(n_sim_trials):
        # Ensure parameters passed to run_mvnes_trial are exactly what it expects
        # Specifically, check if it needs 'threshold_a' or just 'a' etc.
        # Assuming it needs 'threshold_a', 'w_s', 'w_n', and others from BASE_SIM_PARAMS
        trial_params_for_agent = {
            'w_n': full_params['w_n'],
            'threshold_a': full_params['a'], # Use 'a' from fixed_params
            'w_s': full_params['w_s'],       # Use 'w_s' from fixed_params
            **BASE_SIM_PARAMS                # Add other necessary params like t, noise, etc.
        }
        result = agent.run_mvnes_trial(
            salience_input=salience_inputs[i],
            norm_input=norm_inputs[i],
            params=trial_params_for_agent # Pass the combined dictionary
        )
        rt, ch = result['rt'], result['choice']
        rts[i], choices[i] = rt, ch

    # --- Calculate Rich Summary Statistics ---
    summaries = {}

    # Handle cases with no trials (shouldn't happen here, but good practice)
    if n_sim_trials == 0: return {k: np.nan for k in get_summary_stat_keys()} # Define this helper if needed

    # Separate RTs by choice
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)

    # Add choice counts and overall rate
    summaries["n_choice_1"] = n_choice_1
    summaries["n_choice_0"] = n_choice_0
    summaries["choice_rate"] = n_choice_1 / n_sim_trials if n_sim_trials > 0 else np.nan

    # Define statistics to calculate (to avoid repetition)
    stat_funcs = {
        "rt_mean": np.mean,
        "rt_median": np.median,
        "rt_var": np.var,
        "rt_skew": lambda x: np.nan if len(x) < 3 or np.std(x) == 0 else np.mean((x - np.mean(x))**3) / np.std(x)**3,
        "rt_q10": lambda x: np.nanpercentile(x, 10),
        "rt_q30": lambda x: np.nanpercentile(x, 30),
        "rt_q50": lambda x: np.nanpercentile(x, 50),
        "rt_q70": lambda x: np.nanpercentile(x, 70),
        "rt_q90": lambda x: np.nanpercentile(x, 90),
        "rt_min": np.min,
        "rt_max": np.max,
        "rt_range": lambda x: np.max(x) - np.min(x)
    }

    # Calculate overall stats
    for name, func in stat_funcs.items():
        try:
            summaries[name] = func(rts) if len(rts) > 0 else np.nan
        except Exception:
            summaries[name] = np.nan

    # Calculate choice=1 stats
    for name, func in stat_funcs.items():
        key = f"choice_1_{name}"
        try:
            summaries[key] = func(choice_1_rts) if n_choice_1 > 0 else np.nan
        except Exception:
            summaries[key] = np.nan

    # Calculate choice=0 stats
    for name, func in stat_funcs.items():
        key = f"choice_0_{name}"
        try:
            summaries[key] = func(choice_0_rts) if n_choice_0 > 0 else np.nan
        except Exception:
            summaries[key] = np.nan

    # Add RT histogram bins
    try:
        if len(rts) > 0:
            rt_min_val = np.min(rts)
            rt_max_val = np.max(rts)
            # Avoid histogram error if min==max
            hist_range = (rt_min_val, rt_max_val) if rt_max_val > rt_min_val else (rt_min_val - 0.1, rt_max_val + 0.1)
            hist, _ = np.histogram(rts, bins=10, range=hist_range, density=True)
            summaries.update({f"rt_bin_{i}": hist[i] for i in range(10)})
        else:
             summaries.update({f"rt_bin_{i}": np.nan for i in range(10)})
    except Exception:
         summaries.update({f"rt_bin_{i}": np.nan for i in range(10)})

    return summaries


def weighted_distance(sim_summary, obs_summary):
    """
    Calculate weighted Euclidean distance between summary statistics dictionaries.
    Handles NaNs robustly.
    """
    # Define weights (adjust these based on sensitivity analysis/importance)
    weights = {
        "choice_rate": 2.0,
        "rt_mean": 1.0, "rt_var": 1.0, "rt_median": 1.0, "rt_skew": 1.0,
        "rt_q10": 1.0, "rt_q30": 1.0, "rt_q50": 1.0, "rt_q70": 1.0, "rt_q90": 1.0,
        "rt_min": 0.5, "rt_max": 0.5, "rt_range": 0.5,

        "choice_1_rt_mean": 1.5, "choice_1_rt_median": 1.5, "choice_1_rt_var": 1.5,
        "choice_1_rt_skew": 1.5, "choice_1_rt_q10": 1.5, "choice_1_rt_q30": 1.5,
        "choice_1_rt_q50": 1.5, "choice_1_rt_q70": 1.5, "choice_1_rt_q90": 1.5,
        "choice_1_rt_min": 0.75, "choice_1_rt_max": 0.75, "choice_1_rt_range": 0.75,

        "choice_0_rt_mean": 1.5, "choice_0_rt_median": 1.5, "choice_0_rt_var": 1.5,
        "choice_0_rt_skew": 1.5, "choice_0_rt_q10": 1.5, "choice_0_rt_q30": 1.5,
        "choice_0_rt_q50": 1.5, "choice_0_rt_q70": 1.5, "choice_0_rt_q90": 1.5,
        "choice_0_rt_min": 0.75, "choice_0_rt_max": 0.75, "choice_0_rt_range": 0.75,

        "n_choice_1": 1.0, "n_choice_0": 1.0,

        **{f"rt_bin_{i}": 0.5 for i in range(10)} # Add weights for histogram bins
    }

    total_distance = 0.0
    valid_keys_count = 0

    # Iterate through keys present in the observed summary
    for key in obs_summary.keys():
        # Only compare if key is also in simulated summary and has a weight
        if key in sim_summary and key in weights:
            obs_val = obs_summary[key]
            sim_val = sim_summary[key]

            # Skip if observed value is NaN (cannot compare)
            if np.isnan(obs_val):
                continue

            # If simulated value is NaN, assign large penalty
            if np.isnan(sim_val):
                dist = weights[key] * (1000**2) # Large penalty squared
            else:
                # Calculate weighted squared difference
                # Normalize by observed value to handle different scales? Optional.
                # For now, simple weighted squared difference:
                diff = obs_val - sim_val
                dist = weights[key] * (diff**2)

            total_distance += dist
            valid_keys_count += 1

    # Handle case where no valid keys were found (shouldn't happen ideally)
    if valid_keys_count == 0:
        return np.inf # Return infinity if no comparison possible

    return np.sqrt(total_distance) # Return sqrt for Euclidean distance

# --- Main Execution Block ---
if __name__ == '__main__':

    print("Starting W_n Recovery Validation...")
    print(f"Fixed Parameters: a={TRUE_A}, w_s={TRUE_W_S}")
    print(f"Testing w_n grid: {WN_GRID}")
    print(f"Trials per w_n value: {N_TRIALS_PER_WN}")

    # Store results
    recovery_results = []

    # --- Loop through each true w_n value ---
    for true_wn in WN_GRID:
        print("-" * 50)
        print(f"Processing True w_n = {true_wn:.3f}")
        start_time_wn = time.time()

        # 1. GENERATE SYNTHETIC "OBSERVED" DATA for this true_wn
        np.random.seed(int(true_wn * 100)) # Seed based on w_n for consistency if re-run
        salience_inputs = np.zeros(N_TRIALS_PER_WN)
        norm_inputs = np.zeros(N_TRIALS_PER_WN)
        trial_types = []

        # Create structured Neutral vs High-Conflict trials
        for i in range(N_TRIALS_PER_WN):
            if np.random.rand() < P_HIGH_CONFLICT:
                # High-Conflict Trial (like NoGo: S=1, N=1)
                salience_inputs[i] = 1.0
                norm_inputs[i] = 1.0
                trial_types.append("Conflict")
            else:
                # Neutral Trial (like Go: S=1, N=0)
                salience_inputs[i] = 1.0
                norm_inputs[i] = 0.0
                trial_types.append("Neutral")

        observed_data = []
        agent_obs = MVNESAgent(config={}) # Agent for generating observed data
        current_true_params = {
            'w_n': true_wn,
            'a': TRUE_A,
            'w_s': TRUE_W_S,
            **BASE_SIM_PARAMS
        }
        print(f"Generating observed data with params: {current_true_params}")

        for t in range(N_TRIALS_PER_WN):
            # Parameters need to match expected keys in run_mvnes_trial
            params_for_agent = {
                'w_n': current_true_params['w_n'],
                'threshold_a': current_true_params['a'], # Use 'a' key
                'w_s': current_true_params['w_s'],
                 **BASE_SIM_PARAMS # Pass other fixed params
            }
            trial_data = agent_obs.run_mvnes_trial(
                salience_input=salience_inputs[t],
                norm_input=norm_inputs[t],
                params=params_for_agent
            )
            observed_data.append({
                'subj': 0, # Single subject
                'rt': trial_data['rt'],
                'choice': trial_data['choice'],
                'salience_input': salience_inputs[t],
                'norm_input': norm_inputs[t],
                'trial_type': trial_types[t]
            })

        df_obs = pd.DataFrame(observed_data)
        print(f"Generated {len(df_obs)} trials.")
        print(f"Observed Choice Rate: {df_obs['choice'].mean():.3f}")

        # 2. CALCULATE OBSERVED SUMMARY STATISTICS
        # Need to pass fixed params dict to simulate_summary for internal use if needed,
        # although for calculating stats on df_obs, it's not directly used.
        # We simulate with placeholder params={} because stats are calculated on df_obs.
        obs_summ = simulate_summary({}, current_true_params, salience_inputs, norm_inputs)
        # Hack: Re-calculate directly from df_obs to avoid simulation variance
        obs_summ = simulate_summary({'w_n': true_wn}, # Pass dummy param dict
                                    {'a': TRUE_A, 'w_s': TRUE_W_S, **BASE_SIM_PARAMS}, # Pass fixed params
                                    df_obs['salience_input'].values, # Pass actual inputs used
                                    df_obs['norm_input'].values)     # Pass actual inputs used

        # Re-calculate observed summaries directly from the generated dataframe
        # Create a temporary parameter dict just to pass structure
        # This recalculation directly on df_obs avoids simulating again inside simulate_summary
        # when we just want the stats of the data we *just* generated.
        dummy_params_for_stat_calc = {'w_n': true_wn}
        fixed_params_for_stat_calc = {'a': TRUE_A, 'w_s': TRUE_W_S, **BASE_SIM_PARAMS}
        obs_summ = simulate_summary(dummy_params_for_stat_calc, fixed_params_for_stat_calc,
                                    df_obs['salience_input'].values, df_obs['norm_input'].values)

        print("\nCalculated Observed Summary Statistics:")
        # Print a few key observed stats
        print(f"  Obs Choice Rate: {obs_summ.get('choice_rate', 'N/A'):.3f}")
        print(f"  Obs Choice=1 RT Mean: {obs_summ.get('choice_1_rt_mean', 'N/A'):.3f}")
        print(f"  Obs Choice=0 RT Mean: {obs_summ.get('choice_0_rt_mean', 'N/A'):.3f}")
        # Check for NaNs which indicate potential issues
        if any(np.isnan(v) for v in obs_summ.values()):
             print("WARNING: NaNs detected in observed summary statistics!")


        # 3. SETUP AND RUN ABC-SMC
        print("\nRunning ABC-SMC parameter recovery for w_n...")

        # Define prior for w_n (centered loosely around expected range)
        prior = Distribution(w_n=RV("uniform", 0, 2.0)) # Wider uniform prior

        # Create simulator function correctly binding fixed arguments using partial
        # The simulator needs the fixed parameters (a, w_s, etc.) AND the fixed inputs
        fixed_params_for_abc = {'a': TRUE_A, 'w_s': TRUE_W_S, **BASE_SIM_PARAMS}
        simulator_for_abc = partial(simulate_summary,
                                    fixed_params=fixed_params_for_abc,
                                    salience_inputs=salience_inputs,
                                    norm_inputs=norm_inputs)

        # Setup ABCSMC
        abc = ABCSMC(
            models=simulator_for_abc,
            parameter_priors=prior,
            distance_function=weighted_distance, # Use the custom distance function
            population_size=ABC_POPULATION_SIZE
        )

        # Define database path
        temp_dir = tempfile.gettempdir()
        db_path = f"sqlite:///{os.path.join(temp_dir, f'abc_nes_wn_{true_wn:.2f}.db')}"
        # Ensure a clean run if re-running for the same w_n
        if os.path.exists(db_path.replace("sqlite:///", "")):
             os.remove(db_path.replace("sqlite:///", ""))

        # Run ABC
        abc.new(db_path, obs_summ)
        history = abc.run(minimum_epsilon=ABC_MIN_EPSILON, max_nr_populations=ABC_MAX_NR_POPULATIONS)

        print(f"\nABC run finished for w_n = {true_wn:.3f}.")
        end_time_wn = time.time()
        print(f"Time taken: {end_time_wn - start_time_wn:.1f} seconds.")

        # 4. EXTRACT AND STORE RESULTS
        try:
            df_posterior, w = history.get_distribution(m=0)
            if not df_posterior.empty:
                # Calculate weighted mean and median
                mean_w = np.sum(df_posterior['w_n'] * w)
                # Weighted median requires sorting and cumulative sum
                sorted_indices = np.argsort(df_posterior['w_n'])
                sorted_w_n = df_posterior['w_n'].iloc[sorted_indices]
                sorted_w = w[sorted_indices]
                cum_w = np.cumsum(sorted_w)
                median_idx = np.where(cum_w >= 0.5)[0][0]
                median_w = sorted_w_n.iloc[median_idx]

                # Calculate weighted HDI (using arviz helper if possible, or approx)
                # For simplicity, using pandas describe on weighted samples (approx)
                # A better way uses arviz: az.hdi(df_posterior['w_n'].values, hdi_prob=0.94, weights=w)
                # Let's store basic stats for now
                std_w = np.sqrt(np.sum(w * (df_posterior['w_n'] - mean_w)**2))

                recovery_results.append({
                    'true_wn': true_wn,
                    'recovered_mean': mean_w,
                    'recovered_median': median_w,
                    'recovered_std': std_w,
                    'posterior_df': df_posterior # Store samples for detailed plotting later
                })
                print(f"  Recovered Mean: {mean_w:.3f}, Median: {median_w:.3f}, Std: {std_w:.3f}")
            else:
                print("  ERROR: Posterior distribution is empty!")
                recovery_results.append({
                    'true_wn': true_wn, 'recovered_mean': np.nan,
                    'recovered_median': np.nan, 'recovered_std': np.nan,
                    'posterior_df': pd.DataFrame()
                })
        except Exception as e:
             print(f"  ERROR extracting results for w_n = {true_wn:.3f}: {e}")
             recovery_results.append({
                    'true_wn': true_wn, 'recovered_mean': np.nan,
                    'recovered_median': np.nan, 'recovered_std': np.nan,
                    'posterior_df': pd.DataFrame()
                })


    # --- 5. FINAL PLOTTING ---
    print("-" * 50)
    print("Generating final recovery plot...")

    true_vals = [r['true_wn'] for r in recovery_results]
    recovered_means = [r['recovered_mean'] for r in recovery_results]
    recovered_medians = [r['recovered_median'] for r in recovery_results]
    recovered_stds = [r['recovered_std'] for r in recovery_results]

    plt.figure(figsize=(8, 8))
    plt.errorbar(true_vals, recovered_means, yerr=recovered_stds, fmt='o',
                 label='Recovered Mean ± 1SD', capsize=5, alpha=0.7)
    # plt.scatter(true_vals, recovered_medians, color='green', marker='x', label='Recovered Median')
    
    # Add identity line
    min_val = min(WN_GRID) - 0.1
    max_val = max(WN_GRID) + 0.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line (Perfect Recovery)')

    plt.xlabel("True w_n Value", fontsize=12)
    plt.ylabel("Recovered w_n Value (Posterior Mean)", fontsize=12)
    plt.title("Parameter Recovery Validation for w_n (a, w_s fixed)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box') # Ensure equal scaling
    plt.tight_layout()
    plt.show()

    print("\nValidation script finished.")