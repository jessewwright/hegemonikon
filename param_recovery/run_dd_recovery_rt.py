# Filename: param_recovery/run_dd_recovery_rt.py
# Purpose: Perform full parameter recovery validation for NES DD model
#          using RT-inclusive likelihood on the complete synthetic dataset.

import numpy as np
import pandas as pd
import time
import os
from scipy.optimize import minimize
from scipy.stats import binom
import matplotlib.pyplot as plt # Import for plotting

# --- Component Definitions ---
# (Pasting classes again for self-contained execution)
class Comparator:
    """ NES Comparator Module """
    def __init__(self, dt=0.01, noise_std_dev=0.1):
        if dt <= 0: raise ValueError("dt must be positive.")
        if noise_std_dev < 0: raise ValueError("noise cannot be negative.")
        self.dt = dt
        self.noise_std_dev = noise_std_dev
        self.sqrt_dt = np.sqrt(dt)
        self.evidence = {}
        self.time_elapsed = 0.0
    def reset(self):
        self.evidence = {}
        self.time_elapsed = 0.0
    def initialize_actions(self, actions):
        self.reset()
        self.evidence = {action: 0.0 for action in actions}
    def calculate_drift_rate(self, action_attributes, params):
        S = action_attributes.get('S', 0.0)
        N = action_attributes.get('N', 0.0)
        U = action_attributes.get('U', 0.0)
        w_s = params.get('w_s', 1.0)
        w_n = params.get('w_n', 0.0)
        w_u = params.get('w_u', 0.0)
        drift = (w_s * S) + (w_n * N) + (w_u * U)
        return drift
    def step(self, action_attributes_dict, params):
        if not self.evidence: return {}
        current_noise_std = params.get('noise_std_dev', self.noise_std_dev)
        for action, current_evidence in self.evidence.items():
            if action not in action_attributes_dict: continue
            attributes = action_attributes_dict[action]
            drift = self.calculate_drift_rate(attributes, params)
            noise = np.random.normal(0, current_noise_std) * self.sqrt_dt
            self.evidence[action] += drift * self.dt + noise
        self.time_elapsed += self.dt
        return self.evidence.copy()

class AssentGate:
    """ NES Assent Gate """
    def __init__(self, base_threshold=1.0):
        if base_threshold <= 0: raise ValueError("Base threshold must be positive.")
        self.initial_base_threshold = base_threshold
    def check(self, evidence_dict, current_threshold):
        if current_threshold <= 0: current_threshold = 0.01
        winning_action = None
        max_evidence = -float('inf')
        for action, evidence in evidence_dict.items():
            if evidence >= current_threshold:
                 if evidence > max_evidence:
                    max_evidence = evidence
                    winning_action = action
        return winning_action

print("Components defined.")

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    if k < 0: k = 1e-6
    return amount / (1.0 + k * delay)

# --- Single Trial Simulation Function ---
def run_single_dd_trial(comparator, assent_gate, params, ss_option, ll_option):
    """ Runs a single DD trial using EXISTING NES component instances. """
    k = params['k_discount']
    v_ss = hyperbolic_discount(ss_option['amount'], ss_option['delay'], k)
    v_ll = hyperbolic_discount(ll_option['amount'], ll_option['delay'], k)
    actions = ['Choose_LL', 'Choose_SS']
    action_attributes = {
        'Choose_LL': {'S': v_ll, 'N': 0, 'U': 0},
        'Choose_SS': {'S': v_ss, 'N': 0, 'U': 0}
    }
    comparator.initialize_actions(actions)
    accumulated_time = 0.0
    decision = None
    current_threshold = params['base_threshold']
    while accumulated_time < params['max_time']:
        current_evidence = comparator.step(action_attributes, params)
        decision = assent_gate.check(current_evidence, current_threshold)
        if decision is not None: break
        accumulated_time += params['dt']
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision if decision is not None else 'timeout'
    return {'choice': choice, 'rt': rt} # Keep return simple for fitting

print("Simulation function defined.")

# --- Load Full Synthetic Data ---
data_filename = "synthetic_data_NEW.csv"
true_params_filename = "true_parameters_NEW.csv"

try:
    synthetic_data_df = pd.read_csv(data_filename)
    true_params_df = pd.read_csv(true_params_filename)
    print(f"Successfully loaded data ({len(synthetic_data_df)} trials)")
    print(f"Successfully loaded true parameters for {len(true_params_df)} subjects")
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    print("Ensure synthetic_data_NEW.csv and true_parameters_NEW.csv are in the param_recovery directory.")
    # sys.exit(1) # Avoid exit in potentially non-interactive environments

# --- RT-Inclusive Likelihood Function ---
N_SIMS_PER_CONFIG_FIT = 200 # Scaled up simulations
RT_QUANTILES = [25, 50, 75] # 25th, 50th, and 75th percentiles
RT_WEIGHT = 1.0 # Default weight, can be tuned

def get_quantiles(rts):
    """Helper to get RT quantiles, handles insufficient data."""
    if len(rts) < 2:
        return [np.nan] * len(RT_QUANTILES)
    # Ensure input is numpy array for percentile function
    rts_array = np.asarray(rts)
    # Check for NaNs or infinite values in RTs which can cause errors
    if np.any(np.isnan(rts_array)) or np.any(np.isinf(rts_array)):
        return [np.nan] * len(RT_QUANTILES)
    return [np.percentile(rts_array, q) for q in RT_QUANTILES]


def calculate_nll_rt_quantiles(params_to_fit, subject_data, fixed_params, ss_option, ll_amount, all_delays):
    """ Calculates NLL using choice likelihood + RT quantile SSE. """
    k_candidate, thresh_candidate = params_to_fit
    if k_candidate <= 0 or thresh_candidate <= 0.1: return np.inf

    current_params = fixed_params.copy()
    current_params['k_discount'] = k_candidate
    current_params['base_threshold'] = thresh_candidate

    try:
        comparator_fit = Comparator(dt=current_params['dt'], noise_std_dev=current_params['noise_std_dev'])
        assent_gate_fit = AssentGate(base_threshold=current_params['base_threshold'])
    except Exception: return np.inf

    total_choice_loglik = 0.0
    total_rt_sse = 0.0
    sse_terms = 0

    for delay in all_delays:
        delay_data_obs = subject_data[subject_data['ll_delay'] == delay]
        if delay_data_obs.empty: continue

        n_total_obs = len(delay_data_obs)
        n_ll_chosen_obs = (delay_data_obs['choice'] == 'Choose_LL').sum()

        sim_choices = []
        sim_rts_ll = []
        sim_rts_ss = []
        ll_option_fit = {'amount': ll_amount, 'delay': delay}
        for _ in range(N_SIMS_PER_CONFIG_FIT):
            result = run_single_dd_trial(comparator_fit, assent_gate_fit, current_params, ss_option, ll_option_fit)
            sim_choices.append(result['choice'])
            if result['choice'] == 'Choose_LL': sim_rts_ll.append(result['rt'])
            elif result['choice'] == 'Choose_SS': sim_rts_ss.append(result['rt'])

        p_ll_sim = sum(c == 'Choose_LL' for c in sim_choices) / N_SIMS_PER_CONFIG_FIT if N_SIMS_PER_CONFIG_FIT > 0 else 0
        
        epsilon = 1e-6
        p_ll_sim_clipped = np.clip(p_ll_sim, epsilon, 1.0 - epsilon)

        # Handle cases where n_ll_chosen_obs might be outside [0, n_total_obs] if data is odd
        # Though it shouldn't be with generated data. Clamp just in case.
        n_ll_chosen_obs_clipped = np.clip(n_ll_chosen_obs, 0, n_total_obs)

        # Use try-except for binomial likelihood calculation
        try:
             choice_loglik_delay = binom.logpmf(n_ll_chosen_obs_clipped, n_total_obs, p_ll_sim_clipped)
             # Check for NaN/inf again
             if np.isnan(choice_loglik_delay) or np.isinf(choice_loglik_delay):
                 choice_loglik_delay = -np.inf # Assign large negative value (bad fit)
        except ValueError:
             choice_loglik_delay = -np.inf # Assign large negative value if binom fails

        total_choice_loglik += choice_loglik_delay

        # Get RT quantiles for this delay condition
        obs_rts_ll = delay_data_obs[delay_data_obs['choice'] == 'Choose_LL']['rt'].values
        obs_rts_ss = delay_data_obs[delay_data_obs['choice'] == 'Choose_SS']['rt'].values
        obs_q_ll = get_quantiles(obs_rts_ll)
        obs_q_ss = get_quantiles(obs_rts_ss)
        sim_q_ll = get_quantiles(sim_rts_ll)
        sim_q_ss = get_quantiles(sim_rts_ss)

        # Calculate SSE for LL RTs if we have enough data
        if not np.isnan(obs_q_ll).any() and not np.isnan(sim_q_ll).any():
            sse_ll = np.sum((np.array(obs_q_ll) - np.array(sim_q_ll))**2)
            total_rt_sse += sse_ll
            sse_terms += len(RT_QUANTILES)

        # Calculate SSE for SS RTs if we have enough data
        if not np.isnan(obs_q_ss).any() and not np.isnan(sim_q_ss).any():
            sse_ss = np.sum((np.array(obs_q_ss) - np.array(sim_q_ss))**2)
            total_rt_sse += sse_ss
            sse_terms += len(RT_QUANTILES)

    # If total_choice_loglik became -inf, return inf for minimization
    if np.isinf(total_choice_loglik) and total_choice_loglik < 0:
        return np.inf

    # Combine: Minimize (-ChoiceLogLik + Weighted_RT_SSE)
    rt_penalty = 0
    # Add penalty if SSE couldn't be computed reliably across many terms?
    # e.g., if sse_terms < expected_number_of_quantiles
    # For now, assume RT_WEIGHT handles importance.

    combined_nll = -(total_choice_loglik) + RT_WEIGHT * total_rt_sse + rt_penalty

    if np.isnan(combined_nll) or np.isinf(combined_nll): return np.inf
    return combined_nll

# --- Optimization Loop ---
recovered_params = []
# Ensure dataframes are loaded before defining all_subjects
if 'synthetic_data_df' in locals() and not synthetic_data_df.empty:
    all_subjects = sorted(synthetic_data_df['subject'].unique())

    # Define fixed parameters for fitting
    fixed_params_for_fit = {
        'dt': 0.01,
        'noise_std_dev': 0.237,
        'w_s': 0.392,
        'max_time': 5.0
    }

    # Define simulation parameters
    N_SIMS_PER_CONFIG_FIT = 200
    RT_WEIGHT = 1.0

    # Define optimization parameters
    OPTIMIZATION_OPTIONS = {
        'maxiter': 100,
        'ftol': 1e-5
    }

    # Define RT quantiles
    RT_QUANTILES = [10, 30, 50, 70, 90]  # More quantiles for better RT distribution matching

    # Define options and delays needed by NLL function
    ss_option_fit = {'amount': 5, 'delay': 0}
    ll_amount_fit = 10
    ll_delays_fit = sorted(synthetic_data_df['ll_delay'].unique())
    bounds_fit = [(1e-6, 1.0), (0.1, 2.0)] # k bounds, threshold bounds

    print(f"\nStarting RT-inclusive parameter recovery for {len(all_subjects)} subjects...")
    start_fit_time = time.time()

    for subj_id in all_subjects:
        subject_data = synthetic_data_df[synthetic_data_df['subject'] == subj_id].copy()
        x0 = [0.05, 0.6] # Initial guess

        objective_func = lambda p: calculate_nll_rt_quantiles(p, subject_data, fixed_params_for_fit, ss_option_fit, ll_amount_fit, ll_delays_fit)

        opt_result = minimize(
            objective_func,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds_fit,
            options=OPTIMIZATION_OPTIONS
        )

        if opt_result.success:
            rec_k, rec_thresh = opt_result.x
            recovered_params.append({'subject': subj_id, 'recovered_k': rec_k, 'recovered_threshold': rec_thresh, 'nll': opt_result.fun, 'success': True})
        else:
            recovered_params.append({'subject': subj_id, 'recovered_k': np.nan, 'recovered_threshold': np.nan, 'nll': np.nan, 'success': False})

        if (subj_id + 1) % 5 == 0: # Adjust modulo if N_SUBJECTS changed
             print(f"  Finished fitting subject {subj_id + 1}/{len(all_subjects)} (Success: {opt_result.success})")

    end_fit_time = time.time()
    print(f"\nParameter recovery finished in {end_fit_time - start_fit_time:.2f} seconds.")

    # --- Comparison ---
    recovered_params_df = pd.DataFrame(recovered_params)
    print("\n--- Sample Recovered Parameters (First 5 subjects) ---")
    print(recovered_params_df.head().round(4))

    comparison_df = pd.merge(true_params_df, recovered_params_df, on='subject')
    print("\n--- Recovery Success Rate ---")
    success_rate = comparison_df['success'].mean()
    print(f"{success_rate*100:.1f}% of fits converged successfully.")

    print("\n--- Correlations (for successful fits) ---")
    successful_fits = comparison_df[comparison_df['success']].copy() # Filter for successful fits
    if not successful_fits.empty and len(successful_fits) > 1:
        # Add checks for near-constant columns which break correlation
        if successful_fits['true_k'].std() > 1e-6 and successful_fits['recovered_k'].std() > 1e-6:
             corr_k = successful_fits['true_k'].corr(successful_fits['recovered_k'])
             print(f"Correlation(True K, Recovered K): {corr_k:.3f}")
        else:
             print("Correlation(True K, Recovered K): Cannot compute (constant values).")

        if successful_fits['true_threshold'].std() > 1e-6 and successful_fits['recovered_threshold'].std() > 1e-6:
             corr_thresh = successful_fits['true_threshold'].corr(successful_fits['recovered_threshold'])
             print(f"Correlation(True Thresh, Recovered Thresh): {corr_thresh:.3f}")
        else:
            print("Correlation(True Thresh, Recovered Thresh): Cannot compute (constant values).")

        print(f"\nSUCCESS METRIC CHECK: Aiming for r > 0.7")
    else:
        print("Not enough successful fits to calculate correlations.")

    # --- Plotting ---
    print("\n--- Generating Recovery Plots ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    if not successful_fits.empty:
        # Define limits for plots based on true and recovered ranges
        min_k = min(successful_fits['true_k'].min(), successful_fits['recovered_k'].min()) * 0.9
        max_k = max(successful_fits['true_k'].max(), successful_fits['recovered_k'].max()) * 1.1
        min_t = min(successful_fits['true_threshold'].min(), successful_fits['recovered_threshold'].min()) * 0.9
        max_t = max(successful_fits['true_threshold'].max(), successful_fits['recovered_threshold'].max()) * 1.1

        # Plot K
        axes[0].scatter(successful_fits['true_k'], successful_fits['recovered_k'], alpha=0.7)
        axes[0].plot([min_k, max_k], [min_k, max_k], 'r--', label='Identity')
        axes[0].set_xlabel("True k")
        axes[0].set_ylabel("Recovered k")
        axes[0].set_title("K Recovery")
        if 'corr_k' in locals():
            axes[0].text(0.05, 0.95, f"r = {corr_k:.3f}", transform=axes[0].transAxes, verticalalignment='top')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(left=max(0, min_k)) # Ensure k starts at 0 or just above
        axes[0].set_ylim(bottom=max(0, min_k))

        # Plot Threshold
        axes[1].scatter(successful_fits['true_threshold'], successful_fits['recovered_threshold'], alpha=0.7)
        axes[1].plot([min_t, max_t], [min_t, max_t], 'r--', label='Identity')
        axes[1].set_xlabel("True Threshold")
        axes[1].set_ylabel("Recovered Threshold")
        axes[1].set_title("Threshold Recovery")
        if 'corr_thresh' in locals():
            axes[1].text(0.05, 0.95, f"r = {corr_thresh:.3f}", transform=axes[1].transAxes, verticalalignment='top')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim(left=max(0.1, min_t)) # Ensure threshold starts >= 0.1
        axes[1].set_ylim(bottom=max(0.1, min_t))

    else:
        axes[0].text(0.5, 0.5, 'No successful fits for K plot', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No successful fits for Threshold plot', ha='center', va='center')
        axes[0].set_title("K Recovery")
        axes[1].set_title("Threshold Recovery")

    plt.suptitle('Parameter Recovery Results (RT-Inclusive Likelihood)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the plot
    plot_filename = "parameter_recovery_results_rt.png"
    try:
        plt.savefig(plot_filename)
        print(f"\nRecovery plot saved to {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    # plt.show() # Might not work in all environments

else:
    print("Script halted because dataframes could not be loaded.")

print("\nFull parameter recovery script finished.")