# Filename: generate_dd_data_with_rt.py
# Purpose: Generate synthetic Delay Discounting data (choice + RT)
#          based on simplified NES DDM principles for parameter recovery.

import numpy as np
import pandas as pd
import random
import time

print("Starting Synthetic Data Generation (Choice + RT)...")

# --- Parameters for Generation Process ---
# Fixed DDM parameters (should match assumptions in fitting script)
PARAMS_FIXED_GEN = {
    'noise_std_dev': 0.237, # DDM noise sigma (scaled for dt in simulation)
    'w_s': 0.392,          # Weight mapping value diff to drift
    'dt': 0.01,            # Simulation time step
    'max_time': 5.0        # Max RT allowed
}

# Population distributions for subject parameters
TRUE_K_MEAN = 0.05; TRUE_K_SD = 0.04
TRUE_THRESH_MEAN = 0.6; TRUE_THRESH_SD = 0.2
TRUE_T_MEAN = 0.15; TRUE_T_SD = 0.05 # Non-decision time

# --- Helper Functions ---
def hyperbolic_discount_np(amount, delay, k):
    """Standard numpy hyperbolic discount function."""
    k_safe = max(k, 1e-9)
    return amount / (1.0 + k_safe * delay)

# --- Simplified DDM Simulation for a Single Trial ---
def simulate_ddm_trial(params, ss_option, ll_option):
    """
    Simulates one DDM trial based on drift, threshold, t, noise.
    Returns choice (0=SS, 1=LL) and RT.
    Uses Euler-Maruyama approximation.
    """
    k = params['k_discount']
    a = params['threshold'] # Boundary separation (use 'threshold' from params)
    t = params['t']         # Non-decision time
    w_s = params['w_s']
    sigma = params['noise_std_dev'] # Base noise level
    dt = params['dt']
    max_time = params['max_time']

    # Calculate drift rate based on value difference
    v_ss = hyperbolic_discount_np(ss_option['amount'], ss_option['delay'], k)
    v_ll = hyperbolic_discount_np(ll_option['amount'], ll_option['delay'], k)
    drift_rate = w_s * (v_ll - v_ss) # Drift towards LL boundary if v_ll > v_ss

    # DDM simulation loop
    evidence = 0.0 # Start at 0 (unbiased)
    decision_time = 0.0
    noise_scaler = sigma * np.sqrt(dt) # Pre-scale noise std dev
    upper_bound = a / 2.0 # Assuming symmetric bounds around 0
    lower_bound = -a / 2.0

    while accumulated_time < (max_time - t): # Only simulate decision time
        noise = np.random.normal(0, noise_scaler)
        evidence += drift_rate * dt + noise
        accumulated_time += dt

        if evidence >= upper_bound:
            choice = 1 # Choose LL
            decision_time = accumulated_time
            break
        elif evidence <= lower_bound:
            choice = 0 # Choose SS
            decision_time = accumulated_time
            break
    else: # Loop finished without crossing boundary
        choice = -1 # Indicate timeout
        decision_time = max_time - t

    rt = decision_time + t
    # Ensure RT is not exactly 0 or negative, and not > max_time
    rt = max(dt, min(rt, max_time))

    # Map choice code back to names for clarity if needed, or keep 0/1
    # choice_name = 'Choose_LL' if choice == 1 else ('Choose_SS' if choice == 0 else 'timeout')

    return {'choice': choice, 'rt': rt}


# --- Generate True Parameters for Subjects ---
def generate_true_parameters(n_subjects):
    true_params = []
    # Recalculate Gamma params for k
    k_var = max(1e-6, TRUE_K_SD**2)
    k_mean_safe = max(1e-6, TRUE_K_MEAN)
    gamma_k_scale = k_var / k_mean_safe
    gamma_k_shape = k_mean_safe / gamma_k_scale

    for subj_id in range(n_subjects):
        # Sample k from Gamma
        k = np.random.gamma(shape=gamma_k_shape, scale=gamma_k_scale)
        # Sample threshold from Normal, ensure positive
        threshold = np.random.normal(loc=TRUE_THRESH_MEAN, scale=TRUE_THRESH_SD)
        threshold = max(0.1, threshold) # Ensure > 0.1
        # Sample non-decision time from Normal, ensure positive
        t = np.random.normal(loc=TRUE_T_MEAN, scale=TRUE_T_SD)
        t = max(PARAMS_FIXED_GEN['dt'], t) # Ensure t >= dt

        true_params.append({
            'subject': subj_id,
            'true_k': k,
            'true_threshold': threshold,
            'true_t': t # Include non-decision time
        })
    return pd.DataFrame(true_params)

# --- Generate Synthetic Data (Choices and RTs) ---
def generate_synthetic_data(true_params_df, n_trials_per_delay=20):
    all_trial_data = []
    ss_option = {'amount': 5, 'delay': 0}
    ll_amount = 10
    delays = [1, 3, 5, 10, 20, 30, 50]

    total_trials_expected = len(true_params_df) * len(delays) * n_trials_per_delay
    print(f"Generating {total_trials_expected} total trials...")

    start_gen_time = time.time()
    for _, row in true_params_df.iterrows():
        subj_id = int(row['subject'])
        subj_params = PARAMS_FIXED_GEN.copy()
        subj_params['k_discount'] = row['true_k']
        subj_params['threshold'] = row['true_threshold']
        subj_params['t'] = row['true_t']

        for delay in delays:
            ll_option = {'amount': ll_amount, 'delay': delay}
            for rep in range(n_trials_per_delay):
                # Simulate one trial using the DDM function
                result = simulate_ddm_trial(subj_params, ss_option, ll_option)

                if result['choice'] != -1: # Exclude timeouts from saved data
                    all_trial_data.append({
                        'subject': subj_id,
                        'll_delay': delay,
                        'choice': result['choice'], # Store as 0 or 1
                        'rt': result['rt'],
                        # Optionally store true params for verification within data file
                        # 'true_k': row['true_k'],
                        # 'true_threshold': row['true_threshold'],
                        # 'true_t': row['true_t']
                    })
        if (subj_id + 1) % 5 == 0 and delay == delays[-1]: # Progress indicator
             print(f"  Finished generating for subject {subj_id + 1}")

    end_gen_time = time.time()
    print(f"Data generation loop took {end_gen_time - start_gen_time:.2f} seconds.")
    return pd.DataFrame(all_trial_data)

# --- Main ---
if __name__ == "__main__":
    # Settings
    N_SUBJECTS = 20          # Number of subjects to generate
    N_TRIALS_PER_DELAY = 30  # Number of trials per delay point per subject

    output_data_filename = 'synthetic_data_rt.csv'
    output_params_filename = 'true_parameters_rt.csv'

    print(f"Generating synthetic data for {N_SUBJECTS} subjects...")
    true_params_df = generate_true_parameters(N_SUBJECTS)
    synthetic_data_df = generate_synthetic_data(true_params_df, n_trials_per_delay=N_TRIALS_PER_DELAY)

    # Save the data
    try:
        true_params_df.to_csv(output_params_filename, index=False, float_format='%.6f')
        synthetic_data_df.to_csv(output_data_filename, index=False, float_format='%.6f')
        print(f"\nGenerated {len(synthetic_data_df)} valid trials.")
        print(f"True parameters saved to '{output_params_filename}'")
        print(f"Synthetic choices and RTs saved to '{output_data_filename}'")

        print("\nSample of generated data:")
        print(synthetic_data_df.head())
        print("\nTrue parameters of first 5 subjects:")
        print(true_params_df.head().round(4))

    except Exception as e:
        print(f"\nError saving files: {e}")

    print("\nScript finished.")