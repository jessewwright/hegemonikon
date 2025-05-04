# Filename: param_recovery/generate_dissociative_dd_data.py
# Purpose: Generate synthetic data for a Delay Discounting task variant
#          designed to dissociate drift (k) and threshold (a) parameters.
#          Includes Value-Focus blocks and Caution/Speed-Focus blocks.

import numpy as np
import pandas as pd
import time
import itertools # For creating block sequences
import os
import sys

# --- Add project root to path to import NES components ---
# Assuming this script is run from param_recovery directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Try importing NES components ---
try:
    # Assuming components are structured within an 'nes' directory
    # Adjust path if needed based on your structure
    from nes.nes_sim_dd import Comparator, AssentGate, run_single_dd_trial, hyperbolic_discount_np
    print("Successfully imported NES components from nes.nes_sim_dd")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Could not import required components (Comparator, AssentGate, run_single_dd_trial).")
    print("Ensure nes/nes_sim_dd.py exists and defines these components.")
    # Define dummy components/functions if import fails, just to allow script structure check
    print("Defining dummy components/functions to proceed...")
    class Comparator:
        def __init__(self, dt, noise_std_dev): pass
        def initialize_actions(self, actions): pass
        def step(self, action_attributes, params): return {'Choose_LL': 0.1, 'Choose_SS': 0.0} # Dummy evidence
    class AssentGate:
        def __init__(self, base_threshold): pass
        def check(self, evidence, threshold): return 'Choose_SS' if np.random.rand() < 0.5 else 'Choose_LL' # Random choice
    def run_single_dd_trial(comparator, assent_gate, params, ss_option, ll_option): # Dummy function
        return {'choice': 'Choose_SS' if np.random.rand() < 0.5 else 'Choose_LL', 'rt': np.random.uniform(0.1, 2.0)}
    def hyperbolic_discount_np(amount, delay, k): return amount / (1.0 + max(k, 1e-7) * delay) # Dummy function


# --- Simulation Parameters ---

# Fixed DDM parameters (Consistent with previous recovery attempt)
params_fixed_gen = {
    'noise_std_dev': 0.237,
    'w_s': 0.392, # Weight for value difference driving drift
    'w_n': 0.0,   # Assume no norm component for standard DD
    'w_u': 0.0,   # Assume no urgency component
    'dt': 0.01,
    'max_time': 5.0
}

# Population distributions for varying parameters (k and threshold a)
N_SUBJECTS = 15 # Target for pilot data
true_k_mean = 0.05; true_k_sd = 0.04 # Slightly wider range for k
true_thresh_mean = 0.6; true_thresh_sd = 0.2 # Slightly wider range for threshold

# Recalculate Gamma params for k (ensure positivity)
true_k_var = max(1e-6, true_k_sd**2) # Prevent zero variance
true_k_mean_safe = max(1e-6, true_k_mean) # Prevent zero mean
gamma_k_scale = true_k_var / true_k_mean_safe
gamma_k_shape = true_k_mean_safe / gamma_k_scale

# Task Design
# Block Types: 'ValueFocus', 'SpeedFocus', 'CautionFocus'
# ValueFocus: Vary delays widely, standard instructions
# SpeedFocus: Constant intermediate delay, speed instructions (lower threshold expected)
# CautionFocus: Constant intermediate delay, caution instructions (higher threshold expected)

# ValueFocus Block Parameters
vf_ss_option = {'amount': 5, 'delay': 0}
vf_ll_amount = 10
vf_ll_delays = [1, 5, 10, 20, 40, 60] # Wider range, including longer delays
vf_reps_per_delay = 10 # Repetitions per delay point within the block

# Constant-Value Block Parameters (for Speed/Caution)
cv_ss_option = {'amount': 5, 'delay': 0}
cv_ll_amount = 10
cv_ll_delay = 25 # Choose an intermediate delay where choice is often indifferent
cv_reps = 30 # Number of trials in each speed/caution block

# Block sequence (Example: ABACABAC...)
block_types = ['ValueFocus', 'SpeedFocus', 'ValueFocus', 'CautionFocus']
n_block_repeats = 3 # Repeat the sequence 3 times
total_blocks = len(block_types) * n_block_repeats

# --- Data Generation Loop ---
all_subject_data = []
true_subject_params_list = []
np.random.seed(2025) # Seed for reproducibility

print(f"Generating dissociative DD data for {N_SUBJECTS} subjects...")

for subj_id in range(N_SUBJECTS):
    # Sample true *base* parameters for this subject
    subj_k = np.random.gamma(shape=gamma_k_shape, scale=gamma_k_scale)
    subj_base_thresh = np.random.normal(loc=true_thresh_mean, scale=true_thresh_sd)
    subj_base_thresh = max(0.15, subj_base_thresh) # Ensure threshold is reasonably positive

    # Define hypothetical instruction effects on threshold
    # These are part of the 'true' generating process we want to recover evidence for
    thresh_effect_speed = -0.15 # Speed instruction lowers threshold (relative to base)
    thresh_effect_caution = +0.15 # Caution instruction raises threshold (relative to base)

    true_subject_params_list.append({
        'subject': subj_id,
        'true_k': subj_k,
        'true_base_threshold': subj_base_thresh,
        'true_thresh_speed': max(0.05, subj_base_thresh + thresh_effect_speed), # Ensure >0
        'true_thresh_caution': max(0.05, subj_base_thresh + thresh_effect_caution)
    })

    # Initialize components for this subject
    # Noise and w_s are fixed across blocks and subjects in this design
    try:
        comparator = Comparator(dt=params_fixed_gen['dt'], noise_std_dev=params_fixed_gen['noise_std_dev'])
        # AssentGate base threshold will be overridden per block
        assent_gate = AssentGate(base_threshold=subj_base_thresh)
    except Exception as e:
        print(f"Error initializing components for subject {subj_id}: {e}")
        continue

    subj_data = []
    block_counter = 0
    for i_repeat in range(n_block_repeats):
        for block_type in block_types:
            block_counter += 1
            print(f"  Subject {subj_id+1}, Block {block_counter}/{total_blocks} ({block_type})...")
            block_params = params_fixed_gen.copy()
            block_params['k_discount'] = subj_k # Use subject's true k

            # Set threshold based on block type
            if block_type == 'ValueFocus':
                block_params['base_threshold'] = subj_base_thresh
                current_delays = vf_ll_delays
                current_reps = vf_reps_per_delay
                current_ss = vf_ss_option
                current_ll_amount = vf_ll_amount
            elif block_type == 'SpeedFocus':
                block_params['base_threshold'] = max(0.05, subj_base_thresh + thresh_effect_speed) # Apply speed effect
                current_delays = [cv_ll_delay]
                current_reps = cv_reps
                current_ss = cv_ss_option
                current_ll_amount = cv_ll_amount
            elif block_type == 'CautionFocus':
                block_params['base_threshold'] = max(0.05, subj_base_thresh + thresh_effect_caution) # Apply caution effect
                current_delays = [cv_ll_delay]
                current_reps = cv_reps
                current_ss = cv_ss_option
                current_ll_amount = cv_ll_amount

            # Simulate trials within the block
            trial_count_in_block = 0
            for delay in current_delays:
                ll_option = {'amount': current_ll_amount, 'delay': delay}
                for rep in range(current_reps):
                    trial_result = run_single_dd_trial(
                        comparator, assent_gate, block_params, current_ss, ll_option
                    )
                    # Add metadata
                    trial_result['subject'] = subj_id
                    trial_result['block_type'] = block_type
                    trial_result['block_num'] = block_counter
                    trial_result['trial_in_block'] = trial_count_in_block + 1
                    trial_result['ll_delay'] = delay
                    trial_result['effective_threshold'] = block_params['base_threshold'] # Store threshold used
                    # Remove timeouts if any occurred
                    if trial_result['choice'] != 'timeout':
                        subj_data.append(trial_result)
                    trial_count_in_block += 1

    all_subject_data.extend(subj_data)
    print(f"  Finished Subject {subj_id + 1}. Total valid trials: {len(subj_data)}")


# --- Combine and Save ---
dissociative_data_df = pd.DataFrame(all_subject_data)
dissociative_true_params_df = pd.DataFrame(true_subject_params_list)

print(f"\nGenerated {len(dissociative_data_df)} total valid trials across {N_SUBJECTS} subjects.")

# Display sample data and true parameters
print("\n--- Sample Generated Data (First 10 rows) ---")
print(dissociative_data_df.head(10).round(3))

print("\n--- True Base Parameters (First 5 Subjects) ---")
print(dissociative_true_params_df[['subject', 'true_k', 'true_base_threshold']].head(5).round(3))

# Save the generated data
output_dir = "." # Save in current directory for simplicity now
data_filename_out = os.path.join(output_dir, "synthetic_dissociative_dd_data.csv")
true_params_filename_out = os.path.join(output_dir, "true_dissociative_params.csv")

try:
    dissociative_data_df.to_csv(data_filename_out, index=False)
    dissociative_true_params_df.to_csv(true_params_filename_out, index=False)
    print(f"\nSynthetic dissociative data saved to {data_filename_out}")
    print(f"True parameters saved to {true_params_filename_out}")
except Exception as e:
    print(f"\nError saving files: {e}")

print("\nScript finished.")