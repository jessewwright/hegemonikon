# Filename: param_recovery/run_dd_recovery_hddm_proto.py
# Purpose: Prototype HBM fit for NES DD model using HDDM library

import hddm
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

print(f"HDDM version: {hddm.__version__}")

# --- Helper Functions ---
# Use standard Python math here, not PyTensor
def hyperbolic_discount_np(amount, delay, k):
    k_safe = max(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

# --- Data Generation or Loading ---
data_filename = "synthetic_data_NEW.csv"
true_params_filename = "true_parameters_NEW.csv"

# Check if files exist, otherwise generate demo data
if os.path.exists(data_filename) and os.path.exists(true_params_filename):
    print(f"Loading existing data from {data_filename} and {true_params_filename}")
    synthetic_data_df = pd.read_csv(data_filename)
    true_params_df = pd.read_csv(true_params_filename)
    print(f"Loaded {len(synthetic_data_df)} trials for {true_params_df['subject'].nunique()} subjects.")
    # Use only the first 5 subjects for the prototype
    if true_params_df['subject'].nunique() > 5:
        print("Using data from first 5 subjects for prototype fit.")
        subject_ids_to_keep = true_params_df['subject'].unique()[:5]
        synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'].isin(subject_ids_to_keep)].copy()
        true_params_df = true_params_df[true_params_df['subject'].isin(subject_ids_to_keep)].copy()
else:
    print("ERROR: Data files not found. Please generate them first.")
    # Cannot generate easily without the full simulation code here.
    # Exit if files not found for this prototype.
    exit()


# --- Data Preprocessing for HDDM ---
# Filter RTs
min_rt_threshold = 0.05
print(f"\nInitial trials: {len(synthetic_data_df)}")
synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
print(f"Trials after filtering RT > {min_rt_threshold}s: {len(synthetic_data_df)}")

if synthetic_data_df.empty:
    print("Error: No valid trials remain after filtering RTs. Cannot proceed.")
    exit()

# HDDM requires specific column names and coding:
# subj_idx: integer subject ID
# response: 0 (lower boundary/SS) or 1 (upper boundary/LL)
# rt: reaction time (positive)

# Create integer subject index if 'subject' isn't already 0-based integer
synthetic_data_df['subj_idx'], _ = pd.factorize(synthetic_data_df['subject'])

# Create response column (1 for LL, 0 for SS)
synthetic_data_df['response'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, 0)

# --- Defining the Custom Drift Component for HDDM ---
# The challenge: drift v depends on k, which we estimate.
# Approach 1: Use HDDMStimCoding. Requires value columns in the data.
# We need a column representing the stimulus strength, which for us is related to V_LL - V_SS.
# Let's calculate this difference using a *fixed, approximate* k (e.g., group mean)
# This is NOT ideal, as the true k varies. But it's a starting point for HDDMStimCoding.

approx_k = 0.04 # Use approximate group mean k for calculating value diff
w_s_fixed = 0.392 # Fixed w_s from generation
ss_amount = 5; ss_delay = 0; ll_amount = 10

def calculate_nes_drift_input(row):
    v_ss = hyperbolic_discount_np(ss_amount, ss_delay, approx_k)
    v_ll = hyperbolic_discount_np(ll_amount, row['ll_delay'], approx_k)
    # The 'stimulus' for HDDMStimCoding's drift is often the value difference
    # We also incorporate the fixed w_s here.
    return w_s_fixed * (v_ll - v_ss)

synthetic_data_df['nes_drift_input'] = synthetic_data_df.apply(calculate_nes_drift_input, axis=1)

print("\nSample data prepared for HDDM:")
print(synthetic_data_df[['subj_idx', 'rt', 'response', 'll_delay', 'nes_drift_input']].head())

# --- HDDM Model Specification ---
# We want v to depend on our calculated 'nes_drift_input'.
# We want a (threshold) and t (non-decision time) to vary by subject.
# We ALSO want to estimate k hierarchically. HDDM doesn't have 'k' natively.
# Option: Add 'k' as a regression parameter affecting 'v'? This is complex.

# Let's try a standard HDDM fit first, estimating v, a, t hierarchically,
# where 'v' depends on our approximate 'nes_drift_input'.
# This WON'T recover the true 'k', but tests if HDDM runs.

print("\nDefining HDDM model (Estimating v, a, t - v depends on approx value diff)...")

try:
    # Model where v depends on the pre-calculated drift input signal
    # Threshold 'a' and non-decision time 't' vary by subject
    # Bias 'z' is fixed at 0.5 (unbiased)
    m = hddm.HDDM(synthetic_data_df,
                  depends_on={'v': 'nes_drift_input', 'a': 'subj_idx', 't': 'subj_idx'},
                  include=['v', 'a', 't'],
                  p_outlier=0.05) # Include outlier probability

    # --- Run MCMC Sampler (Short run for prototype) ---
    print("Starting HDDM sampling (Prototype)...")
    start_sample_time = time.time()
    n_samples = 500 # Total samples
    n_burn = 200  # Burn-in

    m.sample(n_samples, burn=n_burn, dbname='hddm_proto.db', db='pickle')

    end_sample_time = time.time()
    print(f"Sampling finished in {end_sample_time - start_sample_time:.2f} seconds.")

    # --- Basic Convergence Check & Summary ---
    print("\n--- HDDM Model Summary (Group Parameters) ---")
    m.print_stats() # Prints group means and standard deviations

    # Check Gelman-Rubin statistic (R-hat) for convergence
    # Requires running multiple chains (default is usually >1)
    # We'll check convergence visually if possible or rely on HDDM warnings.
    # rhat = m.dic_info['deviance'].gelman_rubin # Example way to access Rhat sometimes

    print("\n--- Feasibility Check Conclusion ---")
    print("If HDDM sampling ran without errors and produced reasonable group estimates,")
    print("then using HDDM is technically feasible.")
    print("NOTE: This prototype DOES NOT recover 'k'. It fits 'v' based on an *approximate* k.")
    print("Recovering 'k' within HDDM requires more advanced techniques (e.g., custom likelihoods or model structures).")

except ImportError:
    print("\nERROR: HDDM library not found. Please install it (e.g., pip install hddm or conda install -c conda-forge hddm)")
except Exception as e:
    print(f"\nERROR during HDDM execution: {e}")
    print("HDDM setup or sampling failed.")