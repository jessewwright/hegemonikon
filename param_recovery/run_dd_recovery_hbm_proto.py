# Filename: param_recovery/run_dd_recovery_hbm_proto.py
# Purpose: Prototype HBM fit for NES DD model using pm.Wiener
#          Designed to run within an activated virtual environment.

import numpy as np
import pandas as pd
import time
import pymc as pm
import arviz as az
import pytensor.tensor as pt # For PyMC symbolic math
import matplotlib.pyplot as plt # For plotting results
import os # To check for files
import sys # For potentially exiting on error

print("--- HBM Recovery Prototype ---")
print(f"Python Executable: {sys.executable}") # Verify using venv python
try:
    print(f"PyMC version: {pm.__version__}")
    print(f"ArviZ version: {az.__version__}")
except Exception as e:
    print(f"Error importing PyMC/ArviZ: {e}")
    print("Ensure packages are installed in the active virtual environment.")
    sys.exit(1)

# --- Component Definitions (Conceptual Only) ---
# Define classes for context, though not directly used in likelihood
class Comparator:
    def __init__(self, dt=0.01, noise_std_dev=0.1): pass
class AssentGate:
    def __init__(self, base_threshold=1.0): pass

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    k_safe = pt.maximum(k, 1e-7) if isinstance(k, pt.TensorVariable) else max(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

# --- Data Generation or Loading ---
data_filename = "synthetic_dissociative_dd_data.csv"
true_params_filename = "true_dissociative_params.csv"
GENERATED_DATA_USED = False # Flag to track if we generated data

if os.path.exists(data_filename) and os.path.exists(true_params_filename):
    print(f"\nLoading existing data from {data_filename} and {true_params_filename}")
    try:
        synthetic_data_df = pd.read_csv(data_filename)
        true_params_df = pd.read_csv(true_params_filename)
        print(f"Loaded {len(synthetic_data_df)} trials for {true_params_df['subject'].nunique()} subjects.")
        # Use only the first 5 subjects for the prototype
        N_SUBJECTS_PROTO = 5
        if true_params_df['subject'].nunique() >= N_SUBJECTS_PROTO:
            print(f"Using data from first {N_SUBJECTS_PROTO} subjects for prototype fit.")
            subject_ids_to_keep = sorted(true_params_df['subject'].unique())[:N_SUBJECTS_PROTO]
            synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'].isin(subject_ids_to_keep)].copy()
            # Ensure true_params_df only contains the selected subjects
            true_params_df = true_params_df[true_params_df['subject'].isin(subject_ids_to_keep)].copy()
        else:
            print(f"Warning: Fewer than {N_SUBJECTS_PROTO} subjects found in loaded data.")
            N_SUBJECTS_PROTO = true_params_df['subject'].nunique()

        print("True parameters (Loaded Subset):")
        print(true_params_df[['subject', 'true_k', 'true_base_threshold']].round(3)) # Display base thresh from dissociative data

    except Exception as e:
        print(f"Error loading data files: {e}. Will attempt to generate demo data.")
        synthetic_data_df = pd.DataFrame() # Ensure df is empty
else:
    print(f"Data files '{data_filename}' or '{true_params_filename}' not found. Generating PROTOTYPE data...")
    GENERATED_DATA_USED = True
    N_SUBJECTS_PROTO = 5
    N_REPS_PER_DELAY_PROTO = 20
    ll_delays_proto = [1, 5, 10, 20, 50]
    params_fixed_gen = { # Use consistent fixed params
        'noise_std_dev': 0.237, 'w_s': 0.392, 'w_n': 0.0, 'w_u': 0.0,
        'dt': 0.01, 'max_time': 5.0
    }
    true_k_mean_gen = 0.04; true_k_sd_gen = 0.03
    true_thresh_mean_gen = 0.5; true_thresh_sd_gen = 0.15
    gamma_k_scale_gen = (true_k_sd_gen**2) / true_k_mean_gen if true_k_mean_gen > 0 else 1.0
    gamma_k_shape_gen = true_k_mean_gen / gamma_k_scale_gen if gamma_k_scale_gen > 0 else 1.0
    ss_option_gen = {'amount': 5, 'delay': 0}
    ll_amount_gen = 10
    proto_synthetic_data = []
    proto_true_params = []
    np.random.seed(42)

    # Simplified Generation (No need for full sim code here)
    for subj_id in range(N_SUBJECTS_PROTO):
        subj_k = np.random.gamma(shape=gamma_k_shape_gen, scale=gamma_k_scale_gen)
        subj_thresh = np.random.normal(loc=true_thresh_mean_gen, scale=true_thresh_sd_gen)
        subj_thresh = max(0.1, subj_thresh)
        proto_true_params.append({'subject': subj_id, 'true_k': subj_k, 'true_threshold': subj_thresh}) # Store base thresh
        for delay in ll_delays_proto:
            ll_option_gen = {'amount': ll_amount_gen, 'delay': delay}
            v_ss = hyperbolic_discount(ss_option_gen['amount'], ss_option_gen['delay'], subj_k)
            v_ll = hyperbolic_discount(ll_option_gen['amount'], ll_option_gen['delay'], subj_k)
            prob_ll = 1 / (1 + np.exp(-(v_ll - v_ss)))
            for rep in range(N_REPS_PER_DELAY_PROTO):
                 choice = 'Choose_LL' if np.random.rand() < prob_ll else 'Choose_SS'
                 rt_mean = 0.2 + np.abs(v_ll - v_ss) * 0.05
                 rt = np.random.lognormal(mean=np.log(rt_mean), sigma=0.3)
                 rt = max(0.051, rt)
                 proto_synthetic_data.append({
                     'subject': subj_id, 'll_delay': delay, 'choice': choice, 'rt': rt,
                     'true_k': subj_k, 'true_threshold': subj_thresh
                 })
    synthetic_data_df = pd.DataFrame(proto_synthetic_data)
    true_params_df = pd.DataFrame(proto_true_params)
    print(f"Prototype data generated: {len(synthetic_data_df)} valid trials.")
    print("True parameters (Generated Prototype):")
    print(true_params_df.round(3))


# --- Data Preprocessing for PyMC ---
min_rt_threshold = 0.05
print(f"\nInitial trials loaded/generated: {len(synthetic_data_df)}")
synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
print(f"Trials after filtering RT > {min_rt_threshold}s: {len(synthetic_data_df)}")

if synthetic_data_df.empty:
    print("Error: No valid trials remain after filtering RTs. Cannot proceed.")
    sys.exit(1)

# Remap subject IDs to 0..N-1 indices for PyMC coords
synthetic_data_df['subj_idx'], unique_subjects_mapping = pd.factorize(synthetic_data_df['subject'])
n_subjects = len(unique_subjects_mapping)
print(f"Fitting {n_subjects} subjects.")

synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)
synthetic_data_df['signed_rt'] = synthetic_data_df['rt'] * synthetic_data_df['choice_code']

# Prepare data dictionary for PyMC
coords = {
    "subject_coord": np.arange(n_subjects), # Use simple 0..N-1 index for coords
    "obs_id": synthetic_data_df.index
}
# Ensure data passed to PyMC uses the 0..N-1 index
subject_idx_data = synthetic_data_df['subj_idx'].values
ll_delay_data = synthetic_data_df['ll_delay'].values
signed_rt_data = synthetic_data_df['signed_rt'].values

# --- Define PyMC Model ---
with pm.Model(coords=coords) as hbm_nes_dd:
    # --- Hyperpriors ---
    group_k_mu = pm.HalfNormal("group_k_mu", sigma=0.1)
    group_k_sigma = pm.HalfNormal("group_k_sigma", sigma=0.05)
    group_thresh_mu = pm.HalfNormal("group_thresh_mu", sigma=0.5)
    group_thresh_sigma = pm.HalfNormal("group_thresh_sigma", sigma=0.3)
    group_t_mu = pm.HalfNormal("group_t_mu", sigma=0.1)
    group_t_sigma = pm.HalfNormal("group_t_sigma", sigma=0.05)

    # --- Subject-Level Parameters (Non-centered) ---
    k_subj_offset = pm.Normal("k_subj_offset", mu=0, sigma=1, dims="subject_coord")
    k_subj_unbounded = group_k_mu + k_subj_offset * group_k_sigma
    k_subj = pm.Deterministic("k_subj", pt.maximum(1e-6, k_subj_unbounded), dims="subject_coord")

    thresh_subj_offset = pm.Normal("thresh_subj_offset", mu=0, sigma=1, dims="subject_coord")
    thresh_subj_unbounded = group_thresh_mu + thresh_subj_offset * group_thresh_sigma
    thresh_subj = pm.Deterministic("thresh_subj", pt.maximum(0.05, thresh_subj_unbounded), dims="subject_coord")

    t_subj_offset = pm.Normal("t_subj_offset", mu=0, sigma=1, dims="subject_coord")
    t_subj_unbounded = group_t_mu + t_subj_offset * group_t_sigma
    t_subj = pm.Deterministic("t_subj", pt.maximum(1e-6, t_subj_unbounded), dims="subject_coord")

    # --- Calculate Trial-Level Drift Rate 'v' ---
    # Fixed parameters assumed known from generation
    # IMPORTANT: Use the same values as in the generation script
    w_s_fit = 0.392
    ss_amount_fit = 5.0
    ss_delay_fit = 0.0
    ll_amount_fit = 10.0

    # Get parameters corresponding to each observation's subject index
    k_trial = k_subj[subject_idx_data]
    a_trial = thresh_subj[subject_idx_data]
    tau_trial = t_subj[subject_idx_data]

    # Calculate discounted values symbolically
    v_ss_trial = hyperbolic_discount(ss_amount_fit, ss_delay_fit, k_trial)
    v_ll_trial = hyperbolic_discount(ll_amount_fit, ll_delay_data, k_trial)
    drift_v = pm.Deterministic("drift_v", w_s_fit * (v_ll_trial - v_ss_trial), dims="obs_id")

    # --- Likelihood (Using Built-in pm.Wiener) ---
    # Note: Parameter names for pm.Wiener are v, a, tau, beta, s
    likelihood = pm.Wiener(
        "likelihood",
        v=drift_v,       # Drift rate for this trial
        a=a_trial,       # Threshold for this subject's trial
        tau=tau_trial,   # Non-decision time for this subject's trial
        beta=0.5,        # Start bias (0.5 = unbiased)
        s=0.1,           # Noise scaling (fixed standard value)
        observed=signed_rt_data,
        dims="obs_id"
    )

# --- Run MCMC Sampler ---
print("\nStarting PyMC sampling (Prototype)...")
start_sample_time = time.time()
n_draws = 500; n_tune = 1000; n_chains = 2 # Short run for feasibility

with hbm_nes_dd:
    try:
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, cores=1, target_accept=0.90, random_seed=1234)
        sampling_successful = True
    except AttributeError as ae:
        print(f"\nATTRIBUTE ERROR during sampling: {ae}")
        print("This likely means 'pm.Wiener' is still not accessible.")
        print("Ensure 'pymc-experimental' is installed correctly in the Python environment using:")
        print("pip install pymc-experimental")
        idata = None
        sampling_successful = False
    except Exception as e:
        print(f"\nUNEXPECTED ERROR during sampling: {e}")
        idata = None
        sampling_successful = False

end_sample_time = time.time()

# --- Analysis ---
if sampling_successful:
    print(f"Sampling finished in {end_sample_time - start_sample_time:.2f} seconds.")
    print("\n--- MCMC Summary (Group Parameters) ---")
    try:
        summary = az.summary(idata, var_names=['group_k_mu', 'group_k_sigma', 'group_thresh_mu', 'group_thresh_sigma', 'group_t_mu', 'group_t_sigma'], round_to=3)
        print(summary)
        rhat_ok = np.all(summary['r_hat'] < 1.1) if not summary.empty else False
        print(f"\nAll Group R-hats < 1.1: {rhat_ok}")
        if not rhat_ok: print("WARNING: Poor convergence suspected.")
        else: print("Basic convergence looks acceptable.")
    except Exception as e:
        print(f"Could not generate ArviZ summary: {e}")

    # --- Parameter Recovery Check ---
    print("\n--- Parameter Recovery Check (Posterior Means vs True) ---")
    try:
        # Extract posterior means for subject parameters
        k_recovered_mean = az.summary(idata, var_names=['k_subj'])['mean'].values
        thresh_recovered_mean = az.summary(idata, var_names=['thresh_subj'])['mean'].values

        recovered_summary = pd.DataFrame({
            'subject': unique_subjects_mapping, # Use the mapping to get original subject IDs
            'recovered_k': k_recovered_mean,
            'recovered_threshold': thresh_recovered_mean
        })

        # Merge with true parameters (make sure subject IDs match)
        comparison_df = pd.merge(true_params_df, recovered_summary, on='subject')
        print("True vs. Recovered (Posterior Means):")
        print(comparison_df[['subject', 'true_k', 'recovered_k', 'true_threshold', 'recovered_threshold']].round(3))

        # Calculate correlations
        print("\n--- Correlations (True vs. Recovered Means) ---")
        if len(comparison_df) > 1:
            corr_k, corr_thresh = np.nan, np.nan # Initialize
            if comparison_df['true_k'].std() > 1e-6 and comparison_df['recovered_k'].std() > 1e-6:
                 corr_k = comparison_df['true_k'].corr(comparison_df['recovered_k'])
                 print(f"Correlation(True K, Recovered K): {corr_k:.3f}")
            else: print("Correlation(True K, Recovered K): Cannot compute (constant values).")

            # Note: True threshold used here is BASE threshold. Recovered is 'a'. They should correlate but might differ in scale.
            if comparison_df['true_threshold'].std() > 1e-6 and comparison_df['recovered_threshold'].std() > 1e-6:
                 corr_thresh = comparison_df['true_threshold'].corr(comparison_df['recovered_threshold'])
                 print(f"Correlation(True Thresh, Recovered Thresh): {corr_thresh:.3f}")
            else: print("Correlation(True Thresh, Recovered Thresh): Cannot compute (constant values).")

            print(f"\nSUCCESS METRIC CHECK: Aiming for r > 0.7")
            recovery_successful = (corr_k > 0.7 and corr_thresh > 0.7) if (not np.isnan(corr_k) and not np.isnan(corr_thresh)) else False
            print(f"Recovery Successful (r > 0.7 for both): {recovery_successful}")

        else: print("Not enough data points to calculate correlations.")

        # --- Plotting ---
        print("\n--- Generating Recovery Plots ---")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        if len(comparison_df) > 0:
            min_k = min(comparison_df['true_k'].min(), comparison_df['recovered_k'].min()) * 0.9
            max_k = max(comparison_df['true_k'].max(), comparison_df['recovered_k'].max()) * 1.1
            min_t = min(comparison_df['true_threshold'].min(), comparison_df['recovered_threshold'].min()) * 0.9
            max_t = max(comparison_df['true_threshold'].max(), comparison_df['recovered_threshold'].max()) * 1.1

            axes[0].scatter(comparison_df['true_k'], comparison_df['recovered_k'], alpha=0.9)
            axes[0].plot([min_k, max_k], [min_k, max_k], 'r--', label='Identity')
            axes[0].set_xlabel("True k"); axes[0].set_ylabel("Recovered k (Post. Mean)")
            axes[0].set_title(f"K Recovery (r={corr_k:.3f if 'corr_k' in locals() and not np.isnan(corr_k) else 'N/A'})")
            axes[0].legend(); axes[0].grid(True); axes[0].set_xlim(left=max(0, min_k)); axes[0].set_ylim(bottom=max(0, min_k))

            axes[1].scatter(comparison_df['true_threshold'], comparison_df['recovered_threshold'], alpha=0.9)
            axes[1].plot([min_t, max_t], [min_t, max_t], 'r--', label='Identity')
            axes[1].set_xlabel("True Threshold"); axes[1].set_ylabel("Recovered Threshold (Post. Mean)")
            axes[1].set_title(f"Threshold Recovery (r={corr_thresh:.3f if 'corr_thresh' in locals() and not np.isnan(corr_thresh) else 'N/A'})")
            axes[1].legend(); axes[1].grid(True); axes[1].set_xlim(left=max(0.1, min_t)); axes[1].set_ylim(bottom=max(0.1, min_t))
        else:
            axes[0].text(0.5, 0.5, 'No data for K plot', ha='center', va='center'); axes[0].set_title("K Recovery")
            axes[1].text(0.5, 0.5, 'No data for Threshold plot', ha='center', va='center'); axes[1].set_title("Threshold Recovery")

        plt.suptitle('Parameter Recovery Results (HBM Prototype)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = "hbm_parameter_recovery_prototype.png"
        try:
            plt.savefig(plot_filename); print(f"\nRecovery plot saved to {plot_filename}")
        except Exception as e: print(f"\nError saving plot: {e}")
        # plt.show() # Use plt.show() if running interactively

    except Exception as e:
        print(f"Error during analysis or plotting: {e}")

else:
    print("\nSkipping analysis and plotting due to sampling error.")

print("\nFull script finished.")