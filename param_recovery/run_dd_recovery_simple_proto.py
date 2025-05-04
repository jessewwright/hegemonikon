import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import os

print(f"PyMC version: {pm.__version__}")

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    """Calculate hyperbolic discounted value."""
    k_safe = pt.maximum(k, 1e-7) if isinstance(k, pt.TensorVariable) else max(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

def ddm_logp(v, a, tau, rt, beta=0.5, s=0.1):
    """Simplified DDM log-likelihood for a single trial."""
    # Convert signed RT to positive RT and choice
    abs_rt = pt.abs(rt)
    choice = pt.where(rt > 0, 1, 0)  # 1 for LL, 0 for SS
    
    # Add small constant to avoid division by zero
    abs_rt_safe = abs_rt + 1e-8
    
    # Calculate the log-likelihood using a simplified DDM formula
    z = (a - choice * v * abs_rt_safe) / (s * pt.sqrt(abs_rt_safe))
    exp1 = pt.exp(-z**2 / 2)
    exp2 = pt.exp(-(2 * a * v / (s**2) - z)**2 / 2)
    
    # Use log-sum-exp trick for numerical stability
    log_diff = pt.log(exp1 - exp2 + 1e-8)
    
    # Combine all terms
    logp = pt.log(2 * pt.abs(v) / (s**2)) - (v**2 * abs_rt_safe) / (2 * s**2) \
           - pt.log(a) - pt.log(2 * pt.pi * abs_rt_safe) / 2 + log_diff
    
    # Return -inf for invalid cases (e.g., when exp1 < exp2)
    return pt.where(exp1 <= exp2, -np.inf, logp)

# --- Data Loading ---
data_filename = "synthetic_data_NEW.csv"
true_params_filename = "true_parameters_NEW.csv"

# Check if files exist
if not os.path.exists(data_filename) or not os.path.exists(true_params_filename):
    print("Error: Data files not found. Please ensure synthetic_data_NEW.csv and true_parameters_NEW.csv exist.")
    exit(1)

print(f"Loading data from {data_filename} and {true_params_filename}")
synthetic_data_df = pd.read_csv(data_filename)
true_params_df = pd.read_csv(true_params_filename)

# Use only the first 5 subjects for the prototype
N_SUBJECTS_PROTO = 5
subject_ids_to_keep = true_params_df['subject'].unique()[:N_SUBJECTS_PROTO]
synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'].isin(subject_ids_to_keep)].copy()
true_params_df = true_params_df[true_params_df['subject'].isin(subject_ids_to_keep)].copy()

print(f"Using data from {N_SUBJECTS_PROTO} subjects for prototype fit.")
print(f"Trials after filtering: {len(synthetic_data_df)}")
print("True parameters (Subset):")
print(true_params_df.round(3))

# --- Data Preprocessing ---
min_rt_threshold = 0.05
synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
print(f"Trials after RT filtering: {len(synthetic_data_df)}")

if synthetic_data_df.empty:
    print("Error: No valid trials remain after filtering RTs. Cannot proceed.")
    exit(1)

# Create integer subject indices
subject_ids = synthetic_data_df['subject'].unique()
subject_idx, unique_subjects = pd.factorize(synthetic_data_df['subject'])
n_subjects = len(unique_subjects)

# Encode choices for likelihood: +1 for upper bound (LL), -1 for lower bound (SS)
synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)
# Create signed RTs
synthetic_data_df['signed_rt'] = synthetic_data_df['rt'] * synthetic_data_df['choice_code']

# --- Define Fixed Parameters & Task Options ---
params_fixed_fit = {
    'noise_std_dev': 0.237, # Although we use s=0.1 in likelihood, this was the generation value
    'w_s': 0.392,
    'dt': 0.01,           # Used by generation, not directly by likelihood
    'max_time': 5.0       # Used by generation
}
ss_option_fit = {'amount': 5, 'delay': 0}
ll_amount_fit = 10

# --- Define PyMC Model ---
coords = {
    "subject": unique_subjects,
    "obs_id": synthetic_data_df.index
}

with pm.Model(coords=coords) as hbm_ddm_proto:
    # Hyperpriors
    group_k_mu = pm.HalfNormal("group_k_mu", sigma=0.1)
    group_k_sigma = pm.HalfNormal("group_k_sigma", sigma=0.05)
    group_thresh_mu = pm.HalfNormal("group_thresh_mu", sigma=0.5)
    group_thresh_sigma = pm.HalfNormal("group_thresh_sigma", sigma=0.3)
    group_t_mu = pm.HalfNormal("group_t_mu", sigma=0.1)
    group_t_sigma = pm.HalfNormal("group_t_sigma", sigma=0.05)

    # Subject-Level Parameters (Non-centered)
    k_subj_offset = pm.Normal("k_subj_offset", mu=0, sigma=1, dims="subject")
    k_subj = pm.Deterministic("k_subj", pt.maximum(1e-6, group_k_mu + k_subj_offset * group_k_sigma), dims="subject")

    thresh_subj_offset = pm.Normal("thresh_subj_offset", mu=0, sigma=1, dims="subject")
    thresh_subj_unbounded = group_thresh_mu + thresh_subj_offset * group_thresh_sigma
    thresh_subj = pm.Deterministic("thresh_subj", pt.maximum(0.05, thresh_subj_unbounded), dims="subject")

    t_subj_offset = pm.Normal("t_subj_offset", mu=0, sigma=1, dims="subject")
    t_subj_unbounded = group_t_mu + t_subj_offset * group_t_sigma
    t_subj = pm.Deterministic("t_subj", pt.maximum(1e-6, t_subj_unbounded), dims="subject")

    # Calculate Trial-Level Drift Rate 'v'
    w_s = params_fixed_fit['w_s']
    ss_amount = ss_option_fit['amount']
    ss_delay = ss_option_fit['delay']
    ll_amount_hbm = ll_amount_fit

    k_trial = k_subj[subject_idx]
    v_ss_trial = hyperbolic_discount(ss_amount, ss_delay, k_trial)
    v_ll_trial = hyperbolic_discount(ll_amount_hbm, synthetic_data_df['ll_delay'].values, k_trial)
    drift_v = pm.Deterministic("drift_v", w_s * (v_ll_trial - v_ss_trial), dims="obs_id")

    # Custom likelihood using our simplified DDM function
    likelihood = pm.CustomDist(
        "likelihood",
        drift_v,
        thresh_subj[subject_idx],
        t_subj[subject_idx],
        logp=ddm_logp,
        observed=synthetic_data_df['signed_rt'].values,
        dims="obs_id"
    )

# --- Run MCMC Sampler ---
print("\nStarting PyMC sampling (Prototype)...")
start_sample_time = time.time()

# Feasibility check run parameters
n_draws = 500
n_tune = 1000
n_chains = 2

with hbm_ddm_proto:
    try:
        # Using NUTS sampler (default)
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, cores=1, target_accept=0.90, random_seed=123)
        sampling_successful = True
    except Exception as e:
        print(f"\nERROR during sampling: {e}")
        idata = None
        sampling_successful = False

end_sample_time = time.time()
if sampling_successful:
    print(f"Sampling finished in {end_sample_time - start_sample_time:.2f} seconds.")

    # --- Basic Convergence Check & Summary ---
    print("\n--- MCMC Summary (Group Parameters) ---")
    try:
        summary = az.summary(idata, var_names=['group_k_mu', 'group_k_sigma', 'group_thresh_mu', 'group_thresh_sigma', 'group_t_mu', 'group_t_sigma'])
        print(summary)
        rhat_ok = np.all(summary['r_hat'] < 1.1)
        print(f"\nAll Group R-hats < 1.1: {rhat_ok}")
        if not rhat_ok: print("WARNING: Poor convergence suspected.")
        else: print("Basic convergence looks acceptable.")
    except Exception as e:
        print(f"Could not generate ArviZ summary: {e}")

    # --- Parameter Recovery Check ---
    print("\n--- Parameter Recovery Check (Posterior Means vs True) ---")
    # Extract posterior means for subject parameters
    k_recovered_mean = az.summary(idata, var_names=['k_subj'])['mean'].values
    thresh_recovered_mean = az.summary(idata, var_names=['thresh_subj'])['mean'].values

    recovered_summary = pd.DataFrame({
        'subject': unique_subjects,
        'recovered_k': k_recovered_mean,
        'recovered_threshold': thresh_recovered_mean
    })

    # Merge with true parameters
    comparison_df = pd.merge(true_params_df, recovered_summary, on='subject')
    print("\nTrue vs. Recovered (Posterior Means):")
    print(comparison_df[['subject', 'true_k', 'recovered_k', 'true_threshold', 'recovered_threshold']].round(3))

    # Calculate correlations
    print("\n--- Correlations (True vs. Recovered Means) ---")
    if len(comparison_df) > 1:
        if comparison_df['true_k'].std() > 1e-6 and comparison_df['recovered_k'].std() > 1e-6:
            corr_k = comparison_df['true_k'].corr(comparison_df['recovered_k'])
            print(f"Correlation(True K, Recovered K): {corr_k:.3f}")
        else: print("Correlation(True K, Recovered K): Cannot compute (constant values).")
        if comparison_df['true_threshold'].std() > 1e-6 and comparison_df['recovered_threshold'].std() > 1e-6:
            corr_thresh = comparison_df['true_threshold'].corr(comparison_df['recovered_threshold'])
            print(f"Correlation(True Thresh, Recovered Thresh): {corr_thresh:.3f}")
        else: print("Correlation(True Thresh, Recovered Thresh): Cannot compute (constant values).")
        print(f"\nSUCCESS METRIC CHECK: Aiming for r > 0.7")
        recovery_successful = (corr_k > 0.7 and corr_thresh > 0.7) if ('corr_k' in locals() and 'corr_thresh' in locals()) else False
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
        axes[0].set_xlabel("True k")
        axes[0].set_ylabel("Recovered k (Post. Mean)")
        axes[0].set_title(f"K Recovery (r={corr_k:.3f if 'corr_k' in locals() else 'N/A'})")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim(left=max(0, min_k))
        axes[0].set_ylim(bottom=max(0, min_k))

        axes[1].scatter(comparison_df['true_threshold'], comparison_df['recovered_threshold'], alpha=0.9)
        axes[1].plot([min_t, max_t], [min_t, max_t], 'r--', label='Identity')
        axes[1].set_xlabel("True Threshold")
        axes[1].set_ylabel("Recovered Threshold (Post. Mean)")
        axes[1].set_title(f"Threshold Recovery (r={corr_thresh:.3f if 'corr_thresh' in locals() else 'N/A'})")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim(left=max(0.1, min_t))
        axes[1].set_ylim(bottom=max(0.1, min_t))

    else:
        axes[0].text(0.5, 0.5, 'No data for K plot', ha='center', va='center')
        axes[0].set_title("K Recovery")
        axes[1].text(0.5, 0.5, 'No data for Threshold plot', ha='center', va='center')
        axes[1].set_title("Threshold Recovery")

    plt.suptitle('Parameter Recovery Results (Prototype)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = "hbm_parameter_recovery_proto.png"
    plt.savefig(plot_filename)
    print(f"\nRecovery plot saved to {plot_filename}")

else:
    print("\nSkipping summary and plotting due to sampling error.")

print("\nFull script finished.")
