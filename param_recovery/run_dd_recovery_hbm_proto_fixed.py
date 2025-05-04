# Filename: param_recovery/run_dd_recovery_hbm_proto_fixed.py
# Purpose: Prototype HBM fit for NES DD model using pm.Wiener

import numpy as np
import pandas as pd
import time
import pymc as pm
import pymc_experimental as pmx
import arviz as az
import pytensor.tensor as pt # For PyMC symbolic math
import matplotlib.pyplot as plt # For plotting results
import os # To check for files

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# --- Component Definitions (Conceptual Only) ---
# These classes define the cognitive process we assume generated the data.
# They are NOT directly called by the PyMC likelihood function (pm.Wiener).
class Comparator:
    """Conceptual NES Comparator Module for defining generating process"""
    def __init__(self, dt=0.01, noise_std_dev=0.1):
        if dt <= 0: raise ValueError("dt must be positive.")
        if noise_std_dev < 0: raise ValueError("noise cannot be negative.")
        self.dt = dt
        self.noise_std_dev = noise_std_dev
        self.sqrt_dt = np.sqrt(dt)
        self.evidence = {}
        self.time_elapsed = 0.0
    def reset(self): self.evidence = {}; self.time_elapsed = 0.0
    def initialize_actions(self, actions): self.reset(); self.evidence = {action: 0.0 for action in actions}
    def calculate_drift_rate(self, action_attributes, params):
        S = action_attributes.get('S', 0.0); N = action_attributes.get('N', 0.0); U = action_attributes.get('U', 0.0)
        w_s = params.get('w_s', 1.0); w_n = params.get('w_n', 0.0); w_u = params.get('w_u', 0.0)
        return (w_s * S) + (w_n * N) + (w_u * U)
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
    """Conceptual NES Assent Gate for defining generating process"""
    def __init__(self, base_threshold=1.0):
        if base_threshold <= 0: raise ValueError("Base threshold must be positive.")
        self.base_threshold = base_threshold # Corrected attribute name
    def check(self, evidence_dict, current_threshold):
        if current_threshold <= 0: current_threshold = 0.01
        winning_action = None; max_evidence = -float('inf')
        for action, evidence in evidence_dict.items():
            if evidence >= current_threshold:
                 if evidence > max_evidence: max_evidence = evidence; winning_action = action
        return winning_action

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    # Use pt.maximum for symbolic calculation within PyMC model
    # Use max for standard python calculation during data generation
    k_safe = pt.maximum(k, 1e-7) if isinstance(k, pt.TensorVariable) else max(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

# --- Single Trial Simulation Function (for Data Generation ONLY) ---
def run_single_dd_trial_gen(comparator, assent_gate, params, ss_option, ll_option):
    """ Runs a single DD trial for generating synthetic data. """
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
        # Pass full params dict containing w_s etc.
        current_evidence = comparator.step(action_attributes, params)
        decision = assent_gate.check(current_evidence, current_threshold)
        if decision is not None: break
        accumulated_time += params['dt']
    rt = accumulated_time if decision is not None else params['max_time']
    choice = decision if decision is not None else 'timeout'
    # Ensure rt is strictly positive before returning
    rt = max(params.get('dt', 0.01), rt) # Ensure RT is at least dt
    return {'choice': choice, 'rt': rt}

# --- Data Generation or Loading ---
# Define filenames
data_filename = "synthetic_data_NEW.csv"
true_params_filename = "true_parameters_NEW.csv"

# Check if files exist, otherwise generate demo data
if os.path.exists(data_filename) and os.path.exists(true_params_filename):
    print(f"Loading existing data from {data_filename} and {true_params_filename}")
    synthetic_data_df = pd.read_csv(data_filename)
    true_params_df = pd.read_csv(true_params_filename)
    print(f"Loaded {len(synthetic_data_df)} trials for {true_params_df['subject'].nunique()} subjects.")
    # Determine number of subjects from loaded data
    N_SUBJECTS_PROTO = true_params_df['subject'].nunique()
    # Use only the first 5 subjects for the prototype if more were loaded
    if N_SUBJECTS_PROTO > 5:
        print("Using data from first 5 subjects for prototype fit.")
        subject_ids_to_keep = true_params_df['subject'].unique()[:5]
        synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'].isin(subject_ids_to_keep)].copy()
        true_params_df = true_params_df[true_params_df['subject'].isin(subject_ids_to_keep)].copy()
        N_SUBJECTS_PROTO = 5
    print("True parameters (Loaded Subset):")
    print(true_params_df.round(3))

else:
    print("Data files not found. Generating PROTOTYPE synthetic data...")
    N_SUBJECTS_PROTO = 5
    N_REPS_PER_DELAY_PROTO = 20
    ll_delays_proto = [1, 5, 10, 20, 50]
    params_fixed_gen = {
        'noise_std_dev': 0.237, 'w_s': 0.392, 'w_n': 0.0, 'w_u': 0.0,
        'dt': 0.01, 'max_time': 5.0
    }
    true_k_mean_gen = 0.04; true_k_sd_gen = 0.03
    true_thresh_mean_gen = 0.5; true_thresh_sd_gen = 0.15
    gamma_k_scale_gen = (true_k_sd_gen**2) / true_k_mean_gen
    gamma_k_shape_gen = true_k_mean_gen / gamma_k_scale_gen
    ss_option_gen = {'amount': 5, 'delay': 0}
    ll_amount_gen = 10
    proto_synthetic_data = []
    proto_true_params = []
    np.random.seed(42)

    for subj_id in range(N_SUBJECTS_PROTO):
        subj_k = np.random.gamma(shape=gamma_k_shape_gen, scale=gamma_k_scale_gen)
        subj_thresh = np.random.normal(loc=true_thresh_mean_gen, scale=true_thresh_sd_gen)
        subj_thresh = max(0.1, subj_thresh)
        proto_true_params.append({'subject': subj_id, 'true_k': subj_k, 'true_threshold': subj_thresh})
        subj_params = params_fixed_gen.copy()
        subj_params['k_discount'] = subj_k
        subj_params['base_threshold'] = subj_thresh
        comparator_gen = Comparator(dt=subj_params['dt'], noise_std_dev=subj_params['noise_std_dev'])
        assent_gate_gen = AssentGate(base_threshold=subj_params['base_threshold'])
        for delay in ll_delays_proto:
            ll_option_gen = {'amount': ll_amount_gen, 'delay': delay}
            for rep in range(N_REPS_PER_DELAY_PROTO):
                result = run_single_dd_trial_gen(comparator_gen, assent_gate_gen, subj_params, ss_option_gen, ll_option_gen)
                if result['choice'] != 'timeout':
                     result['subject'] = subj_id
                     result['ll_delay'] = delay
                     result['true_k'] = subj_k
                     result['true_threshold'] = subj_thresh
                     proto_synthetic_data.append(result)

    synthetic_data_df = pd.DataFrame(proto_synthetic_data)
    true_params_df = pd.DataFrame(proto_true_params)
    print(f"Prototype data generated: {len(synthetic_data_df)} valid trials.")
    print("True parameters (Generated Prototype):")
    print(true_params_df.round(3))
    # Optional: Save the generated demo data
    # synthetic_data_df.to_csv("synthetic_data_DEMO.csv", index=False)
    # true_params_df.to_csv("true_parameters_DEMO.csv", index=False)

# --- Define Fixed Parameters & Task Options for Model ---
# These are assumed known based on the data generation process
params_fixed_fit = {
    'noise_std_dev': 0.237, # Although Wiener uses 's', this was the generation value
    'w_s': 0.392,
    'dt': 0.01,           # Used by generation, not directly by Wiener likelihood
    'max_time': 5.0       # Used by generation
}
ss_option_fit = {'amount': 5, 'delay': 0}
ll_amount_fit = 10

# --- Data Preprocessing for PyMC ---
min_rt_threshold = 0.05 # Minimum plausible RT
print(f"\nInitial trials: {len(synthetic_data_df)}")
synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
print(f"Trials after filtering RT > {min_rt_threshold}s: {len(synthetic_data_df)}")

if synthetic_data_df.empty:
    print("Error: No valid trials remain after filtering RTs. Cannot proceed.")
    # sys.exit(1)

subject_ids = synthetic_data_df['subject'].unique()
subject_idx, unique_subjects = pd.factorize(synthetic_data_df['subject'])
n_subjects = len(unique_subjects)
synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)
synthetic_data_df['signed_rt'] = synthetic_data_df['rt'] * synthetic_data_df['choice_code']

# --- Define PyMC Model (Using pm.Wiener) ---
coords = {
    "subject": unique_subjects, # Use actual subject IDs if available, else indices
    "obs_id": synthetic_data_df.index # Use dataframe index for observations
}

with pm.Model(coords=coords) as hbm_nes_dd:
    # --- Hyperpriors for Group Parameters ---
    group_k_mu = pm.HalfNormal("group_k_mu", sigma=0.1)
    group_k_sigma = pm.HalfNormal("group_k_sigma", sigma=0.05)
    group_thresh_mu = pm.HalfNormal("group_thresh_mu", sigma=0.5)
    group_thresh_sigma = pm.HalfNormal("group_thresh_sigma", sigma=0.3)
    group_t_mu = pm.HalfNormal("group_t_mu", sigma=0.1)
    group_t_sigma = pm.HalfNormal("group_t_sigma", sigma=0.05)

    # --- Subject-Level Parameters (Non-centered) ---
    # K (discount rate) - Using Gamma ensures positivity
    k_alpha = pt.maximum((group_k_mu / group_k_sigma)**2, 1e-6)
    k_beta = pt.maximum(group_k_mu / group_k_sigma**2, 1e-6)
    k_subj = pm.Gamma("k_subj", alpha=k_alpha, beta=k_beta, dims="subject")

    # Threshold ('a' in Wiener) - Use HalfNormal centered on group mean
    thresh_subj_offset = pm.Normal("thresh_subj_offset", mu=0, sigma=1, dims="subject")
    thresh_subj_unbounded = group_thresh_mu + thresh_subj_offset * group_thresh_sigma
    thresh_subj = pm.Deterministic("thresh_subj", pt.maximum(0.05, thresh_subj_unbounded), dims="subject") # Ensure > 0.05

    # Non-decision time ('tau' in Wiener) - Use HalfNormal centered on group mean
    t_subj_offset = pm.Normal("t_subj_offset", mu=0, sigma=1, dims="subject")
    t_subj_unbounded = group_t_mu + t_subj_offset * group_t_sigma
    t_subj = pm.Deterministic("t_subj", pt.maximum(1e-6, t_subj_unbounded), dims="subject") # Ensure > 0

    # --- Calculate Trial-Level Drift Rate 'v' ---
    # Fixed parameters for model
    w_s = params_fixed_fit['w_s']
    ss_amount = ss_option_fit['amount']
    ss_delay = ss_option_fit['delay']
    ll_amount_hbm = ll_amount_fit

    # Get parameters corresponding to each observation's subject index
    k_trial = k_subj[subject_idx]
    a_trial = thresh_subj[subject_idx]
    tau_trial = t_subj[subject_idx]

    # Calculate discounted values symbolically
    v_ss_trial = hyperbolic_discount(ss_amount, ss_delay, k_trial)
    v_ll_trial = hyperbolic_discount(ll_amount_hbm, synthetic_data_df['ll_delay'].values, k_trial)
    drift_v = pm.Deterministic("drift_v", w_s * (v_ll_trial - v_ss_trial), dims="obs_id")

    # --- Likelihood (Using Built-in pm.Wiener) ---
    likelihood = pmx.Wiener(
        "likelihood",
        v=drift_v,                      # Trial-specific drift rate
        a=a_trial,                      # Subject-specific threshold
        tau=tau_trial,                  # Subject-specific non-decision time
        beta=0.5,                       # Start bias (0.5 = unbiased)
        s=0.1,                          # Noise scaling (fixed standard value)
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

with hbm_nes_dd:
    try:
        # Using NUTS sampler (default)
        idata = pm.sample(draws=n_draws, tune=n_tune, chains=n_chains, cores=1, target_accept=0.90, random_seed=123)
        sampling_successful = True
    except Exception as e:
        print(f"\nERROR during sampling: {e}")
        print("Check model specification and ensure 'pymc-experimental' might be needed if Wiener isn't standard.")
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

    # --- Parameter Recovery Check (using posterior means) ---
    print("\n--- Parameter Recovery Check (Posterior Means vs True) ---")
    # Extract posterior means for subject parameters
    k_recovered_mean = az.summary(idata, var_names=['k_subj'])['mean'].values
    thresh_recovered_mean = az.summary(idata, var_names=['thresh_subj'])['mean'].values

    recovered_summary = pd.DataFrame({
        'subject': unique_subjects, # Match recovered to original subject IDs
        'recovered_k': k_recovered_mean,
        'recovered_threshold': thresh_recovered_mean
    })

    # Merge with true parameters
    comparison_df = pd.merge(true_params_df, recovered_summary, on='subject')
    print("True vs. Recovered (Posterior Means):")
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
        axes[0].set_xlabel("True k"); axes[0].set_ylabel("Recovered k (Post. Mean)")
        axes[0].set_title(f"K Recovery (r={corr_k:.3f if 'corr_k' in locals() else 'N/A'})")
        axes[0].legend(); axes[0].grid(True); axes[0].set_xlim(left=max(0, min_k)); axes[0].set_ylim(bottom=max(0, min_k))

        axes[1].scatter(comparison_df['true_threshold'], comparison_df['recovered_threshold'], alpha=0.9)
        axes[1].plot([min_t, max_t], [min_t, max_t], 'r--', label='Identity')
        axes[1].set_xlabel("True Threshold"); axes[1].set_ylabel("Recovered Threshold (Post. Mean)")
        axes[1].set_title(f"Threshold Recovery (r={corr_thresh:.3f if 'corr_thresh' in locals() else 'N/A'})")
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
    # plt.show()

else:
    print("\nSkipping summary and plotting due to sampling error.")

print("\nFull script finished.")