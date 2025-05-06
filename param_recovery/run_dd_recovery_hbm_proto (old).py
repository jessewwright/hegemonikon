import numpy as np
import pandas as pd
import time
import pymc as pm
import arviz as az
from pymc.distributions import DensityDist
import pytensor.tensor as pt # For PyMC symbolic math

print(f"PyMC version: {pm.__version__}")

# --- Component Definitions (Simplified for use in HBM context) ---
# NOTE: These are NOT directly used in PyMC likelihood, but define the process
#       that generated the data and whose parameters we are modeling.
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

    def step(self, action_attributes, params):
        """Update evidence for each action based on current attributes and noise."""
        for action, attributes in action_attributes.items():
            # Calculate drift for this action
            drift = 0.0
            for weight, value in attributes.items():
                if weight == 'U': continue
                drift += params.get(weight, 0.0) * value
            # Add drift and noise
            self.evidence[action] += drift * self.dt + np.random.normal(0, self.noise_std_dev * self.sqrt_dt)
        return self.evidence

class AssentGate:
    """ Conceptual representation for parameter definition """
    def __init__(self, base_threshold=1.0): self.base_threshold = base_threshold

    def check(self, evidence, current_threshold):
        """Check if any action has reached the threshold."""
        for action, ev in evidence.items():
            if abs(ev) >= current_threshold:
                return action
        return None

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    # Ensure k is positive and non-zero for calculations
    k_safe = pt.maximum(k, 1e-7) if isinstance(k, pt.TensorVariable) else max(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

# --- Regenerate Small Synthetic Dataset (N=5 subjects) ---
# (Because I cannot load your CSVs, replace this section locally if you run it)
N_SUBJECTS_PROTO = 5
N_REPS_PER_DELAY_PROTO = 20 # Keep reasonable number of trials
ll_delays_proto = [1, 5, 10, 20, 50] # Fewer delays ok for prototype

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
np.random.seed(42) # Seed for reproducibility

print(f"Generating PROTOTYPE synthetic data for {N_SUBJECTS_PROTO} subjects...")
for subj_id in range(N_SUBJECTS_PROTO):
    subj_k = np.random.gamma(shape=gamma_k_shape_gen, scale=gamma_k_scale_gen)
    subj_thresh = np.random.normal(loc=true_thresh_mean_gen, scale=true_thresh_sd_gen)
    subj_thresh = max(0.1, subj_thresh) # Ensure threshold > 0.1
    proto_true_params.append({'subject': subj_id, 'true_k': subj_k, 'true_threshold': subj_thresh})
    subj_params = params_fixed_gen.copy()
    subj_params['k_discount'] = subj_k
    subj_params['base_threshold'] = subj_thresh
    # Need actual simulation components for generation
    comparator_gen = Comparator(dt=subj_params['dt'], noise_std_dev=subj_params['noise_std_dev'])
    assent_gate_gen = AssentGate(base_threshold=subj_params['base_threshold'])
    #--- Temp copy of run_single_dd_trial for generation ---
    def run_single_dd_trial_gen(comparator, assent_gate, params, ss_option, ll_option):
        k = params['k_discount']; v_ss = hyperbolic_discount(ss_option['amount'], ss_option['delay'], k); v_ll = hyperbolic_discount(ll_option['amount'], ll_option['delay'], k)
        actions = ['Choose_LL', 'Choose_SS']; action_attributes = {
            'Choose_LL': {'S': v_ll, 'N': 0, 'U': 0},
            'Choose_SS': {'S': v_ss, 'N': 0, 'U': 0}
        }
        comparator.initialize_actions(actions); accumulated_time = 0.0; decision = None; current_threshold = params['base_threshold']
        while accumulated_time < params['max_time']:
            current_evidence = comparator.step(action_attributes, params); decision = assent_gate.check(current_evidence, current_threshold)
            if decision is not None: break
            accumulated_time += params['dt']
        rt = accumulated_time if decision is not None else params['max_time']; choice = decision if decision is not None else 'timeout'
        return {'choice': choice, 'rt': rt}
    #--- End temp copy ---
    for delay in ll_delays_proto:
        ll_option_gen = {'amount': ll_amount_gen, 'delay': delay}
        for rep in range(N_REPS_PER_DELAY_PROTO):
            result = run_single_dd_trial_gen(comparator_gen, assent_gate_gen, subj_params, ss_option_gen, ll_option_gen)
            if result['choice'] != 'timeout': # Only keep valid trials
                 result['subject'] = subj_id
                 result['ll_delay'] = delay
                 result['true_k'] = subj_k
                 result['true_threshold'] = subj_thresh
                 proto_synthetic_data.append(result)

synthetic_data_df = pd.DataFrame(proto_synthetic_data)
true_params_df = pd.DataFrame(proto_true_params)
print(f"Prototype data generated: {len(synthetic_data_df)} valid trials.")
print("True parameters (Prototype):")
print(true_params_df.round(3))

# --- Data Preprocessing for PyMC ---
# Filter out excessively fast RTs which are problematic for DDM likelihoods
min_rt_threshold = 0.05
synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
print(f"Trials after filtering RT > {min_rt_threshold}s: {len(synthetic_data_df)}")

# Create integer subject indices
subject_ids = synthetic_data_df['subject'].unique()
subject_idx, unique_subjects = pd.factorize(synthetic_data_df['subject'])
n_subjects = len(unique_subjects)

# Encode choices for Wiener likelihood: +1 for upper bound (LL), -1 for lower bound (SS)
synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)

# Create signed RTs
synthetic_data_df['signed_rt'] = synthetic_data_df['rt'] * synthetic_data_df['choice_code']

# --- Define PyMC Model ---
coords = {
    "subject": np.arange(n_subjects),
    "obs_id": np.arange(len(synthetic_data_df))
}

with pm.Model(coords=coords) as hbm_nes_dd:

    # --- Hyperpriors for Group Parameters ---
    # Priors centered roughly around expected values, weakly informative
    group_k_mu = pm.HalfNormal("group_k_mu", sigma=0.1) # Expected k around 0.04
    group_k_sigma = pm.HalfNormal("group_k_sigma", sigma=0.05)
    group_thresh_mu = pm.HalfNormal("group_thresh_mu", sigma=0.5) # Expected thresh around 0.5
    group_thresh_sigma = pm.HalfNormal("group_thresh_sigma", sigma=0.3)
    group_t_mu = pm.Gamma("group_t_mu", mu=0.15, sigma=0.05) # Non-decision time ~150ms?
    group_t_sigma = pm.HalfNormal("group_t_sigma", sigma=0.05)

    # --- Subject-Level Parameters ---
    # Parameterize Gamma by mu/sigma for easier interpretation
    k_alpha = (group_k_mu / group_k_sigma)**2
    k_beta = group_k_mu / group_k_sigma**2
    k_subj = pm.Gamma("k_subj", alpha=k_alpha, beta=k_beta, dims="subject")

    # Threshold ('a' in Wiener) - must be positive
    thresh_subj = pm.TruncatedNormal("thresh_subj", mu=group_thresh_mu, sigma=group_thresh_sigma, lower=0.05, dims="subject")

    # Non-decision time ('tau' in Wiener) - must be positive and < RT
    t_alpha = (group_t_mu / group_t_sigma)**2
    t_beta = group_t_mu / group_t_sigma**2
    t_subj = pm.Gamma("t_subj", alpha=t_alpha, beta=t_beta, dims="subject")
    # Ensure tau is always less than observed RT symbolically? Complex. Rely on filtering data for now.

    # --- Calculate Trial-Level Drift Rate 'v' ---
    # Fixed parameters assumed known (can be estimated too later)
    w_s = params_fixed_gen['w_s'] # 0.392
    ss_amount = ss_option_gen['amount']
    ss_delay = ss_option_gen['delay']
    ll_amount_hbm = ll_amount_gen # Renamed to avoid clash

    # Get subject-specific k for each trial
    k_trial = k_subj[subject_idx]

    # Calculate discounted values symbolically
    v_ss_trial = hyperbolic_discount(ss_amount, ss_delay, k_trial)
    v_ll_trial = hyperbolic_discount(ll_amount_hbm, synthetic_data_df['ll_delay'].values, k_trial)

    # Calculate drift rate v = w_s * (V_LL - V_SS)
    drift_v = pm.Deterministic("drift_v", w_s * (v_ll_trial - v_ss_trial))

    # --- Likelihood ---
    # Wiener likelihood for response time and choice (choice encoded in sign of RT)
    # NOTE: Wiener likelihood has parameters: v (drift), a (threshold), tau (non-decision), beta (bias), s (noise scale)
    # Fixing beta (start bias) to 0.5 (unbiased)
    # Fixing s (noise scaling) to 0.1 (standard practice, relates to noise_std_dev in complex way)
    # Calculate expected RT based on drift and threshold
    # Custom DDM likelihood function
    def ddm_logp(v, a, tau, rt):
        """DDM log-likelihood for a single trial."""
        # Convert signed RT to positive RT and choice
        abs_rt = pt.abs_(rt)
        choice = pt.where(rt > 0, 1, 0)  # 1 for LL, 0 for SS
        
        # Calculate the log-likelihood using the DDM formula
        # This is a simplified version that assumes unbiased starting point (beta=0.5)
        # and uses the first-passage time density for the Ornstein-Uhlenbeck process
        z = (a - choice * v * abs_rt) / (0.1 * pt.sqrt(abs_rt))  # Using fixed noise=0.1
        logp = pt.log(2 * pt.abs_(v) / (0.1**2)) - (v**2 * abs_rt) / (2 * 0.1**2) \
               - pt.log(a) - pt.log(2 * pt.pi * abs_rt) / 2 \
               + pt.log(pt.exp(-z**2 / 2) - pt.exp(-(2 * a * v / (0.1**2) - z)**2 / 2))
        return logp
    
    # Create the likelihood using DensityDist
    def ddm_logp(v, a, tau, rt):
        """DDM log-likelihood for a single trial."""
        # Convert signed RT to positive RT and choice
        abs_rt = pt.abs_(rt)
        choice = pt.where(rt > 0, 1, 0)  # 1 for LL, 0 for SS
        
        # Calculate the log-likelihood using the DDM formula
        # This is a simplified version that assumes unbiased starting point (beta=0.5)
        # and uses the first-passage time density for the Ornstein-Uhlenbeck process
        z = (a - choice * v * abs_rt) / (0.1 * pt.sqrt(abs_rt))  # Using fixed noise=0.1
        logp = pt.log(2 * pt.abs_(v) / (0.1**2)) - (v**2 * abs_rt) / (2 * 0.1**2) \
               - pt.log(a) - pt.log(2 * pt.pi * abs_rt) / 2 \
               + pt.log(pt.exp(-z**2 / 2) - pt.exp(-(2 * a * v / (0.1**2) - z)**2 / 2))
        return logp
    
    # Create the likelihood using DensityDist
    likelihood = pm.CustomDist(
        "likelihood",
        drift_v,
        thresh_subj[subject_idx],
        t_subj[subject_idx],
        logp=ddm_logp,
        observed=synthetic_data_df['signed_rt'].values
    )

# --- Run MCMC Sampler ---
print("\nStarting PyMC sampling (Prototype)...")
start_sample_time = time.time()
# Short run for feasibility check
n_draws = 500
n_tune = 1000 # More tuning often needed

with hbm_nes_dd:
    idata = pm.sample(draws=n_draws, tune=n_tune, chains=2, cores=1, target_accept=0.90) # Use 1 core for tool env

end_sample_time = time.time()
print(f"Sampling finished in {end_sample_time - start_sample_time:.2f} seconds.")

# --- Basic Convergence Check & Summary ---
print("\n--- MCMC Summary ---")
try:
    summary = az.summary(idata, var_names=['group_k_mu', 'group_thresh_mu', 'group_t_mu', '~drift_v']) # Summarize key group params
    print(summary)
    rhat_max = summary['r_hat'].max()
    print(f"\nMax R-hat: {rhat_max:.3f}")
    if rhat_max > 1.1:
        print("WARNING: Poor convergence suspected (Max R-hat > 1.1). Consider longer tuning/sampling.")
    else:
        print("Basic convergence looks acceptable (Max R-hat <= 1.1).")
except Exception as e:
    print(f"Could not generate ArviZ summary: {e}")

print("\n--- Feasibility Check Conclusion ---")
print("If the sampler ran without critical errors and R-hats are reasonable,")
print("then specifying the NES-DD model within an HBM framework using PyMC")
print("appears technically feasible. Full recovery analysis requires longer runs")
print("and plotting posteriors against true values on the full dataset.")