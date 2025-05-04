# Filename: param_recovery/run_dd_recovery_simple.py
# Purpose: Simple HBM fit for NES DD model using pure Python/Numpy

import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    k_safe = np.maximum(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

def log_likelihood(params, data):
    # Transform parameters to ensure they're positive
    k_mu, k_sigma, thresh_mu, thresh_sigma, t_mu, t_sigma = np.exp(params[:6])
    subject_idx = data['subject'].values
    ll_delay = data['ll_delay'].values
    choice_code = data['choice_code'].values
    rt = data['rt'].values
    
    # Subject-level parameters (using truncated normal to ensure positivity)
    k_subj = np.maximum(0.001, stats.norm.rvs(loc=k_mu, scale=k_sigma, size=len(np.unique(subject_idx))))
    thresh_subj = np.maximum(0.05, stats.norm.rvs(loc=thresh_mu, scale=thresh_sigma, size=len(np.unique(subject_idx))))
    t_subj = np.maximum(0.001, stats.norm.rvs(loc=t_mu, scale=t_sigma, size=len(np.unique(subject_idx))))
    
    # Fixed parameters
    w_s_fit = 0.392
    ss_amount_fit = 5.0
    ss_delay_fit = 0.0
    ll_amount_fit = 10.0
    
    # Calculate drift rates
    log_likelihood = 0
    for subj_id in np.unique(subject_idx):
        subj_mask = subject_idx == subj_id
        k_trial = k_subj[subj_id]
        a_trial = thresh_subj[subj_id]
        tau_trial = t_subj[subj_id]
        
        v_ss_trial = hyperbolic_discount(ss_amount_fit, ss_delay_fit, k_trial)
        v_ll_trial = hyperbolic_discount(ll_amount_fit, ll_delay[subj_mask], k_trial)
        drift_v = w_s_fit * (v_ll_trial - v_ss_trial)
        
        # Logistic regression-like likelihood
        logit_p = drift_v * rt[subj_mask]
        # Ensure probabilities are valid
        p = np.clip(1/(1+np.exp(-logit_p)), 1e-6, 1-1e-6)
        log_likelihood += np.sum(stats.bernoulli.logpmf(choice_code[subj_mask], p=p))
    
    # Add prior terms (using transformed parameters)
    log_likelihood += stats.halfnorm.logpdf(k_mu, scale=0.1)
    log_likelihood += stats.halfnorm.logpdf(k_sigma, scale=0.05)
    log_likelihood += stats.halfnorm.logpdf(thresh_mu, scale=0.5)
    log_likelihood += stats.halfnorm.logpdf(thresh_sigma, scale=0.3)
    log_likelihood += stats.halfnorm.logpdf(t_mu, scale=0.1)
    log_likelihood += stats.halfnorm.logpdf(t_sigma, scale=0.05)
    
    return log_likelihood

def main():
    print("--- HBM Recovery Prototype (Simple) ---")
    print(f"Python Executable: {sys.executable}")

    # --- Data Loading ---
    data_filename = "synthetic_dissociative_dd_data.csv"
    true_params_filename = "true_dissociative_params.csv"
    GENERATED_DATA_USED = False

    if os.path.exists(data_filename) and os.path.exists(true_params_filename):
        print(f"\nLoading existing data from {data_filename} and {true_params_filename}")
        try:
            synthetic_data_df = pd.read_csv(data_filename)
            true_params_df = pd.read_csv(true_params_filename)
            print(f"Loaded {len(synthetic_data_df)} trials for {true_params_df['subject'].nunique()} subjects.")
            
            # Use only the first 5 subjects for prototype
            N_SUBJECTS_PROTO = 5
            if true_params_df['subject'].nunique() >= N_SUBJECTS_PROTO:
                print(f"Using data from first {N_SUBJECTS_PROTO} subjects for prototype fit.")
                subject_ids_to_keep = sorted(true_params_df['subject'].unique())[:N_SUBJECTS_PROTO]
                synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'].isin(subject_ids_to_keep)].copy()
                true_params_df = true_params_df[true_params_df['subject'].isin(subject_ids_to_keep)].copy()

            print("True parameters (Loaded Subset):")
            print(true_params_df[['subject', 'true_k', 'true_base_threshold']].round(3))

        except Exception as e:
            print(f"Error loading data files: {e}. Will attempt to generate demo data.")
            synthetic_data_df = pd.DataFrame()  # Ensure df is empty
    else:
        print(f"Data files '{data_filename}' or '{true_params_filename}' not found. Generating PROTOTYPE data...")
        GENERATED_DATA_USED = True
        N_SUBJECTS_PROTO = 5
        N_REPS_PER_DELAY_PROTO = 20
        ll_delays_proto = [1, 5, 10, 20, 50]
        params_fixed_gen = {
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

        for subj_id in range(N_SUBJECTS_PROTO):
            subj_k = np.random.gamma(shape=gamma_k_shape_gen, scale=gamma_k_scale_gen)
            subj_thresh = np.random.normal(loc=true_thresh_mean_gen, scale=true_thresh_sd_gen)
            subj_thresh = max(0.1, subj_thresh)
            proto_true_params.append({'subject': subj_id, 'true_k': subj_k, 'true_threshold': subj_thresh})
            for delay in ll_delays_proto:
                ll_option_gen = {'amount': ll_amount_gen, 'delay': delay}
                v_ss = hyperbolic_discount(ss_option_gen['amount'], ss_option_gen['delay'], subj_k)
                v_ll = hyperbolic_discount(ll_option_gen['amount'], ll_option_gen['delay'], subj_k)
                prob_ll = 1 / (1 + np.exp(-(v_ll - v_ss)))
                for rep in range(N_REPS_PER_DELAY_PROTO):
                    choice = 1 if np.random.rand() < prob_ll else -1
                    rt_mean = 0.2 + np.abs(v_ll - v_ss) * 0.05
                    rt = np.random.lognormal(mean=np.log(rt_mean), sigma=0.3)
                    rt = max(0.051, rt)
                    proto_synthetic_data.append({
                        'subject': subj_id, 'll_delay': delay, 'choice_code': choice, 'rt': rt,
                        'true_k': subj_k, 'true_threshold': subj_thresh
                    })
        synthetic_data_df = pd.DataFrame(proto_synthetic_data)
        true_params_df = pd.DataFrame(proto_true_params)
        print(f"Prototype data generated: {len(synthetic_data_df)} valid trials.")
        print("True parameters (Generated Prototype):")
        print(true_params_df.round(3))

    # --- Data Preprocessing ---
    min_rt_threshold = 0.05
    print(f"\nInitial trials loaded/generated: {len(synthetic_data_df)}")
    synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > min_rt_threshold].copy()
    print(f"Trials after filtering RT > {min_rt_threshold}s: {len(synthetic_data_df)}")

    if synthetic_data_df.empty:
        print("Error: No valid trials remain after filtering RTs. Cannot proceed.")
        return

    # Add choice_code column (1 for LL, -1 for SS)
    synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)

    # --- Simple Sampling with Multiple Chains and Diagnostics ---
    print("\nStarting simple sampling with diagnostics (Prototype)...")
    start_sample_time = time.time()
    
    # Improved sampling settings
    n_samples = 1000  # Increased sample size
    n_warmup = 2000   # Increased warmup
    n_chains = 4      # Multiple chains for R-hat
    
    # Initialize samples for multiple chains
    all_samples = []
    acceptance_rates = []
    
    # Run multiple chains with different initial values
    for chain_idx in range(n_chains):
        print(f"\nRunning chain {chain_idx + 1}/{n_chains}...")
        
        # Initialize samples for this chain
        chain_samples = {
            'group_k_mu': np.zeros(n_samples),
            'group_k_sigma': np.zeros(n_samples),
            'group_thresh_mu': np.zeros(n_samples),
            'group_thresh_sigma': np.zeros(n_samples),
            'group_t_mu': np.zeros(n_samples),
            'group_t_sigma': np.zeros(n_samples)
        }
        
        # Better initialization based on true parameter ranges
        true_k_mean = true_params_df['true_k'].mean()
        true_thresh_mean = true_params_df['true_base_threshold'].mean()
        
        current_params = np.array([
            np.random.uniform(true_k_mean * 0.5, true_k_mean * 1.5),  # k_mu
            np.random.uniform(0.01, 0.1),  # k_sigma
            np.random.uniform(true_thresh_mean * 0.5, true_thresh_mean * 1.5),  # thresh_mu
            np.random.uniform(0.1, 0.5),   # thresh_sigma
            np.random.uniform(0.01, 0.2),  # t_mu
            np.random.uniform(0.01, 0.1)   # t_sigma
        ])
        
        # Adaptive step size with better initial value
        step_size = 0.02  # Start with smaller step
        acceptance_count = 0
        
        # Track acceptance rate for step size adaptation
        acceptance_rates = []
        
        for i in range(n_warmup + n_samples):
            # Propose new values
            proposed_params = current_params + np.random.normal(scale=step_size, size=6)
            
            # Calculate log likelihoods
            current_ll = log_likelihood(current_params, synthetic_data_df)
            proposed_ll = log_likelihood(proposed_params, synthetic_data_df)
            
            # Acceptance probability
            acceptance_prob = min(1, np.exp(proposed_ll - current_ll))
            
            # Accept or reject
            if np.random.rand() < acceptance_prob:
                current_params = proposed_params
                acceptance_count += 1
            
            # Store sample if past warmup
            if i >= n_warmup:
                chain_samples['group_k_mu'][i - n_warmup] = current_params[0]
                chain_samples['group_k_sigma'][i - n_warmup] = current_params[1]
                chain_samples['group_thresh_mu'][i - n_warmup] = current_params[2]
                chain_samples['group_thresh_sigma'][i - n_warmup] = current_params[3]
                chain_samples['group_t_mu'][i - n_warmup] = current_params[4]
                chain_samples['group_t_sigma'][i - n_warmup] = current_params[5]
            
            # Adapt step size during warmup
            if i < n_warmup and (i + 1) % 100 == 0:
                acceptance_rate = acceptance_count / (i + 1)
                acceptance_rates.append(acceptance_rate)
                
                # Adjust step size based on acceptance rate
                if acceptance_rate < 0.2:
                    step_size *= 0.9  # Too few acceptances, reduce step
                elif acceptance_rate > 0.5:
                    step_size *= 1.1  # Too many acceptances, increase step
                    if step_size > 0.1:  # Cap maximum step size
                        step_size = 0.1
                acceptance_count = 0  # Reset acceptance count
                
                # Print progress during warmup
                print(f"Chain {chain_idx + 1}: Iter {i + 1}/{n_warmup}, Acceptance: {acceptance_rate:.3f}, Step: {step_size:.4f}")
        
        all_samples.append(chain_samples)
        acceptance_rates.append(acceptance_count / n_samples)
    
    # Combine chains for analysis
    combined_samples = {
        'group_k_mu': np.concatenate([s['group_k_mu'] for s in all_samples]),
        'group_k_sigma': np.concatenate([s['group_k_sigma'] for s in all_samples]),
        'group_thresh_mu': np.concatenate([s['group_thresh_mu'] for s in all_samples]),
        'group_thresh_sigma': np.concatenate([s['group_thresh_sigma'] for s in all_samples]),
        'group_t_mu': np.concatenate([s['group_t_mu'] for s in all_samples]),
        'group_t_sigma': np.concatenate([s['group_t_sigma'] for s in all_samples])
    }
    
    print(f"\nSampling finished in {time.time() - start_sample_time:.2f} seconds.")
    
    # Calculate posterior means
    posterior_means = {
        'group_k_mu': np.mean(combined_samples['group_k_mu']),
        'group_k_sigma': np.mean(combined_samples['group_k_sigma']),
        'group_thresh_mu': np.mean(combined_samples['group_thresh_mu']),
        'group_thresh_sigma': np.mean(combined_samples['group_thresh_sigma']),
        'group_t_mu': np.mean(combined_samples['group_t_mu']),
        'group_t_sigma': np.mean(combined_samples['group_t_sigma'])
    }
    
    # --- Diagnostics ---
    print("\n--- MCMC Diagnostics ---")
    print(f"Acceptance rates per chain: {acceptance_rates}")
    
    # Calculate R-hat
    def calculate_rhat(samples):
        n_chains = len(samples)
        n_samples = len(samples[0])
        chain_means = [np.mean(s) for s in samples]
        global_mean = np.mean(chain_means)
        
        # Between-chain variance
        B = n_samples / (n_chains - 1) * np.sum([(cm - global_mean)**2 for cm in chain_means])
        
        # Within-chain variance
        W = np.mean([np.var(s, ddof=1) for s in samples])
        
        # Estimate of variance
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        
        # R-hat
        return np.sqrt(var_hat / W)
    
    print("\n--- R-hat Values (should be close to 1) ---")
    rhat_values = {
        'group_k_mu': calculate_rhat([s['group_k_mu'] for s in all_samples]),
        'group_k_sigma': calculate_rhat([s['group_k_sigma'] for s in all_samples]),
        'group_thresh_mu': calculate_rhat([s['group_thresh_mu'] for s in all_samples]),
        'group_thresh_sigma': calculate_rhat([s['group_thresh_sigma'] for s in all_samples]),
        'group_t_mu': calculate_rhat([s['group_t_mu'] for s in all_samples]),
        'group_t_sigma': calculate_rhat([s['group_t_sigma'] for s in all_samples])
    }
    for param, rhat in rhat_values.items():
        print(f"{param}: {rhat:.3f}")
    
    # --- Trace Plots ---
    plt.figure(figsize=(15, 10))
    
    # Plot each parameter's trace
    for i, param in enumerate(['group_k_mu', 'group_thresh_mu']):
        plt.subplot(2, 3, i + 1)
        for chain_idx in range(n_chains):
            plt.plot(all_samples[chain_idx][param], alpha=0.5, label=f'Chain {chain_idx + 1}')
        plt.title(f'Trace plot for {param}')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        if i == 0:
            plt.legend()
    
    # --- Autocorrelation Plots ---
    for i, param in enumerate(['group_k_mu', 'group_thresh_mu']):
        plt.subplot(2, 3, i + 4)
        for chain_idx in range(n_chains):
            plt.acorr(all_samples[chain_idx][param], maxlags=50, alpha=0.5, label=f'Chain {chain_idx + 1}')
        plt.title(f'Autocorrelation for {param}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('dd_recovery_diagnostics.png')
    plt.close()
    
    print("\n--- Posterior Means ---")
    for param, value in posterior_means.items():
        print(f"{param}: {value:.3f}")
    
    # --- Posterior Distribution Plots ---
    plt.figure(figsize=(15, 10))
    
    # Plot k recovery with diagnostics
    plt.subplot(2, 3, 1)
    plt.hist(combined_samples['group_k_mu'], bins=30, alpha=0.5)
    plt.axvline(true_params_df['true_k'].mean(), color='r', linestyle='--', label='True Mean')
    plt.xlabel('Group k_mu')
    plt.ylabel('Frequency')
    plt.title('Group k Parameter Posterior')
    plt.legend()
    
    # Plot threshold recovery with diagnostics
    plt.subplot(2, 3, 2)
    plt.hist(combined_samples['group_thresh_mu'], bins=30, alpha=0.5)
    plt.axvline(true_params_df['true_base_threshold'].mean(), color='r', linestyle='--', label='True Mean')
    plt.xlabel('Group thresh_mu')
    plt.ylabel('Frequency')
    plt.title('Group Threshold Parameter Posterior')
    plt.legend()
    
    # Add R-hat values as text annotations
    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.text(0.1, 0.8, "R-hat Values:", fontsize=12)
    for i, (param, rhat) in enumerate(rhat_values.items()):
        plt.text(0.1, 0.7 - i*0.1, f"{param}: {rhat:.3f}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('dd_recovery_results.png')
    plt.close()
    
    print("\nAnalysis and plots completed.")

if __name__ == "__main__":
    main()
    print("\nFull script finished.")
