import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Hyperbolic discounting function
def hyperbolic_discount(amount, delay, k):
    return amount / (1.0 + k * delay)

# Simplified model using built-in distributions
def main():
    print("--- HBM Recovery Prototype (PyMC) ---")
    print(f"Python Executable: {sys.executable}")
    print()

def main():
    print("--- HBM Recovery Prototype (PyMC) ---")
    print(f"Python Executable: {sys.executable}")
    print()
    
    # Load data
    try:
        synthetic_data_df = pd.read_csv('synthetic_dissociative_dd_data.csv')
        true_params_df = pd.read_csv('true_dissociative_params.csv')
        print("\nLoading existing data from synthetic_dissociative_dd_data.csv and true_dissociative_params.csv")
    except FileNotFoundError:
        print("\nGenerating demo dataset...")
        # Create demo data
        n_subjects = 5
        n_trials = 540
        
        # Generate synthetic data
        synthetic_data_df = pd.DataFrame({
            'subject': np.repeat(np.arange(n_subjects), n_trials),
            'll_delay': np.random.uniform(1, 30, n_subjects * n_trials),
            'choice': np.random.choice(['Choose_LL', 'Choose_SS'], n_subjects * n_trials),
            'rt': np.random.uniform(0.2, 2.0, n_subjects * n_trials)
        })
        
        # Add choice_code column (1 for LL, -1 for SS)
        synthetic_data_df['choice_code'] = np.where(synthetic_data_df['choice'] == 'Choose_LL', 1, -1)
        
        # Save demo data
        synthetic_data_df.to_csv('synthetic_dissociative_dd_data.csv', index=False)
        
        # Create demo true parameters
        true_params_df = pd.DataFrame({
            'subject': np.arange(n_subjects),
            'true_k': np.random.uniform(0.01, 0.1, n_subjects),
            'true_base_threshold': np.random.uniform(0.2, 0.8, n_subjects)
        })
        true_params_df.to_csv('true_dissociative_params.csv', index=False)
        
        print("Demo dataset generated and saved.")
    
    print(f"\nLoaded {len(synthetic_data_df)} trials for {len(true_params_df)} subjects.")
    print("Using data from first 5 subjects for prototype fit.")
    print("\nTrue parameters (Loaded Subset):")
    print(true_params_df.head())
    
    # Filter data for prototype
    synthetic_data_df = synthetic_data_df[synthetic_data_df['subject'] < 5]
    
    # Filter trials with RT < 0.05s
    print(f"\nInitial trials loaded/generated: {len(synthetic_data_df)}")
    synthetic_data_df = synthetic_data_df[synthetic_data_df['rt'] > 0.05]
    print(f"Trials after filtering RT > 0.05s: {len(synthetic_data_df)}")
    
    # Convert data to PyMC-friendly format
    data = {
        'subject': synthetic_data_df['subject'].values,
        'll_delay': synthetic_data_df['ll_delay'].values,
        'choice_code': synthetic_data_df['choice_code'].values,
        'rt': synthetic_data_df['rt'].values
    }
    
    # --- PyMC Model ---
    print("\nBuilding PyMC model...")
    
    with pm.Model() as model:
        # Group-level parameters
        k_mu = pm.HalfNormal('k_mu', sigma=0.1)
        k_sigma = pm.HalfNormal('k_sigma', sigma=0.05)
        thresh_mu = pm.HalfNormal('thresh_mu', sigma=0.5)
        thresh_sigma = pm.HalfNormal('thresh_sigma', sigma=0.3)
        t_mu = pm.HalfNormal('t_mu', sigma=0.1)
        t_sigma = pm.HalfNormal('t_sigma', sigma=0.05)
        
        # Subject-level parameters
        k_subj = pm.Normal('k_subj', mu=k_mu, sigma=k_sigma, shape=len(np.unique(data['subject'])))
        thresh_subj = pm.Normal('thresh_subj', mu=thresh_mu, sigma=thresh_sigma, shape=len(np.unique(data['subject'])))
        t_subj = pm.Normal('t_subj', mu=t_mu, sigma=t_sigma, shape=len(np.unique(data['subject'])))
        
        # Fixed parameters
        w_s_fit = 0.392
        ss_amount_fit = 5.0
        ss_delay_fit = 0.0
        ll_amount_fit = 10.0
        
        # Calculate drift rates and likelihood for each trial
        for i in range(len(data['subject'])):
            subj_id = data['subject'][i]
            ll_delay = data['ll_delay'][i]
            choice = data['choice_code'][i]
            rt = data['rt'][i]
            
            # Get subject-specific parameters
            k_trial = k_subj[subj_id]
            a_trial = thresh_subj[subj_id]
            tau_trial = t_subj[subj_id]
            
            # Calculate drift rate
            v_ss_trial = hyperbolic_discount(ss_amount_fit, ss_delay_fit, k_trial)
            v_ll_trial = hyperbolic_discount(ll_amount_fit, ll_delay, k_trial)
            drift_v = w_s_fit * (v_ll_trial - v_ss_trial)
            
            # Use logistic regression-like model for choice
            choice_prob = pm.math.invlogit(drift_v * rt)
            pm.Bernoulli(f'choice_{i}', p=choice_prob, observed=(choice + 1) / 2)
            
            # Use normal model for RT
            rt_prob = pm.Normal(f'rt_{i}', mu=tau_trial, sigma=0.1)
            pm.Potential(f'rt_obs_{i}', rt_prob.logp(rt))
    
    # --- Sampling ---
    print("\nStarting PyMC sampling with diagnostics...")
    start_sample_time = time.time()
    
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True
        )
    
    print(f"\nSampling finished in {time.time() - start_sample_time:.2f} seconds.")
    
    # --- Diagnostics ---
    print("\n--- MCMC Diagnostics ---")
    
    # Calculate R-hat
    print("\n--- R-hat Values (should be close to 1) ---")
    rhat = az.rhat(trace)
    for param, rhat_val in rhat.items():
        print(f"{param}: {np.max(rhat_val):.3f}")
    
    # Calculate posterior means
    posterior_means = {
        'group_k_mu': trace.posterior['k_mu'].mean().item(),
        'group_k_sigma': trace.posterior['k_sigma'].mean().item(),
        'group_thresh_mu': trace.posterior['thresh_mu'].mean().item(),
        'group_thresh_sigma': trace.posterior['thresh_sigma'].mean().item(),
        'group_t_mu': trace.posterior['t_mu'].mean().item(),
        'group_t_sigma': trace.posterior['t_sigma'].mean().item()
    }
    
    print("\n--- Posterior Means ---")
    for param, value in posterior_means.items():
        print(f"{param}: {value:.3f}")
    
    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    
    # Plot k recovery with diagnostics
    plt.subplot(2, 3, 1)
    az.plot_posterior(trace.posterior['k_mu'])
    plt.axvline(true_params_df['true_k'].mean(), color='r', linestyle='--', label='True Mean')
    plt.title('Group k Parameter Posterior')
    plt.legend()
    
    # Plot threshold recovery with diagnostics
    plt.subplot(2, 3, 2)
    az.plot_posterior(trace.posterior['thresh_mu'])
    plt.axvline(true_params_df['true_base_threshold'].mean(), color='r', linestyle='--', label='True Mean')
    plt.title('Group Threshold Parameter Posterior')
    plt.legend()
    
    # Add R-hat values as text annotations
    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.text(0.1, 0.8, "R-hat Values:", fontsize=12)
    for i, (param, rhat_val) in enumerate(rhat.items()):
        plt.text(0.1, 0.7 - i*0.1, f"{param}: {np.max(rhat_val):.3f}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('dd_recovery_results.png')
    plt.close()
    
    print("\nAnalysis and plots completed.")
    
    # Save trace for future analysis
    az.to_netcdf(trace, 'dd_recovery_trace.nc')
    print("\nTrace saved to dd_recovery_trace.nc")

if __name__ == "__main__":
    main()
