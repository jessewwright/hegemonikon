# Filename: param_recovery/run_dd_recovery_numpyro.py
# Purpose: Prototype HBM fit for NES DD model using NumPyro

import numpy as np
import pandas as pd
import time
import numpyro
import numpyro.distributions as dist
import arviz as az
import matplotlib.pyplot as plt
import os
import sys

# --- Helper Functions ---
def hyperbolic_discount(amount, delay, k):
    k_safe = np.maximum(k, 1e-7)
    return amount / (1.0 + k_safe * delay)

def run_dd_model(data):
    n_subjects = len(np.unique(data['subject']))
    n_obs = len(data)
    
    # Extract data
    subject_idx = data['subject'].values
    ll_delay = data['ll_delay'].values
    choice_code = data['choice_code'].values
    rt = data['rt'].values
    
    with numpyro.plate("subjects", n_subjects):
        # Group-level priors
        group_k_mu = numpyro.sample("group_k_mu", dist.HalfNormal(0.1))
    
    # Fixed parameters
    w_s_fit = 0.392
    ss_amount_fit = 5.0
    ss_delay_fit = 0.0
    ll_amount_fit = 10.0
    
    # Calculate drift rates and likelihood for each trial
    for i in range(len(data)):
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
        
        # Calculate likelihood
        with numpyro.plate('trials', 1):
            numpyro.sample('obs', dist.Delta(ddm_likelihood(drift_v, a_trial, tau_trial, rt, choice)), obs=np.array([1]))

def main():
    print("--- HBM Recovery Prototype (NumPyro) ---")
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

    # --- Run MCMC Sampling ---
    print("\nStarting NumPyro sampling (Prototype)...")
    start_sample_time = time.time()
    
    # Convert data to NumPy arrays
    data_dict = {
        'subject': synthetic_data_df['subject'].values,
        'll_delay': synthetic_data_df['ll_delay'].values,
        'choice_code': synthetic_data_df['choice_code'].values,
        'rt': synthetic_data_df['rt'].values
    }
    
    # Run MCMC
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(run_dd_model),
        num_warmup=1000,
        num_samples=500,
        num_chains=2
    )
    
    try:
        mcmc.run(numpyro.diagnostics.random.PRNGKey(0), data_dict)
        print(f"Sampling finished in {time.time() - start_sample_time:.2f} seconds.")
        
        # Convert to ArviZ InferenceData
        posterior = mcmc.get_samples()
        idata = az.from_numpyro(mcmc)
        
        # Print summary
        print("\n--- MCMC Summary (Group Parameters) ---")
        print(az.summary(idata, var_names=['group_k_mu', 'group_k_sigma', 'group_thresh_mu', 'group_thresh_sigma', 'group_t_mu', 'group_t_sigma'], round_to=3))
        
        # Extract subject-level parameters
        k_recovered_mean = posterior['k_subj'].mean(axis=0)
        thresh_recovered_mean = posterior['thresh_subj'].mean(axis=0)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot k recovery
        plt.subplot(1, 2, 1)
        plt.scatter(true_params_df['true_k'], k_recovered_mean)
        plt.plot([0, 0.1], [0, 0.1], 'k--')
        plt.xlabel('True k')
        plt.ylabel('Recovered k')
        plt.title('k Parameter Recovery')
        
        # Plot threshold recovery
        plt.subplot(1, 2, 2)
        plt.scatter(true_params_df['true_threshold'], thresh_recovered_mean)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True Threshold')
        plt.ylabel('Recovered Threshold')
        plt.title('Threshold Parameter Recovery')
        
        plt.tight_layout()
        plt.savefig('dd_recovery_plots.png')
        plt.close()
        
        print("\nAnalysis and plots completed.")
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        print("\nSkipping analysis and plotting due to sampling error.")

if __name__ == "__main__":
    main()
    print("\nFull script finished.")
