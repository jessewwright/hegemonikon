# nes_parameter_recovery.py

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from agent_mvnes import MVNESAgent
import matplotlib.pyplot as plt
import pytensor.tensor as at
from pytensor.compile.mode import get_mode

# Build a FAST_COMPILE mode without the buggy rewrites
fast_mode = get_mode("FAST_COMPILE").excluding("canonicalize", "stabilize")

# Configuration variables
n_subjects = 1  # Single subject for testing
n_trials   = 50  # Reduced trials for pilot test

if __name__ == '__main__':
    # Define drift function
    def drift_function(w_n, w_s, salience, norm):
        """Calculate drift rate based on MVNES model parameters."""
        return w_n * salience + w_s * norm

    print("Starting parameter recovery...")
    print(f"Generating data for {n_subjects} subjects with {n_trials} trials each")
    
    # 1. SIMULATION: synthetic data
    true_params = {'w_n': np.random.normal(1.0, 0.2, size=n_subjects),
                   'a'  : np.random.normal(1.5, 0.3, size=n_subjects),
                   'w_s': np.random.normal(0.5, 0.1, size=n_subjects)}
    print("True parameters:")
    print(f"w_n mean: {true_params['w_n'].mean():.2f}, std: {true_params['w_n'].std():.2f}")
    print(f"a mean: {true_params['a'].mean():.2f}, std: {true_params['a'].std():.2f}")
    print(f"w_s mean: {true_params['w_s'].mean():.2f}, std: {true_params['w_s'].std():.2f}")

    data = []
    for subj in range(n_subjects):
        agent = MVNESAgent(config={'veto_flag': False})
        for t in range(n_trials):
            # Simulate a Go trial (salience=1, norm=0)
            trial_data = agent.run_mvnes_trial(
                salience_input=1.0,
                norm_input=0.0,
                params={
                    'w_s': true_params['w_s'][subj],
                    'w_n': true_params['w_n'][subj],
                    'threshold_a': true_params['a'][subj],
                    't': 0.1,
                    'noise_std_dev': 1.0,
                    'dt': 0.01,
                    'max_time': 2.0,
                    'affect_stress_threshold_reduction': -0.3
                }
            )
            data.append({
                'subj': subj,
                'rt': trial_data['rt'],
                'choice': trial_data['choice'],
                'salience_input': 1.0,  # Store the input values
                'norm_input': 0.0     # Store the input values
            })
    df = pd.DataFrame(data)

    # 2. MINIMAL PILOT MODEL (for single parameter inference)
    with pm.Model() as nes_hbm:
        print("\nModel structure:")
        print("- Creating priors and parameters...")
        
        # Fix threshold and norm drift at true values
        a   = 1.5  # Fixed threshold
        w_s = 0.5  # Fixed norm drift
        
        # Only infer salience drift
        w_n = pm.Normal("w_n", 1.0, 1.0)  # Drift rate for salience

        # Get data for single subject
        subj = 0
        subj_data = df.query('subj==@subj')
        
        # Build drift vector for all trials symbolically
        drift = w_n * at.as_tensor_variable(subj_data['salience_input'].values) \
              + w_s * at.as_tensor_variable(subj_data['norm_input'].values)

        # Create deterministic drift for diagnostics
        pm.Deterministic("drift", drift)
        
        # Custom log-likelihood function for joint RT + choice using pure PyTensor ops
        def joint_logp(value, drift, threshold, t0):
            """Calculate log-likelihood for joint RT and choice using PyTensor operations."""
            rt = value[0]  # Reaction time
            choice = value[1]  # 1 for correct, 0 for error
            
            # Convert choice to direction (1 for correct, -1 for error)
            direction = 2 * choice - 1
            
            # Calculate drift term
            w = (rt - t0) * drift * direction
            
            # Calculate z-score
            z = (threshold + w) / at.sqrt(2 * at.abs(w) + 1e-8)
            
            # Calculate log-likelihood for choice (using erf)
            logp_choice = at.log(0.5 * (1 + at.erf(direction * w / at.sqrt(4 * at.abs(w) + 1e-8))))
            
            # Calculate log-likelihood for RT (using normal density)
            logp_rt = -0.5 * z**2 - 0.5 * at.log(2 * np.pi) - at.log(at.sqrt(2 * at.abs(w) + 1e-8))
            
            # Return combined log-likelihood
            return logp_choice + logp_rt
        
        # Create DensityDist for joint RT + choice
        # Note: Parameters must be positional arguments before logp
        pm.DensityDist(
            "obs_wiener",
            drift,
            a,
            0.1,  # t0 as fixed value
            logp=joint_logp,
            observed=[
                subj_data['rt'].values,
                subj_data['choice'].values
            ]
        )
        
        print("\nModel created successfully!")

    # 3. QUICK VI CHECK
    with nes_hbm:
        print("\nRunning Variational Inference for quick model check...")
        approx = pm.fit(20000, method="advi", callbacks=[pm.callbacks.CheckParametersConvergence()])
        trace_vi = approx.sample(500)
        
        # Print VI results
        print("\nVI Results:")
        print("Posterior means:")
        print(pm.summary(trace_vi).round(2))
        
    # 3. INFERENCE (NUTS)
    with nes_hbm:
        print("\nStarting inference with NUTS sampler...")
        print("Using 2 chains with FAST_COMPILE")
        print("Tuning for 200 iterations")
        print("Using custom FAST_COMPILE mode to avoid problematic rewrites")
        
        # Sample with diagnostics
        trace = pm.sample(
            draws=200,
            tune=200,
            chains=2,
            cores=2,
            target_accept=0.9,
            compile_kwargs={"mode": fast_mode},
            return_inferencedata=True
        )
        
        # Print summary of results
        print("\nSampling complete. Results:")
        print(pm.summary(trace))
        
        # Plot posterior
        az.plot_trace(trace)
        plt.show()

    # 4. DIAGNOSTICS: recovered vs true
    print("\nInference complete. Generating recovery plots...")
    
    # Since we're fitting only a single w_n, get it from trace
    w_n_post = trace.posterior['w_n'].mean(dim=('chain','draw')).values
    az.plot_posterior(
        {'w_n': w_n_post, 'a': a_post, 'w_s': w_s_post},
        ax=axes,
        hdi_prob=0.95
    )
    
    # Add true values as vertical lines
    for ax, param in zip(axes, ['w_n', 'a', 'w_s']):
        ax.axvline(true_params[param][0], color='r', linestyle='--', label='True value')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
