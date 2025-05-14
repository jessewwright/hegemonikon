"""
quick_hddm_recovery_test.py

A diagnostic script to test HDDM's parameter recovery with various configurations.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import hddm
import matplotlib.pyplot as plt
from datetime import datetime
import torch

# Try to import FlatHDDM, but make it optional
try:
    from hddm.models.hddm_flat import HDDM as FlatHDDM
    HAS_FLAT_HDDM = True
except ImportError:
    HAS_FLAT_HDDM = False
    print("Note: hddm_flat module not found. Flat model fitting will be disabled.")

from src.models.sbi_wrapper import SBINESEstimator

def run_hybrid_inference(trials, prior_params):
    def hddm_compat_simulator(params):
        # Wrapper that matches HDDM's expected format
        return torch.tensor(run_mvnes_trial(params))
    
    estimator = SBINESEstimator(hddm_compat_simulator, prior_params)
    return estimator.train_posterior(num_simulations=500)

def parse_args():
    parser = argparse.ArgumentParser(description='Run HDDM parameter recovery test')
    if HAS_FLAT_HDDM:
        parser.add_argument('--flat', action='store_true', help='Use flat (non-hierarchical) model')
    parser.add_argument('--trials', type=int, default=3000, help='Number of trials per subject')
    parser.add_argument('--subjects', type=int, default=5, help='Number of subjects')
    parser.add_argument('--samples', type=int, default=2000, help='Number of MCMC samples')
    parser.add_argument('--burn', type=int, default=500, help='Number of burn-in samples')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid mode with SBI posterior')
    
    args = parser.parse_args()
    
    # If flat model was requested but not available, warn and continue with hierarchical
    if hasattr(args, 'flat') and args.flat and not HAS_FLAT_HDDM:
        print("Warning: Flat model requested but not available. Using hierarchical model.")
        args.flat = False
        
    return args

def run_quick_recovery_test():
    """Run a parameter recovery test with configurable settings."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Determine model type
    model_type = "hierarchical"
    if hasattr(args, 'flat') and args.flat and HAS_FLAT_HDDM:
        model_type = "flat"
    elif hasattr(args, 'hybrid') and args.hybrid:
        model_type = "hybrid"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"hddm_recovery_results_{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Database name
    db_name = os.path.join(output_dir, f"{model_type}_recovery_{timestamp}.db")
    
    # --- 1. Define True Parameters ---
    TRUE_PARAMS = {
        'a': 1.5,      # Decision threshold
        'v': 0.5,      # Drift rate
        't': 0.2,      # Non-decision time (seconds)
        'z': 0.5,      # Starting point (fixed)
        'sv': 0.0,     # Inter-trial drift variability (fixed)
        'sz': 0.0,     # Starting point variability (fixed)
        'st': 0.0      # Non-decision time variability (fixed)
    }
    
    # --- 2. Generate Synthetic Data ---
    print("Generating synthetic data with known parameters...")
    print(f"Configuration: {model_type.capitalize()} model, "
          f"{args.subjects} subjects, {args.trials} trials/subject")
    
    n_trials = args.trials
    n_subjects = args.subjects
    
    # Generate data for each subject and combine
    all_data = []
    for subj in range(n_subjects):
        # Generate data for one subject with all parameters
        subj_data, _ = hddm.generate.gen_rand_data({
            'a': TRUE_PARAMS['a'],
            'v': TRUE_PARAMS['v'],
            't': TRUE_PARAMS['t'],
            'sv': TRUE_PARAMS['sv'],
            'sz': TRUE_PARAMS['sz'],
            'st': TRUE_PARAMS['st'],
            'z': TRUE_PARAMS['z']
        }, size=n_trials, subjs=1)
        
        # Add subject ID
        subj_data['subj_idx'] = subj
        all_data.append(subj_data)
    
    # Combine all subject data
    data = pd.concat(all_data, ignore_index=True)
    print(f"Generated {len(data)} trials across {n_subjects} subjects")
    
    # --- 3. Fit HDDM Model ---
    print("\nFitting HDDM model...")
    
    # Explicitly include only the parameters we want to estimate (a, v, t)
    include_params = {
        'v': True,  # Drift rate
        'a': True,  # Boundary separation
        't': True,  # Non-decision time
        'sv': False,  # Fixed at 0
        'sz': False,  # Fixed at 0
        'st': False,  # Fixed at 0
        'z': False   # Fixed at 0.5
    }
    
    if hasattr(args, 'flat') and args.flat and HAS_FLAT_HDDM:
        print("Using flat (non-hierarchical) model")
        model = FlatHDDM(
            data,
            include=include_params,  # Only estimate a, v, t
            bias=False,
            p_outlier=0.05
        )
    elif hasattr(args, 'hybrid') and args.hybrid:
        print("Using hybrid model with SBI posterior")
        posterior = run_hybrid_inference(n_trials, include_params)
        # Use the posterior as the model
        model = posterior
    else:
        print("Using hierarchical model")
        model = hddm.HDDM(
            data,
            include=include_params,  # Only estimate a, v, t
            bias=False,
            p_outlier=0.05
        )
    
    # Fit the model
    print(f"\nFitting model with {args.samples} samples (burn={args.burn}, thin=1)...")
    
    # Set reasonable starting values close to true values
    if not hasattr(args, 'hybrid') or not args.hybrid:
        model.find_starting_values()
    
    # Run MCMC sampling
    if not hasattr(args, 'hybrid') or not args.hybrid:
        model.sample(
            args.samples,
            burn=args.burn,
            thin=1  # No thinning for maximum effective sample size
        )
    
    # --- 4. Analyze Results ---
    print("\nAnalyzing results...")
    
    # Print and save summary statistics
    print("\nPosterior Summary Statistics:")
    if hasattr(args, 'hybrid') and args.hybrid:
        stats = model.summary()
    else:
        stats = model.gen_stats()
    print(stats)
    
    # Save stats to CSV
    stats_file = os.path.join(output_dir, f"recovery_stats_{timestamp}.csv")
    stats.to_csv(stats_file)
    
    # Save trace plot for each parameter
    print("\nSaving trace plots...")
    for param in ['a', 'v', 't']:
        if param in include_params and include_params[param]:
            plt.figure(figsize=(10, 4))
            if hasattr(args, 'hybrid') and args.hybrid:
                trace = model[param].detach().numpy()
            else:
                trace = model.nodes_db.node[param].trace()
            plt.plot(trace, alpha=0.7)
            plt.title(f'Trace plot for {param}')
            plt.xlabel('Sample')
            plt.ylabel(f'Value of {param}')
            trace_plot_file = os.path.join(output_dir, f"trace_{param}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(trace_plot_file)
            plt.close()
            
            # Save trace data
            pd.DataFrame({param: trace}).to_csv(
                os.path.join(output_dir, f"trace_{param}_{timestamp}.csv"),
                index=False
            )
    
    # Skip saving the full model to avoid database issues
    print("\nSkipping model save to avoid database issues")
    
    # If flat model, also fit each subject individually
    if hasattr(args, 'flat') and args.flat and HAS_FLAT_HDDM and n_subjects > 1:
        print("\nFitting individual subjects...")
        for subj_idx in range(n_subjects):
            subj_data = data[data['subj_idx'] == subj_idx].copy()
            subj_db = os.path.join(output_dir, f"recovery_subj{subj_idx}_{timestamp}.db")
            
            print(f"  Fitting subject {subj_idx+1}/{n_subjects}")
            
            try:
                # Fit individual subject
                subj_model = FlatHDDM(
                    subj_data,
                    include=include_params,
                    bias=False,
                    p_outlier=0.05,
                    db='pickle',
                    dbname=subj_db
                )
                subj_model.find_starting_values()
                subj_model.sample(args.samples // 2, burn=args.burn // 2, thin=2)
                
                # Save subject stats
                subj_stats = subj_model.gen_stats()
                subj_stats_file = os.path.join(output_dir, f"recovery_subj{subj_idx}_stats_{timestamp}.csv")
                subj_stats.to_csv(subj_stats_file)
                
                # Skip saving subject model to avoid database issues
                pass
                
            except Exception as e:
                print(f"    Error fitting subject {subj_idx}: {str(e)}")
    
    # Plot posterior distributions for main parameters
    fig = plt.figure(figsize=(12, 4))
    for i, param in enumerate(['a', 'v', 't']):
        plt.subplot(1, 3, i+1)
        if hasattr(args, 'hybrid') and args.hybrid:
            posterior = model[param].detach().numpy()
        else:
            posterior = model.nodes_db.node[param].trace()
        plt.hist(posterior, bins=20, alpha=0.7)
        plt.axvline(TRUE_PARAMS[param], color='red', linestyle='--', label='True')
        plt.title(f"{param} (True: {TRUE_PARAMS[param]:.2f})")
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f"recovery_plot_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"\nPlot saved to {plot_file}")
    
    # Save recovery summary to text file
    summary_file = os.path.join(output_dir, f"recovery_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"=== HDDM Parameter Recovery Test ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {model_type.capitalize()} model\n")
        f.write(f"Subjects: {n_subjects}\n")
        f.write(f"Trials per subject: {n_trials}\n")
        f.write(f"Samples: {args.samples} (burn={args.burn}, thin=1)\n")
        f.write(f"Parameters estimated: {', '.join([k for k, v in include_params.items() if v])}\n\n")
        
        f.write("True Parameters:\n")
        for param, value in TRUE_PARAMS.items():
            f.write(f"  {param}: {value}\n")
            
        f.write("\nRecovery Results:\n")
        for param in ['a', 'v', 't']:
            if param in stats.index:
                row = stats.loc[param]
                true_val = TRUE_PARAMS[param]
                recov_mean = row['mean']
                rel_error = abs((recov_mean - true_val) / true_val) * 100
                f.write(f"  {param}: True={true_val:.4f}, Recovered={recov_mean:.4f} "
                       f"(Error: {rel_error:.1f}%)\n")
    
    print(f"Detailed summary saved to {summary_file}")
    
    # Print parameter recovery summary
    print("\nParameter Recovery Summary:")
    print("Parameter  True Value  Recovered Mean  Recovered 95% HPD  Recovered")
    print("-" * 70)
    
    recovery_results = []
    for param in ['a', 'v', 't']:
        try:
            if hasattr(args, 'hybrid') and args.hybrid:
                trace = model[param].detach().numpy()
            else:
                trace = model.nodes_db.node[param].trace()
            trace_mean = np.mean(trace)
            hpd = np.percentile(trace, [2.5, 97.5])
            
            # Check if true value is within 95% HPD
            true_val = TRUE_PARAMS[param]
            within_hpd = hpd[0] <= true_val <= hpd[1]
            
            # Calculate relative error
            rel_error = abs((trace_mean - true_val) / true_val) * 100
            
            print(f"{param:9} {true_val:11.4f} {trace_mean:14.4f}  [{hpd[0]:.4f}, {hpd[1]:.4f}]  ", 
                  f"{'✓' if within_hpd else '✗'} (Error: {rel_error:.1f}%)")
            
            recovery_results.append({
                'parameter': param,
                'true_value': true_val,
                'recovered_mean': trace_mean,
                'hpd_lower': hpd[0],
                'hpd_upper': hpd[1],
                'within_hpd': within_hpd,
                'relative_error_pct': rel_error
            })
        except Exception as e:
            print(f"Could not compute statistics for {param}: {str(e)}")
            recovery_results.append({
                'parameter': param,
                'error': str(e)
            })
    
    # Save recovery results to CSV
    recovery_df = pd.DataFrame(recovery_results)
    summary_csv = os.path.join(output_dir, f"recovery_summary_{timestamp}.csv")
    recovery_df.to_csv(summary_csv, index=False)
    
    # Also save the posterior samples for analysis
    for param in ['a', 'v', 't']:
        if param in model.nodes_db.node:
            trace = model.nodes_db.node[param].trace()
            trace_df = pd.DataFrame({param: trace})
            trace_file = os.path.join(output_dir, f"trace_{param}_{timestamp}.csv")
            trace_df.to_csv(trace_file, index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"- {os.path.basename(plot_file)}")
    print(f"- {os.path.basename(stats_file)}")
    print(f"- {os.path.basename(summary_file)}")
    print(f"- {os.path.basename(summary_csv)}")
    
    # Show the plot if possible
    try:
        plt.show()
    except:
        print("\nNote: Could not display plot interactively. Check the saved PNG file.")
    
    # Print final summary
    print("\n=== Recovery Test Complete ===")
    print(f"Model type: {model_type.capitalize()}")
    print(f"Subjects: {n_subjects}, Trials/subject: {n_trials}")
    print(f"Samples: {args.samples} (burn={args.burn}, thin=1)")
    print(f"Effective samples: ~{args.samples - args.burn} (assuming good mixing)")
    
    # Print parameter recovery summary
    print("\nParameter Recovery:")
    print("Parameter  True Value  Recovered Mean  Recovered 95% HPD  Recovered")
    print("-" * 70)
    
    for param in ['a', 'v', 't']:
        if param in stats.index:
            row = stats.loc[param]
            true_val = TRUE_PARAMS[param]
            recov_mean = row['mean']
            hpd_lower = row['2.5q']
            hpd_upper = row['97.5q']
            recovered = "✓" if hpd_lower <= true_val <= hpd_upper else "✗"
            rel_error = abs((recov_mean - true_val) / true_val) * 100
            print(f"{param:9} {true_val:10.4f} {recov_mean:14.4f} [{hpd_lower:5.3f}, {hpd_upper:5.3f}]  {recovered} (Error: {rel_error:.1f}%)")
    
    print("\nCheck the generated files for detailed results.")
    print("If recovery looks good, you can proceed with the full test.")

if __name__ == "__main__":
    run_quick_recovery_test()
