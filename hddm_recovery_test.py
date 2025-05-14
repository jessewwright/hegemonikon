"""
hddm_recovery_test.py

A diagnostic script to test HDDM's ability to recover known parameters from
generated data using its own simulator.
"""

import numpy as np
import pandas as pd
import hddm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
from IPython.display import display as ipy_display

def display(*args, **kwargs):
    """Wrapper for display function that works in both Jupyter and standard Python"""
    try:
        return ipy_display(*args, **kwargs)
    except NameError:
        # If we're not in IPython, just print
        for arg in args:
            print(arg)

# Set random seed for reproducibility
np.random.seed(123)

# --- 1. Define True Parameters ---
TRUE_PARAMS = {
    'a': 1.5,      # Decision threshold
    'v': 0.5,      # Drift rate
    't': 0.2,      # Non-decision time (seconds)
    'sv': 0.1,     # Inter-trial variability in drift rate
    'sz': 0.1,     # Inter-trial variability in starting point (as proportion of a)
    'st': 0.1,     # Inter-trial variability in non-decision time
    'z': 0.5,      # Starting point (as proportion of a)
}

# --- 2. Generate Synthetic Data ---
print("Generating synthetic data with known parameters...")
n_trials = 1000
n_subjects = 10

# Generate data using HDDM's simulator
print("Generating data with parameters:", TRUE_PARAMS)

# Generate data for each subject and combine
all_data = []
for subj in range(n_subjects):
    # Generate data for one subject
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
model = hddm.HDDM(
    data,
    include={
        'v': True,  # Include drift rate
        'a': True,  # Include boundary separation
        't': True,  # Include non-decision time
        'sv': True, # Include inter-trial variability in drift
        'sz': True, # Include starting point variability
        'st': True, # Include non-decision time variability
        'z': True   # Include starting point bias
    },
    bias=False,  # We're using 'z' for starting point
    p_outlier=0.05
)

# Set reasonable starting values
model.find_starting_values()

# Sample from the posterior
print("Sampling from posterior...")
model.sample(2000, burn=500, thin=2)

# --- 4. Analyze Results ---
print("\nAnalyzing results...")

# Print summary of posterior
print("\nPosterior Summary Statistics:")
stats = model.gen_stats()
display(stats)

# Save stats to CSV
stats.to_csv(os.path.join("hddm_recovery_results", f"recovery_stats_{timestamp}.csv"))

# Plot posterior distributions
fig = plt.figure(figsize=(12, 8))
for i, param in enumerate(['a', 'v', 't', 'sv', 'sz', 'st', 'z']):
    plt.subplot(3, 3, i+1)
    plt.hist(model.nodes_db.node[param].trace(), bins=30, alpha=0.7)
    plt.axvline(TRUE_PARAMS[param], color='red', linestyle='--', label='True')
    plt.title(f"{param} (True: {TRUE_PARAMS[param]:.2f})")
    if i == 0:
        plt.legend()

plt.tight_layout()

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("hddm_recovery_results", exist_ok=True)
plt.savefig(f"hddm_recovery_results/recovery_{timestamp}.png")
print(f"\nResults saved to hddm_recovery_results/recovery_{timestamp}.png")

# Print parameter recovery summary
print("\nParameter Recovery Summary:")
print("Parameter  True Value  Recovered Mean  Recovered 95% HPD  Recovered")
print("-" * 70)

recovery_results = []
for param in ['a', 'v', 't', 'sv', 'sz', 'st', 'z']:
    try:
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
recovery_df.to_csv(os.path.join("hddm_recovery_results", f"recovery_summary_{timestamp}.csv"), index=False)

print("\nDone!")

# Save the full model for later inspection
model.save(os.path.join("hddm_recovery_results", f"model_{timestamp}.pickle"))

print(f"\nResults saved to hddm_recovery_results/")
print(f"- recovery_{timestamp}.png")
print(f"- recovery_stats_{timestamp}.csv")
print(f"- recovery_summary_{timestamp}.csv")
print(f"- model_{timestamp}.pickle")

# Show the plot (if running interactively)
try:
    plt.show()
except:
    print("\nNote: Could not display plot interactively. Check the saved PNG file.")
