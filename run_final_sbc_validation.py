"""
Final SBC Validation for NES w_n Parameter Recovery using HDDM

This script performs Simulation-Based Calibration (SBC) to validate the recovery
of the w_n parameter in the NES model using HDDM for inference. It implements
the regression-based approach for deriving w_n from drift rates at different
conflict levels.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import traceback
import pickle
from scipy import stats

# Import NES agent and utilities
try:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
except ImportError as e:
    print(f"ERROR: Failed to import NES modules: {e}")
    sys.exit(1)

# Import HDDM
try:
    import hddm
    print(f"Successfully imported HDDM version: {getattr(hddm, '__version__', 'unknown')}")
except ImportError as e:
    print("ERROR: HDDM library not found. Please install it.")
    print(f"Error details: {e}")
    sys.exit(1)

# --- Configuration ---
DEFAULT_N_SBC_ITERATIONS = 20  # Increased to 20 for final validation
DEFAULT_SEED = 42

# Simulation Parameters
N_SUBJECTS = 20  # Increased for better hierarchical shrinkage
N_TRIALS_PER_SUB = 600  # Increased for more stable estimates

# Fixed NES Parameters for Data Generation
TRUE_NES_A = 1.5
TRUE_NES_W_S = 0.7

# Simulation parameters (must match HDDM's expected scaling)
BASE_SIM_PARAMS = {
    't': T_NONDECISION,
    'noise_std_dev': 1.0,  # Critical: must match HDDM's sigma=1 assumption
    'dt': DT,
    'max_time': MAX_TIME,
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
}

# Use only 3 conflict levels as requested
CONFLICT_LEVELS = np.array([0.0, 0.25, 0.5])
# Adjust proportions to be uniform across the three levels
CONFLICT_PROPORTIONS = np.ones(len(CONFLICT_LEVELS)) / len(CONFLICT_LEVELS)

# HDDM Sampling Parameters
HDDM_SAMPLES = 3000
HDDM_BURN = 1000
HDDM_THIN = 3

# Prior for w_n (uniform between 0.1 and 2.0)
WN_PRIOR_MIN = 0.1
WN_PRIOR_MAX = 2.0

def generate_stroop_trial_inputs(n_trials, conflict_levels, conflict_proportions, seed=None):
    """Generate trial inputs with specified conflict levels."""
    rng = np.random.default_rng(seed)
    n_lvls = len(conflict_levels)
    level_indices = rng.choice(np.arange(n_lvls), size=n_trials, p=conflict_proportions)
    return conflict_levels[level_indices]

def generate_nes_data(true_w_n, n_subjects, n_trials_per_sub, seed=None):
    """Generate synthetic NES data for SBC."""
    all_data = []
    agent = MVNESAgent(config={})
    
    for subj_idx in range(n_subjects):
        subj_seed = seed + subj_idx if seed is not None else None
        rng = np.random.default_rng(subj_seed)
        
        # Generate conflict levels for this subject
        conflict_levels = generate_stroop_trial_inputs(
            n_trials_per_sub, CONFLICT_LEVELS, CONFLICT_PROPORTIONS, seed=subj_seed
        )
        
        for trial_idx, conflict in enumerate(conflict_levels):
            salience_input = 1.0 - conflict
            norm_input = conflict
            
            try:
                trial_result = agent.run_mvnes_trial(
                    salience_input=salience_input,
                    norm_input=norm_input,
                    params={
                        'w_n': true_w_n,
                        'threshold_a': TRUE_NES_A,
                        'w_s': TRUE_NES_W_S,
                        **BASE_SIM_PARAMS
                    }
                )
                
                rt = trial_result.get('rt', np.nan)
                response = trial_result.get('choice', np.nan)
                
                if not (np.isnan(rt) or np.isnan(response) or 
                        rt <= BASE_SIM_PARAMS['t'] or 
                        rt >= BASE_SIM_PARAMS['max_time'] or 
                        response not in [0, 1]):
                    all_data.append({
                        'subj_idx': subj_idx,
                        'rt': rt,
                        'response': int(response),
                        'condition': f"L{conflict:.2f}".replace(".", "_")
                    })
                        
            except Exception as e:
                print(f"Warning: Error in trial {trial_idx} for subject {subj_idx}: {e}")
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def fit_hddm_and_estimate_wn(data, true_wn=None):
    """Fit HDDM model and estimate w_n from drift rates."""
    if data.empty:
        print("No data provided for HDDM fitting.")
        return None, None, {}
    
    # Filter extreme RTs
    data = data[(data['rt'] > 0.1) & (data['rt'] < 10.0)].copy()
    
    # Get unique conditions and sort by conflict level
    conditions = sorted(data['condition'].unique(), 
                       key=lambda x: float(x[1:].replace('_', '.')))
    
    print(f"Fitting HDDM with conditions: {conditions}")
    
    try:
        # Create HDDM model with separate drift rates per condition
        model = hddm.HDDM(data,
                         depends_on={'v': 'condition'},
                         include=['v'],
                         p_outlier=0.05)
        
        # Find starting values
        model.find_starting_values()
        
        # Sample from posterior
        print("Sampling from posterior...")
        model.sample(HDDM_SAMPLES, burn=HDDM_BURN, thin=HDDM_THIN,
                    dbname='hddm_traces.db', db='pickle',
                    progress_bar=True)
        
        # Extract drift rates for each condition
        drift_rates = {}
        for cond in conditions:
            node_name = f'v({cond})'
            if node_name in model.nodes_db.node:
                drift_rates[cond] = model.nodes_db.node[node_name].trace()
        
        if not drift_rates:
            print("No valid drift rate traces found.")
            return None, None, {}
        
        # Calculate mean drift rate for each condition
        cond_means = {}
        cond_levels = []
        drift_means = []
        
        for cond, trace in sorted(drift_rates.items(), 
                                key=lambda x: float(x[0][1:].replace('_', '.'))):
            conflict = float(cond[1:].replace('_', '.'))
            mean_v = np.mean(trace)
            std_v = np.std(trace)
            cond_means[cond] = (mean_v, std_v)
            cond_levels.append(conflict)
            drift_means.append(mean_v)
            print(f"Condition {cond} (λ={conflict:.2f}): v = {mean_v:.3f} ± {std_v:.3f}")
        
        # Perform linear regression to estimate w_n
        wn_posterior = estimate_wn_from_drifts(cond_levels, drift_means)
        
        # Cleanup
        if os.path.exists('hddm_traces.db'):
            os.remove('hddm_traces.db')
            
        return wn_posterior, cond_means, drift_rates
        
    except Exception as e:
        print(f"Error in HDDM fitting: {e}")
        traceback.print_exc()
        if os.path.exists('hddm_traces.db'):
            try: os.remove('hddm_traces.db')
            except: pass
        return None, None, {}

def estimate_wn_from_drifts(conflict_levels, drift_means):
    """Estimate w_n from drift rates at different conflict levels using regression."""
    conflict_levels = np.asarray(conflict_levels, dtype=float)
    drift_means = np.asarray(drift_means, dtype=float)
    
    # Add intercept term
    X = np.column_stack((np.ones_like(conflict_levels), conflict_levels))
    
    try:
        # Fit linear model
        beta, residuals_ss, rank, _ = np.linalg.lstsq(X, drift_means, rcond=None)
        
        # Calculate standard errors
        if len(drift_means) > 2:
            mse = residuals_ss[0] / (len(drift_means) - 2) if len(residuals_ss) > 0 else 1.0
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta_se = np.sqrt(np.diag(XtX_inv) * mse)
            
            # Estimate w_n = -(beta0 + beta1)
            wn_mean = -(beta[0] + beta[1])
            
            # Calculate standard error of the sum
            var_wn = max(beta_se[0]**2 + beta_se[1]**2 + 2 * XtX_inv[0,1] * mse, 1e-10)
            wn_std = np.sqrt(var_wn)
            
            # Generate samples from the posterior
            n_samples = 1000
            wn_posterior = np.random.normal(wn_mean, wn_std, n_samples)
            
            return wn_posterior
            
    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def calculate_rank(posterior_samples, true_value):
    """Calculate rank of true value in posterior samples."""
    if posterior_samples is None or len(posterior_samples) == 0:
        return np.nan
    return np.sum(posterior_samples < true_value)

def plot_sbc_histogram(ranks, n_bins=20, parameter_name="w_n", params=None):
    """Plot SBC rank histogram."""
    valid_ranks = np.array([r for r in ranks if not np.isnan(r)])
    if len(valid_ranks) == 0:
        print("No valid ranks to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(valid_ranks, bins=n_bins, 
                               alpha=0.7, color='teal',
                               edgecolor='black')
    
    # Add uniform reference line
    expected = len(valid_ranks) / n_bins
    plt.axhline(y=expected, color='red', linestyle='--', 
                label=f'Expected ({expected:.1f})')
    
    # Add labels and title
    plt.xlabel(f'Rank of True {parameter_name}')
    plt.ylabel('Count')
    
    title = f'SBC Rank Histogram for {parameter_name}\n'
    if params:
        title += f"Subjects: {params.get('N_SUBJECTS', '?')}, "
        title += f"Trials/sub: {params.get('N_TRIALS_PER_SUB', '?')}, "
        title += f"HDDM: {params.get('HDDM_SAMPLES', '?')}s/{params.get('HDDM_BURN', '?')}b\n"
        title += f"w_s: {params.get('TRUE_NES_W_S', '?')}, "
        title += f"a: {params.get('TRUE_NES_A', '?')}, "
        title += f"w_n: [{params.get('WN_PRIOR_MIN', '?')}-{params.get('WN_PRIOR_MAX', '?')}]"
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plots_dir = Path("sbc_results")
    plots_dir.mkdir(exist_ok=True)
    
    param_str = f"_subs{params.get('N_SUBJECTS', '?')}_trials{params.get('N_TRIALS_PER_SUB', '?')}"
    param_str += f"_s{params.get('HDDM_SAMPLES', '?')}b{params.get('HDDM_BURN', '?')}"
    param_str += f"_ws{str(params.get('TRUE_NES_W_S', '?')).replace('.', '')}"
    param_str += f"_a{str(params.get('TRUE_NES_A', '?')).replace('.', '')}"
    param_str += f"_wn{str(params.get('WN_PRIOR_MIN', '?')).replace('.', '')}-{str(params.get('WN_PRIOR_MAX', '?')).replace('.', '')}"
    
    filename = plots_dir / f"sbc_hist_{parameter_name}_{timestamp}{param_str}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SBC histogram to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Run final SBC validation for w_n recovery using HDDM.')
    parser.add_argument('--iterations', type=int, default=DEFAULT_N_SBC_ITERATIONS,
                       help=f'Number of SBC iterations (default: {DEFAULT_N_SBC_ITERATIONS})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed (default: {DEFAULT_SEED})')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print("="*60)
    print(f"Running Final SBC Validation for w_n (N={args.iterations})")
    print(f"Fixed Parameters: w_s={TRUE_NES_W_S}, a={TRUE_NES_A}")
    print(f"Conflict Levels: {CONFLICT_LEVELS}")
    print(f"HDDM Settings: Samples={HDDM_SAMPLES}, Burn={HDDM_BURN}, Thin={HDDM_THIN}")
    print(f"Prior for w_n: Uniform({WN_PRIOR_MIN}, {WN_PRIOR_MAX})")
    print("="*60)
    
    # Prepare results storage
    results = []
    
    for i in range(args.iterations):
        print(f"\n--- Iteration {i+1}/{args.iterations} ---")
        
        # 1. Draw true w_n from prior
        true_wn = np.random.uniform(WN_PRIOR_MIN, WN_PRIOR_MAX)
        print(f"True w_n: {true_wn:.4f}")
        
        # 2. Generate synthetic data
        print("Generating synthetic data...")
        data = generate_nes_data(true_wn, N_SUBJECTS, N_TRIALS_PER_SUB, 
                               seed=args.seed + i + 1)
        
        if data.empty or len(data) < N_SUBJECTS * N_TRIALS_PER_SUB * 0.1:
            print("Warning: Insufficient valid trials. Skipping iteration.")
            results.append({
                'iteration': i,
                'true_wn': true_wn,
                'rank': np.nan,
                'posterior_mean': np.nan,
                'n_samples': 0
            })
            continue
        
        # 3. Fit HDDM and estimate w_n
        print("Fitting HDDM and estimating w_n...")
        wn_posterior, cond_means, _ = fit_hddm_and_estimate_wn(data, true_wn)
        
        # 4. Calculate rank
        rank = np.nan
        posterior_mean = np.nan
        n_samples = 0
        
        if wn_posterior is not None and len(wn_posterior) > 0:
            rank = calculate_rank(wn_posterior, true_wn)
            posterior_mean = np.mean(wn_posterior)
            n_samples = len(wn_posterior)
            print(f"Estimated w_n: {posterior_mean:.4f}, "
                  f"True w_n: {true_wn:.4f}, "
                  f"Rank: {rank}/{n_samples}")
        
        # Store results
        results.append({
            'iteration': i,
            'true_wn': true_wn,
            'rank': rank,
            'posterior_mean': posterior_mean,
            'n_samples': n_samples
        })
        
        # Save intermediate results
        save_results(results, args.seed)
    
    # Final analysis and plotting
    analyze_results(results, args.seed)

def save_results(results, seed):
    """Save current results to CSV."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = Path("sbc_results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename with parameters
    param_str = f"_subs{N_SUBJECTS}_trials{N_TRIALS_PER_SUB}"
    param_str += f"_s{HDDM_SAMPLES}b{HDDM_BURN}"
    param_str += f"_ws{str(TRUE_NES_W_S).replace('.', '')}"
    param_str += f"_a{str(TRUE_NES_A).replace('.', '')}"
    param_str += f"_wn{str(WN_PRIOR_MIN).replace('.', '')}-{str(WN_PRIOR_MAX).replace('.', '')}"
    
    filename = results_dir / f"sbc_results{param_str}_seed{seed}.csv"
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")

def analyze_results(results, seed):
    """Analyze and plot SBC results."""
    if not results:
        print("No results to analyze.")
        return
    
    df = pd.DataFrame(results)
    valid_results = df[df['rank'].notna()]
    
    print("\n" + "="*60)
    print("SBC Results Summary:")
    print("-"*60)
    print(f"Total iterations: {len(df)}")
    print(f"Valid iterations: {len(valid_results)}")
    
    if len(valid_results) > 0:
        print("\nRank Statistics:")
        print(f"  Min rank: {valid_results['rank'].min()}")
        print(f"  Max rank: {valid_results['rank'].max()}")
        print(f"  Mean rank: {valid_results['rank'].mean():.1f}")
        
        # Create parameter dict for plotting
        params = {
            'N_SUBJECTS': N_SUBJECTS,
            'N_TRIALS_PER_SUB': N_TRIALS_PER_SUB,
            'HDDM_SAMPLES': HDDM_SAMPLES,
            'HDDM_BURN': HDDM_BURN,
            'TRUE_NES_W_S': TRUE_NES_W_S,
            'TRUE_NES_A': TRUE_NES_A,
            'WN_PRIOR_MIN': WN_PRIOR_MIN,
            'WN_PRIOR_MAX': WN_PRIOR_MAX
        }
        
        # Plot histogram
        plot_sbc_histogram(valid_results['rank'].tolist(), 
                          parameter_name="w_n", 
                          params=params)
    
    print("\n" + "="*60)
    print("SBC validation complete.")
    print("="*60)

if __name__ == '__main__':
    main()
