<<<<<<< HEAD
import sys
sys.path.append('src')  # Add src directory to Python path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from agent_mvnes import MVNESAgent
from agent_config import VETO_FLAG
import pyabc
from pyabc import Distribution, RV

# Constants
GLOBAL_SEED = 42
TRUE_A = 1.5
TRUE_W_S = 0.5
BASE_SIM_PARAMS = {
    't': 0.3,
    'noise_std_dev': 0.1,
    'dt': 0.01,
    'max_time': 5.0
}
WN_GRID = np.linspace(0.1, 1.0, 10)  # Test w_n values from 0.1 to 1.0
N_TRIALS_PER_WN = 200
P_HIGH_CONFLICT = 0.5  # 50% high-conflict trials
ABC_POPULATION_SIZE = 100
ABC_MAX_NR_POPULATIONS = 3
ABC_MIN_EPSILON = 0.1

# Function to generate trial inputs
def generate_trial_inputs(n_trials, p_high_conflict):
    """
    Generate trial inputs with specified proportion of high-conflict trials.
    
    Args:
        n_trials (int): Number of trials to generate
        p_high_conflict (float): Proportion of high-conflict trials (0 to 1)
        
    Returns:
        tuple: (salience_inputs, norm_inputs, trial_types)
    """
    np.random.seed(GLOBAL_SEED)
    
    # Generate trial types (0 = low conflict, 1 = high conflict)
    trial_types = np.random.binomial(1, p_high_conflict, n_trials)
    
    # Generate inputs based on trial type
    salience_inputs = np.zeros(n_trials)
    norm_inputs = np.zeros(n_trials)
    
    for i in range(n_trials):
        if trial_types[i] == 1:  # High conflict
            # Generate inputs that are more likely to conflict
            salience_inputs[i] = np.random.normal(0.5, 0.1)
            norm_inputs[i] = np.random.normal(0.5, 0.1)
        else:  # Low conflict
            # Generate inputs that are less likely to conflict
            if np.random.rand() < 0.5:  # Both positive
                salience_inputs[i] = np.random.normal(0.7, 0.1)
                norm_inputs[i] = np.random.normal(0.7, 0.1)
            else:  # Both negative
                salience_inputs[i] = np.random.normal(0.3, 0.1)
                norm_inputs[i] = np.random.normal(0.3, 0.1)
    
    return salience_inputs, norm_inputs, trial_types

# Function to simulate trials
def simulate_trials(params, salience_inputs, norm_inputs):
    """
    Simulate trials using the MVNES agent with given parameters.
    
    Args:
        params (dict): Model parameters
        salience_inputs (np.array): Salience inputs for each trial
        norm_inputs (np.array): Norm inputs for each trial
        
    Returns:
        pd.DataFrame: DataFrame with trial results
    """
    n_trials = len(salience_inputs)
    rts = np.zeros(n_trials)
    choices = np.zeros(n_trials)
    
    agent = MVNESAgent(config={})
    for i in range(n_trials):
        result = agent.run_mvnes_trial(
            salience_input=salience_inputs[i],
            norm_input=norm_inputs[i],
            params=params
        )
        rts[i], choices[i] = result['rt'], result['choice']
    
    return pd.DataFrame({
        'rt': rts,
        'choice': choices
    })

# Function to calculate summary statistics
def calculate_summary_stats(df):
    """
    Calculate summary statistics from trial data.
    
    Args:
        df (pd.DataFrame): DataFrame with trial data (rt, choice)
        
    Returns:
        dict: Dictionary of summary statistics
    """
    # Handle empty or NaN data
    if df.empty or df.isnull().values.any():
        return {
            'rt_mean': np.nan,
            'rt_var': np.nan,
            'choice_rate': np.nan,
            'choice_1_rt_mean': np.nan,
            'choice_1_rt_var': np.nan,
            'choice_1_rt_median': np.nan,
            'choice_0_rt_mean': np.nan,
            'choice_0_rt_var': np.nan,
            'choice_0_rt_median': np.nan
        }
    
    rts = df['rt'].values
    choices = df['choice'].values
    
    # Separate RTs by choice
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    
    # Calculate overall statistics
    overall_stats = {
        'rt_mean': np.mean(rts),
        'rt_var': np.var(rts),
        'choice_rate': np.mean(choices)
    }
    
    # Calculate choice-specific statistics
    choice_stats = {}
    for choice, rts in [(1, choice_1_rts), (0, choice_0_rts)]:
        if len(rts) > 0:
            choice_stats.update({
                f'choice_{choice}_rt_mean': np.mean(rts),
                f'choice_{choice}_rt_var': np.var(rts),
                f'choice_{choice}_rt_median': np.median(rts)
            })
        else:
            choice_stats.update({
                f'choice_{choice}_rt_mean': np.nan,
                f'choice_{choice}_rt_var': np.nan,
                f'choice_{choice}_rt_median': np.nan
            })
    
    return {**overall_stats, **choice_stats}

# Function to run ABC-SMC
def run_abc_for_wn(true_wn, observed_summaries, fixed_params, task_params, seed):
    """
    Run ABC-SMC for parameter recovery of w_n.
    
    Args:
        true_wn (float): True value of w_n
        observed_summaries (dict): Observed summary statistics
        fixed_params (dict): Fixed parameters for simulation
        task_params (dict): Task parameters (inputs)
        seed (int): Random seed for reproducibility
        
    Returns:
        pyabc.History: History object containing ABC results
    """
    np.random.seed(seed)
    
    # Define simulator function
    def simulator(params):
        current_params = {**fixed_params, 'w_n': params['w_n']}
        df_sim = simulate_trials(current_params, **task_params)
        return calculate_summary_stats(df_sim)
    
    # Define distance function
    def distance(x, x_0):
        """
        Calculate distance between two sets of summary statistics, ignoring NaNs.
        
        Args:
            x (dict): Simulated summary statistics
            x_0 (dict): Observed summary statistics
            
        Returns:
            float: Distance between x and x_0
        """
        # Only consider keys that exist in both dictionaries
        common_keys = set(x.keys()).intersection(x_0.keys())
        
        dist = 0
        n_valid = 0
        
        for key in common_keys:
            if np.isnan(x_0[key]) or np.isnan(x[key]):
                continue
            dist += (x[key] - x_0[key])**2
            n_valid += 1
        
        if n_valid == 0:
            return np.inf  # Return infinity if no valid comparisons
            
        return np.sqrt(dist / n_valid)  # Normalize by number of valid comparisons
    
    # Define prior distribution with more focused range
    prior = Distribution(
        w_n=RV('uniform', 0, 1.5)  # Adjusted range to better match true values
    )
    
    # Initialize ABC-SMC
    abc = pyabc.ABCSMC(
        models=simulator,
        parameter_priors=prior,
        distance_function=distance,
        population_size=ABC_POPULATION_SIZE
    )
    
    # Initialize database
    db_path = f"sqlite:///abc_wn_recovery_{true_wn:.2f}.db"
    abc.new(db_path, observed_summaries)
    
    # Run ABC-SMC
    history = abc.run(
        max_nr_populations=ABC_MAX_NR_POPULATIONS,
        minimum_epsilon=ABC_MIN_EPSILON
    )
    
    return history

# Function to calculate SBC rank
def calculate_sbc_rank(samples, true_value):
    """
    Calculate rank of true value in the posterior samples.
    
    Args:
        samples (np.array): Posterior samples
        true_value (float): True value of the parameter
        
    Returns:
        int: Rank of the true value in the samples
    """
    if len(samples) == 0:
        return np.nan
    
    # Sort samples
    sorted_samples = np.sort(samples)
    # Find rank of true value
    rank = np.searchsorted(sorted_samples, true_value)
    # Normalize to [0, 1] range
    normalized_rank = rank / len(samples)
    return normalized_rank

print("="*60)
print("Starting W_n Recovery Validation Script")
print(f"Global Seed: {GLOBAL_SEED}")
print(f"Fixed Parameters: a={TRUE_A}, w_s={TRUE_W_S}")
print(f"Base Sim Params: {BASE_SIM_PARAMS}")
print(f"Testing w_n grid: {WN_GRID}")
print(f"Trials per w_n value: {N_TRIALS_PER_WN}")
print(f"Task: {P_HIGH_CONFLICT*100}% High-Conflict Trials")
print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Pops={ABC_MAX_NR_POPULATIONS}, Min Eps={ABC_MIN_EPSILON}")
print("="*60)

# Store results
recovery_results = []

# Generate the fixed trial inputs ONCE
print("Generating fixed trial inputs...")
salience_inputs, norm_inputs, trial_types = generate_trial_inputs(N_TRIALS_PER_WN, P_HIGH_CONFLICT)
task_params = {
    'salience_inputs': salience_inputs,
    'norm_inputs': norm_inputs
    # Could add trial_types if needed for analysis later
}
print(f"Generated {len(salience_inputs)} trials inputs.")

# --- Loop through each true w_n value ---
for i, true_wn in enumerate(WN_GRID):
    print("\n" + "-" * 50)
    print(f"Processing Run {i+1}/{len(WN_GRID)}: True w_n = {true_wn:.3f}")
    run_seed = GLOBAL_SEED + i # Vary seed slightly per run for ABC robustness
    start_time_wn = time.time()

    # 1. GENERATE SYNTHETIC "OBSERVED" DATA for this true_wn
    print("  Generating observed data...")
    current_true_params = {
        'w_n': true_wn,
        'threshold_a': TRUE_A,
        'w_s': TRUE_W_S,
        't': BASE_SIM_PARAMS['t'],
        'noise_std_dev': BASE_SIM_PARAMS['noise_std_dev'],
        'dt': BASE_SIM_PARAMS['dt'],
        'max_time': BASE_SIM_PARAMS['max_time']
    }
    df_obs = simulate_trials(current_true_params, salience_inputs, norm_inputs)
    print(f"  Generated {len(df_obs)} trials. Observed Choice Rate: {df_obs['choice'].mean():.3f}")

    # 2. CALCULATE OBSERVED SUMMARY STATISTICS
    print("  Calculating observed summary statistics...")
    observed_summaries = calculate_summary_stats(df_obs)
    if any(np.isnan(v) for v in observed_summaries.values()):
         print("  WARNING: NaNs detected in observed summary statistics! Check simulation output.")
         # Decide whether to skip ABC or proceed with NaN handling in distance
         # continue # Option to skip if stats are bad

    # 3. RUN ABC-SMC
    fixed_params_for_abc = {
        'threshold_a': TRUE_A,
        'w_s': TRUE_W_S,
        't': BASE_SIM_PARAMS['t'],
        'noise_std_dev': BASE_SIM_PARAMS['noise_std_dev'],
        'dt': BASE_SIM_PARAMS['dt'],
        'max_time': BASE_SIM_PARAMS['max_time']
    }
    history = run_abc_for_wn(true_wn, observed_summaries, fixed_params_for_abc, task_params, seed=run_seed)

    # 4. EXTRACT AND STORE RESULTS
    print("  Extracting results...")
    run_result = {'true_wn': true_wn}
    try:
        # Get weighted posterior distribution from the last generation
        df_posterior, w = history.get_distribution(m=0, t=history.max_t)
        if not df_posterior.empty:
            # Ensure weights sum to 1 for weighted stats
            w = np.array(w)
            w_norm = w / np.sum(w)
            
            # Calculate weighted statistics
            w_n_values = df_posterior['w_n'].values
            mean_w = np.sum(w_n_values * w_norm)
            std_w = np.sqrt(np.sum(w_norm * (w_n_values - mean_w)**2))
            
            # Calculate weighted median
            sorted_indices = np.argsort(w_n_values)
            cum_w = np.cumsum(w_norm[sorted_indices])
            median_idx = np.searchsorted(cum_w, 0.5)
            median_w = w_n_values[sorted_indices[median_idx]]

            # Calculate SBC Rank
            # Using unweighted samples for simplicity, acknowledge limitation
            sbc_rank = calculate_sbc_rank(w_n_values, true_wn)

            run_result.update({
                'recovered_mean': mean_w,
                'recovered_median': median_w,
                'recovered_std': std_w,
                'sbc_rank': sbc_rank,
                'n_posterior_samples': len(df_posterior)
            })
            print(f"  Recovered Mean: {mean_w:.3f}, Median: {median_w:.3f}, Std: {std_w:.3f}, SBC Rank: {sbc_rank}")
        else:
            print("  ERROR: Posterior distribution is empty!")
            run_result.update({'recovered_mean': np.nan, 'recovered_median': np.nan,
                               'recovered_std': np.nan, 'sbc_rank': np.nan,
                               'n_posterior_samples': 0})
    except Exception as e:
        print(f"  ERROR extracting results: {str(e)}")
        run_result.update({'recovered_mean': np.nan, 'recovered_median': np.nan,
                           'recovered_std': np.nan, 'sbc_rank': np.nan,
                           'n_posterior_samples': 0})

    recovery_results.append(run_result)
    end_time_wn = time.time()
    print(f"  Finished processing w_n={true_wn:.3f} in {end_time_wn - start_time_wn:.1f} sec.")


# --- 5. FINAL ANALYSIS & PLOTTING ---
print("\n" + "="*60)
print("Finished all runs. Generating final plots...")

# Generate both plots
plot_recovery(recovery_results)
plot_sbc_ranks(recovery_results)

# Function to plot recovery results

def plot_sbc_ranks(recovery_results):
    """
    Plot the SBC rank distribution.
    
    Args:
        recovery_results (list): List of dictionaries containing recovery results
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(recovery_results)
    
    # Filter out NaN ranks
    valid_ranks = results_df['sbc_rank'].dropna()
    
    if len(valid_ranks) > 0:
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot SBC rank distribution
        plt.hist(valid_ranks, bins=20, range=(0, 1), alpha=0.7)
        plt.title('SBC Rank Distribution')
        plt.xlabel('SBC Rank')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Add reference line for uniform distribution
        plt.axhline(len(valid_ranks)/20, color='r', linestyle='--', label='Uniform Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('sbc_rank_distribution.png')
        plt.show()
        plt.close()

# Function to plot recovery results
def plot_recovery(recovery_results):
    """
    Plot the parameter recovery results.
    
    Args:
        recovery_results (list): List of dictionaries containing recovery results
    """
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(recovery_results)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean recovery with error bars
    # Handle NaN values by filtering them out
    valid_idx = ~np.isnan(results_df['recovered_mean']) & ~np.isnan(results_df['recovered_std'])
    ax1.errorbar(results_df['true_wn'][valid_idx], results_df['recovered_mean'][valid_idx], 
                 yerr=results_df['recovered_std'][valid_idx], fmt='o-', label='Recovered Mean')
    ax1.plot(results_df['true_wn'], results_df['true_wn'], 'k--', label='True Value')
    ax1.set_xlabel('True w_n')
    ax1.set_ylabel('Recovered w_n')
    ax1.set_title('Parameter Recovery Results')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: SBC Rank distribution
    ax2.hist(results_df['sbc_rank'], bins=20, range=(0, 1), alpha=0.7)
    ax2.set_xlabel('SBC Rank')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SBC Rank Distribution')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('wn_recovery_results.png')
    plt.show()
    plt.close()

if recovery_results:
    # Convert results to DataFrame for easier handling if needed
    results_df = pd.DataFrame(recovery_results)
    print("\nRecovery Summary:")
    print(results_df[['true_wn', 'recovered_mean', 'recovered_median', 'recovered_std']].round(3))
    
    # Plot recovery results
    plot_recovery(recovery_results)

    # Generate Plots
    plot_sbc_ranks(recovery_results)
else:
    print("No valid recovery results were obtained.")

print("\nValidation script finished.")
print("="*60)
=======
# Filename: validate_wn_recovery.py
# Purpose: Test parameter recovery for w_n under simplified, controlled conditions
#          using ABC-SMC and rich summary statistics, incorporating best practices.

import sys
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import time
import os
import tempfile
from pathlib import Path

# --- 1. Robust Imports & Dependency Checks ---
try:
    # Dynamically add 'src' to path based on script location
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    sys.path.insert(0, str(src_dir))

    from agent_mvnes import MVNESAgent
    # Import necessary fixed parameters (or define defaults)
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2 # Using a more realistic noise value
        DT = 0.01
        MAX_TIME = 2.0

except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    print("Ensure this script is in the project root or adjust paths.")
    print("Make sure 'src/agent_mvnes.py' and 'src/agent_config.py' exist.")
    sys.exit(1)

try:
    import pyabc
    from pyabc import Distribution, RV, ABCSMC
except ImportError:
    print("ERROR: pyabc library not found.")
    print("Please install it: pip install pyabc")
    sys.exit(1)

# --- 2. Configuration & Constants ---

# Reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Simulation Parameters
N_TRIALS_PER_WN = 300   # Number of trials per w_n value
WN_GRID = [0.2, 0.5, 0.8, 1.1, 1.4] # Grid of true w_n values to test
TRUE_A = 1.0            # Fixed threshold value
TRUE_W_S = 0.7          # Fixed salience weight

# Define base simulation parameters (excluding w_n, a, w_s)
BASE_SIM_PARAMS = {
    't': T_NONDECISION,
    'noise_std_dev': NOISE_STD_DEV,
    'dt': DT,
    'max_time': MAX_TIME,
    # Add other necessary params consistently used by run_mvnes_trial
}

# Simplified Task Parameters
P_HIGH_CONFLICT = 0.5   # Probability of a high-conflict trial
# Define inputs for the two trial types
NEUTRAL_SALIENCE = 1.0
NEUTRAL_NORM = 0.0
CONFLICT_SALIENCE = 1.0 # Assume Go drive is present
CONFLICT_NORM = 1.0     # Norm opposes Go drive

# ABC Configuration
# Use smaller pop size for quicker testing, increase for final runs
ABC_POPULATION_SIZE = 150
ABC_MAX_NR_POPULATIONS = 10 # Allow more generations if needed
ABC_MIN_EPSILON = 0.01 # Target epsilon

# --- 3. Core Functions (Modular Structure) ---

def generate_trial_inputs(n_trials, p_conflict):
    """Generates fixed arrays of salience and norm inputs for the simplified task."""
    salience_inputs = np.zeros(n_trials)
    norm_inputs = np.zeros(n_trials)
    trial_types = []
    for i in range(n_trials):
        if np.random.rand() < p_conflict:
            salience_inputs[i] = CONFLICT_SALIENCE
            norm_inputs[i] = CONFLICT_NORM
            trial_types.append("Conflict")
        else:
            salience_inputs[i] = NEUTRAL_SALIENCE
            norm_inputs[i] = NEUTRAL_NORM
            trial_types.append("Neutral")
    return salience_inputs, norm_inputs, trial_types

def simulate_trials(params_dict, salience_inputs, norm_inputs):
    """
    Simulates N trials for a given parameter set using fixed inputs.
    Returns a DataFrame of trial results.
    """
    n_sim_trials = len(salience_inputs)
    results_list = []
    # Ensure agent is instantiated fresh or properly reset if stateful
    agent = MVNESAgent(config={}) # Assuming stateless for run_mvnes_trial

    for i in range(n_sim_trials):
        # run_mvnes_trial expects 'threshold_a' key
        params_for_agent = {
            'w_n': params_dict['w_n'],
            'threshold_a': params_dict['a'],
            'w_s': params_dict['w_s'],
            **BASE_SIM_PARAMS
        }
        trial_result = agent.run_mvnes_trial(
            salience_input=salience_inputs[i],
            norm_input=norm_inputs[i],
            params=params_for_agent
        )
        results_list.append({
            'rt': trial_result['rt'],
            'choice': trial_result['choice']
            # Add trial type if needed for conditional stats later
        })
    return pd.DataFrame(results_list)

def calculate_summary_stats(df_results):
    """
    Calculates a rich set of summary statistics from trial data (DataFrame).
    Handles cases with no trials or only one choice type robustly.
    """
    summaries = {}
    n_sim_trials = len(df_results)

    if n_sim_trials == 0: # Handle empty dataframe
        # Return dict with NaN for all expected keys
        keys = get_summary_stat_keys() # Need helper function defining keys
        return {k: np.nan for k in keys}

    rts = df_results['rt'].values
    choices = df_results['choice'].values

    # Separate RTs by choice
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)

    # Add choice counts and overall rate
    summaries["n_choice_1"] = n_choice_1
    summaries["n_choice_0"] = n_choice_0
    summaries["choice_rate"] = n_choice_1 / n_sim_trials if n_sim_trials > 0 else np.nan

    # Define statistics calculation functions (with NaN handling)
    def safe_stat(data, func, min_len=1, check_std=False):
        if len(data) < min_len: return np.nan
        if check_std and np.std(data) == 0: return np.nan # Avoid division by zero for skew
        try: return func(data)
        except Exception: return np.nan

    stat_funcs = {
        "rt_mean": partial(safe_stat, func=np.mean),
        "rt_median": partial(safe_stat, func=np.median),
        "rt_var": partial(safe_stat, func=np.var),
        "rt_skew": partial(safe_stat, func=lambda x: np.mean((x - np.mean(x))**3) / np.std(x)**3, min_len=3, check_std=True),
        "rt_q10": partial(safe_stat, func=lambda x: np.nanpercentile(x, 10)),
        "rt_q30": partial(safe_stat, func=lambda x: np.nanpercentile(x, 30)),
        "rt_q50": partial(safe_stat, func=lambda x: np.nanpercentile(x, 50)),
        "rt_q70": partial(safe_stat, func=lambda x: np.nanpercentile(x, 70)),
        "rt_q90": partial(safe_stat, func=lambda x: np.nanpercentile(x, 90)),
        "rt_min": partial(safe_stat, func=np.min),
        "rt_max": partial(safe_stat, func=np.max),
        "rt_range": partial(safe_stat, func=lambda x: np.max(x) - np.min(x))
    }

    # Calculate overall stats
    for name, func in stat_funcs.items():
        summaries[name] = func(rts)

    # Calculate choice=1 stats
    for name, func in stat_funcs.items():
        summaries[f"choice_1_{name}"] = func(choice_1_rts)

    # Calculate choice=0 stats
    for name, func in stat_funcs.items():
        summaries[f"choice_0_{name}"] = func(choice_0_rts)

    # Add RT histogram bins
    try:
        if len(rts) > 0:
            rt_min_val, rt_max_val = np.min(rts), np.max(rts)
            hist_range = (rt_min_val, rt_max_val) if rt_max_val > rt_min_val else (rt_min_val - 0.1, rt_max_val + 0.1)
            hist, _ = np.histogram(rts, bins=10, range=hist_range, density=True)
            summaries.update({f"rt_bin_{i}": hist[i] for i in range(10)})
        else:
            summaries.update({f"rt_bin_{i}": np.nan for i in range(10)})
    except Exception:
        summaries.update({f"rt_bin_{i}": np.nan for i in range(10)})

    return summaries

def get_summary_stat_keys():
    """Helper function to define the expected keys in summary dicts."""
    # Define based on the keys created in calculate_summary_stats
    keys = ["n_choice_1", "n_choice_0", "choice_rate"]
    stat_names = ["rt_mean", "rt_median", "rt_var", "rt_skew", "rt_q10",
                  "rt_q30", "rt_q50", "rt_q70", "rt_q90", "rt_min",
                  "rt_max", "rt_range"]
    keys.extend(stat_names) # Overall
    keys.extend([f"choice_1_{s}" for s in stat_names]) # Choice 1
    keys.extend([f"choice_0_{s}" for s in stat_names]) # Choice 0
    keys.extend([f"rt_bin_{i}" for i in range(10)]) # Histogram
    return keys

def weighted_distance(sim_summary, obs_summary):
    """
    Calculate weighted Euclidean distance between summary statistics dictionaries.
    Handles NaNs robustly by assigning a large penalty.
    """
    # Define weights (adjust these based on sensitivity analysis/importance)
    # Example weights - refine these!
    weights = {
        "choice_rate": 3.0, # Higher weight
        "rt_mean": 1.0, "rt_var": 1.0, "rt_median": 1.0, "rt_skew": 0.5,
        "rt_q10": 1.0, "rt_q30": 1.0, "rt_q50": 1.0, "rt_q70": 1.0, "rt_q90": 1.0,
        "rt_min": 0.2, "rt_max": 0.2, "rt_range": 0.2,

        "choice_1_rt_mean": 1.5, "choice_1_rt_median": 1.5, "choice_1_rt_var": 1.5,
        "choice_1_rt_skew": 0.75, "choice_1_rt_q10": 1.5, "choice_1_rt_q30": 1.5,
        "choice_1_rt_q50": 1.5, "choice_1_rt_q70": 1.5, "choice_1_rt_q90": 1.5,
        "choice_1_rt_min": 0.3, "choice_1_rt_max": 0.3, "choice_1_rt_range": 0.3,

        "choice_0_rt_mean": 1.5, "choice_0_rt_median": 1.5, "choice_0_rt_var": 1.5,
        "choice_0_rt_skew": 0.75, "choice_0_rt_q10": 1.5, "choice_0_rt_q30": 1.5,
        "choice_0_rt_q50": 1.5, "choice_0_rt_q70": 1.5, "choice_0_rt_q90": 1.5,
        "choice_0_rt_min": 0.3, "choice_0_rt_max": 0.3, "choice_0_rt_range": 0.3,

        "n_choice_1": 1.0, "n_choice_0": 1.0,

        **{f"rt_bin_{i}": 0.1 for i in range(10)} # Lower weight for hist bins
    }
    # Define a large penalty for NaN differences
    nan_penalty_squared = 1000.0**2

    total_distance_sq = 0.0
    keys_used = 0

    # Use union of keys to ensure we try to compare all stats
    all_keys = set(sim_summary.keys()) | set(obs_summary.keys())

    for key in all_keys:
        if key in weights: # Only compare stats we have weights for
            obs_val = obs_summary.get(key, np.nan)
            sim_val = sim_summary.get(key, np.nan)

            # Skip if observed is NaN (no target to compare against)
            if np.isnan(obs_val):
                continue

            weight = weights[key]
            if np.isnan(sim_val):
                # Apply large penalty if simulation failed to produce stat
                dist_sq = weight * nan_penalty_squared
            else:
                # Calculate weighted squared difference
                diff = obs_val - sim_val
                dist_sq = weight * (diff**2)

            total_distance_sq += dist_sq
            keys_used += 1

    if keys_used == 0:
        return np.inf

    # Return Euclidean distance
    return np.sqrt(total_distance_sq)


def run_abc_for_wn(true_wn, observed_summaries, fixed_params, task_params, seed):
    """Sets up and runs pyabc.ABCSMC for a single true_wn."""

    print(f"  Setting up ABC for true_wn = {true_wn:.3f}...")

    # Define prior for w_n
    # Prior should be reasonably wide but centered based on expectation
    prior = Distribution(w_n=RV("uniform", 0, 2.0)) # Or RV("norm", 1.0, 0.5) etc.

    # Create simulator function with fixed arguments bound
    simulator_for_abc = partial(simulate_trials_and_summarize, # Use combined function
                                fixed_params=fixed_params,
                                salience_inputs=task_params['salience_inputs'],
                                norm_inputs=task_params['norm_inputs'])

    # Setup ABCSMC
    abc = ABCSMC(
        models=simulator_for_abc,
        parameter_priors=prior,
        distance_function=weighted_distance,
        population_size=ABC_POPULATION_SIZE,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=os.cpu_count()) # Use multiple cores
    )

    # Define database path (unique per run or clear existing)
    temp_dir = tempfile.gettempdir()
    db_file = f"abc_nes_wn_{true_wn:.2f}_seed_{seed}.db"
    db_path = f"sqlite:///{os.path.join(temp_dir, db_file)}"
    if os.path.exists(db_path.replace("sqlite:///", "")):
         os.remove(db_path.replace("sqlite:///", ""))

    # Run ABC
    print(f"  Starting ABC run (DB: {db_path})...")
    abc.new(db_path, observed_summaries)
    history = abc.run(minimum_epsilon=ABC_MIN_EPSILON, max_nr_populations=ABC_MAX_NR_POPULATIONS)
    print("  ABC run finished.")

    return history

def simulate_trials_and_summarize(params_abc, fixed_params, salience_inputs, norm_inputs):
    """Helper function combining simulation and summary calculation for ABC."""
    # Combine parameters correctly
    full_params = {
        'w_n': params_abc['w_n'], # Parameter sampled by ABC
        'a': fixed_params['a'],
        'w_s': fixed_params['w_s'],
        **BASE_SIM_PARAMS
        }
    df_simulated = simulate_trials(full_params, salience_inputs, norm_inputs)
    return calculate_summary_stats(df_simulated)

def calculate_sbc_rank(posterior_samples, true_value):
    """Calculates the rank of the true value within the posterior samples for SBC."""
    # Ensure posterior_samples is a numpy array
    samples = np.asarray(posterior_samples)
    rank = np.sum(samples < true_value)
    # Normalize rank for histogram (optional, raw rank is often used)
    # normalized_rank = (rank + 1) / (len(samples) + 1)
    return rank # Return raw rank for histogramming

def plot_recovery(results_list):
    """Plots true vs. recovered parameter values."""
    if not results_list:
        print("No results to plot for recovery.")
        return

    true_vals = [r['true_wn'] for r in results_list]
    recovered_means = [r['recovered_mean'] for r in results_list]
    recovered_stds = [r['recovered_std'] for r in results_list]

    plt.figure(figsize=(8, 8))
    plt.errorbar(true_vals, recovered_means, yerr=recovered_stds, fmt='o',
                 label='Recovered Mean ± 1SD', capsize=5, alpha=0.7, color='blue')

    min_val = min(true_vals) - 0.2
    max_val = max(true_vals) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity Line')

    plt.xlabel("True w_n Value", fontsize=12)
    plt.ylabel("Recovered w_n Value (Posterior Mean)", fontsize=12)
    plt.title(f"Parameter Recovery Validation for w_n\n(a={TRUE_A}, w_s={TRUE_W_S} fixed, N={N_TRIALS_PER_WN} trials)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def plot_sbc_ranks(results_list, n_posterior_samples):
    """Plots the histogram of SBC ranks."""
    if not results_list:
        print("No results to plot for SBC.")
        return

    ranks = [r['sbc_rank'] for r in recovery_results if 'sbc_rank' in r and not np.isnan(r['sbc_rank'])]
    if not ranks:
        print("No valid SBC ranks found to plot.")
        return

    plt.figure(figsize=(8, 6))
    # Expected number of bins is often sqrt(n_simulations) or related to posterior sample size
    # For visualization, let's use a reasonable number like 20 or 30
    n_bins = min(30, max(10, int(np.sqrt(len(ranks)))))
    plt.hist(ranks, bins=n_bins, density=True, alpha=0.7, color='green', edgecolor='black')

    # Add uniform line for comparison
    # Expected density is 1 / n_posterior_samples if rank is 0 to n_samples-1
    # If rank is 0 to n_samples, expected density is 1 / (n_posterior_samples + 1) - let's use simple rank 0..N-1
    expected_density = 1.0 / n_posterior_samples
    # Need to know the range of ranks (0 to n_samples-1)
    plt.axhline(expected_density, color='red', linestyle='--', label='Uniform (Expected)')

    plt.xlabel("Rank of True Value in Posterior Samples", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(f"Simulation-Based Calibration (SBC) Ranks for w_n (N_posterior={n_posterior_samples})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':

    print("="*60)
    print("Starting W_n Recovery Validation Script")
    print(f"Global Seed: {GLOBAL_SEED}")
    print(f"Fixed Parameters: a={TRUE_A}, w_s={TRUE_W_S}")
    print(f"Base Sim Params: {BASE_SIM_PARAMS}")
    print(f"Testing w_n grid: {WN_GRID}")
    print(f"Trials per w_n value: {N_TRIALS_PER_WN}")
    print(f"Task: {P_HIGH_CONFLICT*100}% High-Conflict Trials")
    print(f"ABC Settings: Pop Size={ABC_POPULATION_SIZE}, Max Pops={ABC_MAX_NR_POPULATIONS}, Min Eps={ABC_MIN_EPSILON}")
    print("="*60)

    # Store results
    recovery_results = []

    # Generate the fixed trial inputs ONCE
    print("Generating fixed trial inputs...")
    salience_inputs, norm_inputs, trial_types = generate_trial_inputs(N_TRIALS_PER_WN, P_HIGH_CONFLICT)
    task_params = {
        'salience_inputs': salience_inputs,
        'norm_inputs': norm_inputs
        # Could add trial_types if needed for analysis later
    }
    print(f"Generated {len(salience_inputs)} trials inputs.")

    # --- Loop through each true w_n value ---
    for i, true_wn in enumerate(WN_GRID):
        print("\n" + "-" * 50)
        print(f"Processing Run {i+1}/{len(WN_GRID)}: True w_n = {true_wn:.3f}")
        run_seed = GLOBAL_SEED + i # Vary seed slightly per run for ABC robustness
        start_time_wn = time.time()

        # 1. GENERATE SYNTHETIC "OBSERVED" DATA for this true_wn
        print("  Generating observed data...")
        current_true_params = {
            'w_n': true_wn,
            'a': TRUE_A,
            'w_s': TRUE_W_S,
            **BASE_SIM_PARAMS
        }
        df_obs = simulate_trials(current_true_params, salience_inputs, norm_inputs)
        print(f"  Generated {len(df_obs)} trials. Observed Choice Rate: {df_obs['choice'].mean():.3f}")

        # 2. CALCULATE OBSERVED SUMMARY STATISTICS
        print("  Calculating observed summary statistics...")
        observed_summaries = calculate_summary_stats(df_obs)
        if any(np.isnan(v) for v in observed_summaries.values()):
             print("  WARNING: NaNs detected in observed summary statistics! Check simulation output.")
             # Decide whether to skip ABC or proceed with NaN handling in distance
             # continue # Option to skip if stats are bad

        # 3. RUN ABC-SMC
        fixed_params_for_abc = {'a': TRUE_A, 'w_s': TRUE_W_S, **BASE_SIM_PARAMS}
        history = run_abc_for_wn(true_wn, observed_summaries, fixed_params_for_abc, task_params, seed=run_seed)

        # 4. EXTRACT AND STORE RESULTS
        print("  Extracting results...")
        run_result = {'true_wn': true_wn}
        try:
            # Get weighted posterior distribution from the last generation
            df_posterior, w = history.get_distribution(m=0, t=history.max_t)
            if not df_posterior.empty:
                # Ensure weights sum to 1 for weighted stats
                w_norm = w / np.sum(w)
                # Calculate weighted mean
                mean_w = np.sum(df_posterior['w_n'] * w_norm)
                # Calculate weighted std dev
                std_w = np.sqrt(np.sum(w_norm * (df_posterior['w_n'] - mean_w)**2))
                # Calculate weighted median more robustly
                sorted_indices = np.argsort(df_posterior['w_n'])
                cum_w = np.cumsum(w_norm[sorted_indices])
                median_idx = np.searchsorted(cum_w, 0.5)
                median_w = df_posterior['w_n'].iloc[sorted_indices[median_idx]]

                # Calculate SBC Rank
                # Using unweighted samples for simplicity, acknowledge limitation
                sbc_rank = calculate_sbc_rank(df_posterior['w_n'].values, true_wn)

                run_result.update({
                    'recovered_mean': mean_w,
                    'recovered_median': median_w,
                    'recovered_std': std_w,
                    'sbc_rank': sbc_rank,
                    'n_posterior_samples': len(df_posterior)
                    # Avoid storing large posterior_df for multiple runs unless needed
                })
                print(f"  Recovered Mean: {mean_w:.3f}, Median: {median_w:.3f}, Std: {std_w:.3f}, SBC Rank: {sbc_rank}")
            else:
                print("  ERROR: Posterior distribution is empty!")
                run_result.update({'recovered_mean': np.nan, 'recovered_median': np.nan,
                                   'recovered_std': np.nan, 'sbc_rank': np.nan,
                                   'n_posterior_samples': 0})
        except Exception as e:
             print(f"  ERROR extracting results: {e}")
             run_result.update({'recovered_mean': np.nan, 'recovered_median': np.nan,
                                'recovered_std': np.nan, 'sbc_rank': np.nan,
                                'n_posterior_samples': 0})

        recovery_results.append(run_result)
        end_time_wn = time.time()
        print(f"  Finished processing w_n={true_wn:.3f} in {end_time_wn - start_time_wn:.1f} sec.")


    # --- 5. FINAL ANALYSIS & PLOTTING ---
    print("\n" + "="*60)
    print("Finished all runs. Generating final plots...")

    if recovery_results:
        # Convert results to DataFrame for easier handling if needed
        results_df = pd.DataFrame(recovery_results)
        print("\nRecovery Summary:")
        print(results_df[['true_wn', 'recovered_mean', 'recovered_median', 'recovered_std']].round(3))

        # Generate Plots
        plot_recovery(recovery_results)
        # Get sample size from first valid run for SBC plot title
        n_samples_for_sbc_plot = next((r['n_posterior_samples'] for r in recovery_results if r['n_posterior_samples'] > 0), ABC_POPULATION_SIZE)
        plot_sbc_ranks(recovery_results, n_samples_for_sbc_plot)
    else:
        print("No valid recovery results were obtained.")

    print("\nValidation script finished.")
    print("="*60)
>>>>>>> 663a63908ed938d2148a0f546587b865295230ca
