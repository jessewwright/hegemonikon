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