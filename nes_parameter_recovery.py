# nes_parameter_recovery.py

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from agent_mvnes import MVNESAgent
from agent_config import VETO_FLAG
import matplotlib.pyplot as plt
from functools import partial
import pyabc
from pyabc import Distribution, RV
import tempfile

# CONFIGURATION
# Speed up settings
n_subjects = 0.5  # Fraction of data to use
n_trials   = 50  # Reduced number of trials for faster inference
population_size = 100  # Reduced population size
max_populations = 3  # Reduced number of populations
min_epsilon = 0.1  # Increased minimum epsilon

# Fixed parameter values
true_threshold_a = 1.5  # Fixed threshold value
true_w_s = 0.5  # Fixed norm weight value

# Constants
VETO_FLAG = False

# Fixed parameter values
true_threshold_a = 1.5  # Fixed threshold value
true_w_s = 0.5  # Fixed norm weight value

# Constants
VETO_FLAG = False

def simulate_summary(params, salience_inputs, norm_inputs):
    """
    Given params = {'w_n':…} and fixed trial inputs, simulate N trials and return rich summary statistics.
    Fixed values for a and w_s are used internally.
    
    These summaries capture both the central tendency and the shape of the RT distribution,
    which is important for capturing the effects of norm weight (w_n).
    
    Args:
        params (dict): Model parameters (w_n, a, w_s)
        salience_inputs (np.array): Fixed salience inputs for each trial
        norm_inputs (np.array): Fixed norm inputs for each trial
    """
    # Initialize params dictionary with fixed values
    fixed_params = {
        'threshold_a': true_threshold_a,
        'w_s': true_w_s,
        't': true_params['t'][0],
        'noise_std_dev': true_params['noise_std_dev'][0],
        'dt': true_params['dt'][0],
        'max_time': true_params['max_time'][0]
    }
    
    # Combine fixed parameters with sampled parameter
    params = {**fixed_params, **params}
    
    # Initialize arrays
    rts = np.zeros(len(salience_inputs))
    choices = np.zeros(len(salience_inputs))
    
    # Simulate trials using the same inputs as the observed data
    agent = MVNESAgent(config={})
    for i in range(len(salience_inputs)):
        result = agent.run_mvnes_trial(
            salience_input=salience_inputs[i],
            norm_input=norm_inputs[i],
            params=params
        )
        rt, ch = result['rt'], result['choice']
        rts[i], choices[i] = rt, ch

    rt_min = np.min(rts)
    rt_max = np.max(rts)
    hist, _ = np.histogram(rts, bins=10, range=(rt_min, rt_max), density=True)
    
    # Separate RTs by choice
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)

    # Calculate choice-conditional statistics
    try:
        # Choice=1 statistics (handle empty arrays gracefully)
        choice_1_stats = {}
        if n_choice_1 > 0:
            choice_1_stats = {
                "choice_1_rt_mean": np.mean(choice_1_rts),
                "choice_1_rt_median": np.median(choice_1_rts),
                "choice_1_rt_var": np.var(choice_1_rts),
                "choice_1_rt_skew": np.nan if n_choice_1 < 3 or np.std(choice_1_rts) == 0 else np.mean((choice_1_rts - np.mean(choice_1_rts))**3) / np.std(choice_1_rts)**3,
                
                # Quantiles
                "choice_1_rt_q10": np.nanpercentile(choice_1_rts, 10),
                "choice_1_rt_q30": np.nanpercentile(choice_1_rts, 30),
                "choice_1_rt_q50": np.nanpercentile(choice_1_rts, 50),
                "choice_1_rt_q70": np.nanpercentile(choice_1_rts, 70),
                "choice_1_rt_q90": np.nanpercentile(choice_1_rts, 90),
                
                # Range
                "choice_1_rt_min": np.min(choice_1_rts),
                "choice_1_rt_max": np.max(choice_1_rts),
                "choice_1_rt_range": np.max(choice_1_rts) - np.min(choice_1_rts)
            }
        else:
            # If no choice_1 trials, return NaN for all stats
            choice_1_stats = {
                f"choice_1_{stat}": np.nan for stat in [
                    "rt_mean", "rt_median", "rt_var", "rt_skew",
                    "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                    "rt_min", "rt_max", "rt_range"
                ]
            }
    except Exception as e:
        print(f"Warning: Error calculating choice_1 statistics: {str(e)}")
        choice_1_stats = {
            f"choice_1_{stat}": np.nan for stat in [
                "rt_mean", "rt_median", "rt_var", "rt_skew",
                "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                "rt_min", "rt_max", "rt_range"
            ]
        }
    
    try:
        # Choice=0 statistics (handle empty arrays gracefully)
        choice_0_stats = {}
        if n_choice_0 > 0:
            choice_0_stats = {
                "choice_0_rt_mean": np.mean(choice_0_rts),
                "choice_0_rt_median": np.median(choice_0_rts),
                "choice_0_rt_var": np.var(choice_0_rts),
                "choice_0_rt_skew": np.nan if n_choice_0 < 3 or np.std(choice_0_rts) == 0 else np.mean((choice_0_rts - np.mean(choice_0_rts))**3) / np.std(choice_0_rts)**3,
                
                # Quantiles
                "choice_0_rt_q10": np.nanpercentile(choice_0_rts, 10),
                "choice_0_rt_q30": np.nanpercentile(choice_0_rts, 30),
                "choice_0_rt_q50": np.nanpercentile(choice_0_rts, 50),
                "choice_0_rt_q70": np.nanpercentile(choice_0_rts, 70),
                "choice_0_rt_q90": np.nanpercentile(choice_0_rts, 90),
                
                # Range
                "choice_0_rt_min": np.min(choice_0_rts),
                "choice_0_rt_max": np.max(choice_0_rts),
                "choice_0_rt_range": np.max(choice_0_rts) - np.min(choice_0_rts)
            }
        else:
            # If no choice_0 trials, return NaN for all stats
            choice_0_stats = {
                f"choice_0_{stat}": np.nan for stat in [
                    "rt_mean", "rt_median", "rt_var", "rt_skew",
                    "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                    "rt_min", "rt_max", "rt_range"
                ]
            }
    except Exception as e:
        print(f"Warning: Error calculating choice_0 statistics: {str(e)}")
        choice_0_stats = {
            f"choice_0_{stat}": np.nan for stat in [
                "rt_mean", "rt_median", "rt_var", "rt_skew",
                "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                "rt_min", "rt_max", "rt_range"
            ]
        }

    # Add choice counts and overall statistics
    summaries = {
        "n_choice_1": n_choice_1,
        "n_choice_0": n_choice_0,
        "choice_rate": n_choice_1 / len(choices),
        
        # Overall statistics (for all trials)
        "rt_mean": np.mean(rts),
        "rt_var": np.var(rts),
        "rt_median": np.median(rts),
        "rt_skew": np.nan if len(rts) < 3 or np.std(rts) == 0 else np.mean((rts - np.mean(rts))**3) / np.std(rts)**3,
        "rt_q10": np.nanpercentile(rts, 10),
        "rt_q30": np.nanpercentile(rts, 30),
        "rt_q50": np.nanpercentile(rts, 50),
        "rt_q70": np.nanpercentile(rts, 70),
        "rt_q90": np.nanpercentile(rts, 90),
        "rt_min": np.min(rts),
        "rt_max": np.max(rts),
        "rt_range": np.max(rts) - np.min(rts)
    }

    # Add choice-specific quantiles
    summaries.update(choice_1_stats)
    summaries.update(choice_0_stats)

    # Add RT histogram bins
    summaries.update({f"rt_bin_{i}": hist[i] for i in range(10)})

    return summaries

def drift_function(w_n, salience, norm):
    """Calculate drift rate based on MVNES model parameters.
    Using fixed w_s value."""
    return w_n * salience + true_w_s * norm
# Configuration variables
n_subjects = 1  # Single subject for testing
n_trials   = 200  # Increased number of trials for better inference

if __name__ == '__main__':
    # Define drift function (using fixed w_s)
    def drift_function(w_n, salience, norm):
        """Calculate drift rate based on MVNES model parameters.
        Using fixed w_s value."""
        return w_n * salience + true_w_s * norm

    print("Starting parameter recovery...")
    print(f"Generating data for {n_subjects} subjects with {n_trials} trials each")
    
    # 1. SIMULATION: synthetic data
    # Use fixed values for a and w_s
    true_a = 1.5  # Fixed threshold value
    true_w_s = 0.5  # Fixed norm weight value
    
    # 2) Store true parameters for later comparison
    true_params = {
        'w_n': np.random.normal(1.0, 0.1, n_subjects),  # Mean 1.0, std 0.1
        'threshold_a': np.full(n_subjects, true_threshold_a),  # Fixed threshold
        'w_s': np.full(n_subjects, true_w_s),  # Fixed salience weight
        't': np.full(n_subjects, 0.3),  # Fixed non-decision time
        'noise_std_dev': np.full(n_subjects, 0.1),  # Fixed noise standard deviation
        'dt': np.full(n_subjects, 0.01),  # Fixed time step
        'max_time': np.full(n_subjects, 5.0)  # Fixed maximum trial time
    }
    print("True parameters:")
    print(f"w_n mean: {true_params['w_n'].mean():.2f}, std: {true_params['w_n'].std():.2f}")
    print(f"Using fixed values: threshold_a={true_threshold_a}, w_s={true_w_s}, t={true_params['t'][0]}, noise_std_dev={true_params['noise_std_dev'][0]}, dt={true_params['dt'][0]}, max_time={true_params['max_time'][0]}")

    # Generate fixed trial inputs (same for all subjects)
    np.random.seed(42)  # For reproducibility
    salience_inputs = np.random.rand(n_trials)
    norm_inputs = np.random.rand(n_trials)

    data = []
    for subj in range(n_subjects):
        agent = MVNESAgent(config={})
        for t in range(n_trials):
            trial_data = agent.run_mvnes_trial(
                salience_input=salience_inputs[t],
                norm_input=norm_inputs[t],
                params={
                    'w_s': true_w_s,
                    'w_n': true_params['w_n'][subj],
                    'threshold_a': true_a,
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
                'salience_input': salience_inputs[t],
                'norm_input': norm_inputs[t]
            })

    # Compute observed summaries from the generated data
    df = pd.DataFrame(data)
    
    # Calculate RT statistics
    rts = df['rt'].values
    rt_min = np.min(rts)
    rt_max = np.max(rts)
    hist, _ = np.histogram(rts, bins=10, range=(rt_min, rt_max), density=True)
    
    # Separate RTs by choice
    choice_1_rts = rts[df['choice'] == 1]
    choice_0_rts = rts[df['choice'] == 0]
    
    # Calculate choice counts
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)
    
    # Calculate overall statistics
    obs_summ = {
        "n_choice_1": n_choice_1,
        "n_choice_0": n_choice_0,
        "choice_rate": n_choice_1 / len(rts),
        
        # Overall statistics (for all trials)
        "rt_mean": np.mean(rts),
        "rt_var": np.var(rts),
        "rt_median": np.median(rts),
        "rt_skew": np.nan if len(rts) < 3 or np.std(rts) == 0 else np.mean((rts - np.mean(rts))**3) / np.std(rts)**3,
        "rt_q10": np.nanpercentile(rts, 10),
        "rt_q30": np.nanpercentile(rts, 30),
        "rt_q50": np.nanpercentile(rts, 50),
        "rt_q70": np.nanpercentile(rts, 70),
        "rt_q90": np.nanpercentile(rts, 90),
        "rt_min": np.min(rts),
        "rt_max": np.max(rts),
        "rt_range": np.max(rts) - np.min(rts)
    }
    
    # Calculate choice-conditional statistics
    try:
        # Choice=1 statistics (handle empty arrays gracefully)
        choice_1_stats = {}
        if n_choice_1 > 0:
            choice_1_stats = {
                "choice_1_rt_mean": np.mean(choice_1_rts),
                "choice_1_rt_median": np.median(choice_1_rts),
                "choice_1_rt_var": np.var(choice_1_rts),
                "choice_1_rt_skew": np.nan if n_choice_1 < 3 or np.std(choice_1_rts) == 0 else np.mean((choice_1_rts - np.mean(choice_1_rts))**3) / np.std(choice_1_rts)**3,
                
                # Quantiles
                "choice_1_rt_q10": np.nanpercentile(choice_1_rts, 10),
                "choice_1_rt_q30": np.nanpercentile(choice_1_rts, 30),
                "choice_1_rt_q50": np.nanpercentile(choice_1_rts, 50),
                "choice_1_rt_q70": np.nanpercentile(choice_1_rts, 70),
                "choice_1_rt_q90": np.nanpercentile(choice_1_rts, 90),
                
                # Range
                "choice_1_rt_min": np.min(choice_1_rts),
                "choice_1_rt_max": np.max(choice_1_rts),
                "choice_1_rt_range": np.max(choice_1_rts) - np.min(choice_1_rts)
            }
        else:
            # If no choice_1 trials, return NaN for all stats
            choice_1_stats = {
                f"choice_1_{stat}": np.nan for stat in [
                    "rt_mean", "rt_median", "rt_var", "rt_skew",
                    "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                    "rt_min", "rt_max", "rt_range"
                ]
            }
    except Exception as e:
        print(f"Warning: Error calculating choice_1 statistics: {str(e)}")
        choice_1_stats = {
            f"choice_1_{stat}": np.nan for stat in [
                "rt_mean", "rt_median", "rt_var", "rt_skew",
                "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                "rt_min", "rt_max", "rt_range"
            ]
        }
    
    try:
        # Choice=0 statistics (handle empty arrays gracefully)
        choice_0_stats = {}
        if n_choice_0 > 0:
            choice_0_stats = {
                "choice_0_rt_mean": np.mean(choice_0_rts),
                "choice_0_rt_median": np.median(choice_0_rts),
                "choice_0_rt_var": np.var(choice_0_rts),
                "choice_0_rt_skew": np.nan if n_choice_0 < 3 or np.std(choice_0_rts) == 0 else np.mean((choice_0_rts - np.mean(choice_0_rts))**3) / np.std(choice_0_rts)**3,
                
                # Quantiles
                "choice_0_rt_q10": np.nanpercentile(choice_0_rts, 10),
                "choice_0_rt_q30": np.nanpercentile(choice_0_rts, 30),
                "choice_0_rt_q50": np.nanpercentile(choice_0_rts, 50),
                "choice_0_rt_q70": np.nanpercentile(choice_0_rts, 70),
                "choice_0_rt_q90": np.nanpercentile(choice_0_rts, 90),
                
                # Range
                "choice_0_rt_min": np.min(choice_0_rts),
                "choice_0_rt_max": np.max(choice_0_rts),
                "choice_0_rt_range": np.max(choice_0_rts) - np.min(choice_0_rts)
            }
        else:
            # If no choice_0 trials, return NaN for all stats
            choice_0_stats = {
                f"choice_0_{stat}": np.nan for stat in [
                    "rt_mean", "rt_median", "rt_var", "rt_skew",
                    "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                    "rt_min", "rt_max", "rt_range"
                ]
            }
    except Exception as e:
        print(f"Warning: Error calculating choice_0 statistics: {str(e)}")
        choice_0_stats = {
            f"choice_0_{stat}": np.nan for stat in [
                "rt_mean", "rt_median", "rt_var", "rt_skew",
                "rt_q10", "rt_q30", "rt_q50", "rt_q70", "rt_q90",
                "rt_min", "rt_max", "rt_range"
            ]
        }
    
    # Add histogram bins
    obs_summ.update({f"rt_bin_{i}": hist[i] for i in range(10)})
    
    # Combine all statistics
    obs_summ.update(choice_1_stats)
    obs_summ.update(choice_0_stats)
    
    print("\nObserved summaries:")
    print(f"RT mean: {obs_summ['rt_mean']:.3f}")
    print(f"RT variance: {obs_summ['rt_var']:.3f}")
    print(f"Choice rate: {obs_summ['choice_rate']:.3f}")
    print(f"Choice-1 RT mean: {obs_summ['choice_1_rt_mean']:.3f}")
    print(f"Choice-0 RT mean: {obs_summ['choice_0_rt_mean']:.3f}")

    # 4. ABC-SMC PARAMETER RECOVERY
    print("\nRunning ABC-SMC parameter recovery...")
    import pyabc
    import tempfile
    import pandas as pd

    # Get fixed trial inputs from the data
    df = pd.DataFrame(data)
    salience_inputs = df['salience_input'].values
    norm_inputs = df['norm_input'].values

    # 4a) Define prior over parameters (only w_n)
    prior = pyabc.Distribution(
        w_n=pyabc.RV("norm", 1.0, 0.2)
    )

    # 4b) Create simulator function that uses fixed inputs
    simulator_with_inputs = partial(simulate_summary,
                                              salience_inputs=salience_inputs,
                                              norm_inputs=norm_inputs)

    # 4b) Custom weighted distance function
    def weighted_distance(x, x_0):
        # Define weights for different types of summaries
        weights = {
            # Overall choice rate (most important for w_n)
            "choice_rate": 2.0,
            
            # Overall RT distribution
            "rt_mean": 1.0,
            "rt_var": 1.0,
            "rt_median": 1.0,
            "rt_skew": 1.0,
            "rt_q10": 1.0,
            "rt_q30": 1.0,
            "rt_q50": 1.0,
            "rt_q70": 1.0,
            "rt_q90": 1.0,
            "rt_min": 0.5,
            "rt_max": 0.5,
            "rt_range": 0.5,
            
            # Choice-1 specific statistics (important for w_n and a)
            "choice_1_rt_mean": 1.5,
            "choice_1_rt_median": 1.5,
            "choice_1_rt_var": 1.5,
            "choice_1_rt_skew": 1.5,
            "choice_1_rt_q10": 1.5,
            "choice_1_rt_q30": 1.5,
            "choice_1_rt_q50": 1.5,
            "choice_1_rt_q70": 1.5,
            "choice_1_rt_q90": 1.5,
            "choice_1_rt_min": 0.75,
            "choice_1_rt_max": 0.75,
            "choice_1_rt_range": 0.75,
            
            # Choice-0 specific statistics (important for a)
            "choice_0_rt_mean": 1.5,
            "choice_0_rt_median": 1.5,
            "choice_0_rt_var": 1.5,
            "choice_0_rt_skew": 1.5,
            "choice_0_rt_q10": 1.5,
            "choice_0_rt_q30": 1.5,
            "choice_0_rt_q50": 1.5,
            "choice_0_rt_q70": 1.5,
            "choice_0_rt_q90": 1.5,
            "choice_0_rt_min": 0.75,
            "choice_0_rt_max": 0.75,
            "choice_0_rt_range": 0.75,
            
            # Choice counts (important for w_n)
            "n_choice_1": 1.0,
            "n_choice_0": 1.0,
            
            # RT histogram bins (less important)
            "rt_bin_0": 0.5,
            "rt_bin_1": 0.5,
            "rt_bin_2": 0.5,
            "rt_bin_3": 0.5,
            "rt_bin_4": 0.5,
            "rt_bin_5": 0.5,
            "rt_bin_6": 0.5,
            "rt_bin_7": 0.5,
            "rt_bin_8": 0.5,
            "rt_bin_9": 0.5,
        }
        
        # Ensure both dictionaries have the same keys
        keys = set(x.keys()) & set(x_0.keys())
        
        # Calculate weighted sum of squared differences
        # Handle NaN values appropriately
        distances = []
        for key in keys:
            if key in weights:
                x_val = x[key]
                x0_val = x_0[key]
                
                # Skip if both values are NaN
                if np.isnan(x_val) and np.isnan(x0_val):
                    continue
                
                # If only one value is NaN, treat it as a large distance
                if np.isnan(x_val) or np.isnan(x0_val):
                    distances.append(weights[key] * 1000)  # Large penalty for missing data
                else:
                    distances.append(weights[key] * (x_val - x0_val)**2)
        
        return sum(distances)

    # 4a) Define prior distribution for w_n
    prior = Distribution(
        w_n=RV('uniform', 0, 2)  # w_n between 0 and 2
    )

    # 4c) Create ABC-SMC object with population size using partial
    abc = pyabc.ABCSMC(
        models=simulator_with_inputs,
        parameter_priors=prior,
        distance_function=weighted_distance,
        population_size=population_size
    )

    # 4d) Create an in-memory (or sqlite) database to store results
    temp_dir = tempfile.gettempdir()
    db_path = f"sqlite:///{temp_dir}/abc_nes.db"

    # 4d) Run ABC-SMC for parameter recovery
    abc.new(db_path, obs_summ)
    history = abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_populations)
    
    print("\nABC-SMC parameter recovery completed!")
    print(f"Final epsilon: {history.get_all_populations().iloc[-1].epsilon:.5f}")
    print(f"Number of populations: {len(history.get_all_populations())}")
    # Calculate total number of simulations by summing samples across populations
    total_samples = history.get_all_populations()['samples'].sum()
    print(f"Total number of samples: {total_samples}")
    # Calculate wall time using the population_end_time column
    wall_time = (history.get_all_populations().iloc[-1]['population_end_time'] - 
                 history.get_all_populations().iloc[0]['population_end_time']).total_seconds()
    print(f"Total wall time: {wall_time:.1f} seconds")

    # 4e) Extract and analyze posterior
    df, w = history.get_distribution(m=0)  # m=0 for last population
    
    # 4f) Print ABC posterior summary
    print("\nABC posterior summary:")
    print(df.describe())
    
    # 4g) Save results
    df.to_csv("abc_posterior.csv")
    print("\nResults saved to abc_posterior.csv")

    # 4h) Plot ABC posterior and save to file
    plt.figure(figsize=(8, 6))
    df['w_n'].hist(bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(true_params['w_n'].mean(), color="red", linestyle='--', label='True value')
    plt.title("w_n ABC posterior distribution", fontsize=14)
    plt.xlabel("w_n value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('abc_posterior_plot.png')
    print("\nPlot saved to abc_posterior_plot.png")
