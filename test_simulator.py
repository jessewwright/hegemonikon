import numpy as np
from agent_mvnes import MVNESAgent
from agent_config import VETO_FLAG
import json

# Fixed parameter values
global true_threshold_a, true_w_s, true_params
def drift_function(w_n, salience, norm):
    """Calculate drift rate based on MVNES model parameters.
    Using fixed w_s value."""
    return w_n * salience + true_w_s * norm

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

def test_simulator():
    # Global variables needed by simulate_summary
    global true_threshold_a, true_w_s, true_params
    
    # True parameters
    true_params = {
        'w_n': 1.1,
        'threshold_a': 1.5,
        'w_s': 0.5,
        't': 0.3,
        'noise_std_dev': 0.1,
        'dt': 0.01,
        'max_time': 5.0
    }
    
    # Fixed parameter values
    true_threshold_a = true_params['threshold_a']
    true_w_s = true_params['w_s']
    
    # Generate synthetic trial inputs (using the same as in the main script)
    np.random.seed(42)  # For reproducibility
    n_trials = 50  # Using fewer trials for testing
    
    # Generate random inputs (using the same ranges as in the main script)
    salience_inputs = np.random.normal(0.5, 0.1, n_trials)
    norm_inputs = np.random.normal(0.5, 0.1, n_trials)
    
    # Run simulator
    summaries = simulate_summary({'w_n': true_params['w_n']}, salience_inputs, norm_inputs)
    
    # Print and save results
    print("\nSummary statistics:")
    for key, value in summaries.items():
        print(f"{key}: {value}")
    
    # Save to JSON file
    with open('target_stats.json', 'w') as f:
        json.dump(summaries, f, indent=4)
    print("\nResults saved to target_stats.json")

if __name__ == '__main__':
    test_simulator()
