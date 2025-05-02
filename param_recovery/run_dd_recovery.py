# --- Parameter Recovery Fitting ---
from scipy.optimize import minimize
from scipy.stats import binom
import numpy as np # Re-import numpy just in case
import pandas as pd # Re-import pandas just in case

# Read synthetic data and true parameters from CSV files
true_params_df = pd.read_csv('true_parameters.csv')
synthetic_data_df = pd.read_csv('synthetic_data.csv')

# Import necessary components from nes_sim_dd.py
import sys
sys.path.append('..')  # Add parent directory to Python path
from nes.nes_sim_dd import Comparator, AssentGate, run_single_dd_trial

# --- Likelihood Function ---
N_SIMS_PER_CONFIG = 500 # Number of simulations to estimate choice probability

def calculate_nll(params_to_fit, subject_data, fixed_params, ss_option, ll_amount, all_delays):
    """
    Calculates the negative log-likelihood of observing subject's choices
    given a set of candidate parameters.

    Args:
        params_to_fit (list/array): [k_discount, base_threshold]
        subject_data (pd.DataFrame): Data for a single subject.
        fixed_params (dict): Dictionary of fixed DDM parameters.
        ss_option (dict): SS option definition.
        ll_amount (float): Amount of the LL option.
        all_delays (list): List of all LL delays tested.

    Returns:
        float: Negative log-likelihood. Returns inf if params are invalid.
    """
    k_candidate, thresh_candidate = params_to_fit

    # --- Parameter Validation & Bounds ---
    # Enforce bounds implicitly or check here
    if k_candidate <= 0 or thresh_candidate <= 0.01: # k must be positive, threshold slightly above 0
        return np.inf # Return infinity for invalid parameters

    # Combine parameters
    current_params = fixed_params.copy()
    current_params['k_discount'] = k_candidate
    current_params['base_threshold'] = thresh_candidate

    # Initialize components with candidate parameters
    # This is inefficient inside the NLL function, ideally initialize once outside minimize loop?
    # For L-BFGS-B this might be okay as it evaluates multiple times per subject anyway.
    try:
        # Need component classes defined in this scope
        comparator = Comparator(
            dt=current_params['dt'],
            noise_std_dev=current_params['noise_std_dev']
        )
        assent_gate = AssentGate(
            base_threshold=current_params['base_threshold']
        )
    except Exception as e:
        # print(f"Warning: Component init failed for params {params_to_fit}: {e}") # Reduce noise
        return np.inf # Failed to initialize

    total_log_likelihood = 0.0

    # Loop through each unique delay condition for this subject
    for delay in all_delays:
        # Get subject's actual choices for this delay
        delay_data = subject_data[subject_data['ll_delay'] == delay]
        if delay_data.empty:
            continue # Skip if subject has no data for this delay (shouldn't happen here)

        n_total = len(delay_data)
        n_ll_chosen_actual = (delay_data['choice'] == 'Choose_LL').sum()

        # Simulate N times with candidate parameters to get p(Choose_LL)
        sim_choices = []
        ll_option = {'amount': ll_amount, 'delay': delay}
        for _ in range(N_SIMS_PER_CONFIG):
            # Need run_single_dd_trial defined in this scope
            result = run_single_dd_trial(
                current_params, ss_option, ll_option
            )
            sim_choices.append(result['choice'])

        # Estimate probability
        n_ll_chosen_sim = sum(c == 'Choose_LL' for c in sim_choices)
        p_ll = n_ll_chosen_sim / N_SIMS_PER_CONFIG

        # Handle edge cases for binomial likelihood
        # Clamp probability slightly away from 0 and 1 to avoid log(0)
        epsilon = 1e-6
        p_ll = np.clip(p_ll, epsilon, 1.0 - epsilon)

        # Calculate binomial log-likelihood for this delay condition
        log_likelihood_delay = binom.logpmf(n_ll_chosen_actual, n_total, p_ll)
        total_log_likelihood += log_likelihood_delay

        # Check for invalid likelihoods
        if np.isnan(total_log_likelihood) or np.isinf(total_log_likelihood):
             # print(f"Warning: Invalid log-likelihood at delay {delay} for params {params_to_fit}. p_ll={p_ll}") # Reduce noise
             return np.inf # Stop early if likelihood becomes invalid

    # Return negative log-likelihood (for minimization)
    return -total_log_likelihood

# --- Optimization Loop ---
recovered_params = []
all_subjects = synthetic_data_df['subject'].unique()[:10]  # Only first 10 subjects for testing

# Define fixed params, options, delays needed by NLL function
fixed_params_for_fit = {
    'noise_std_dev': 0.237, # Assume known noise
    'w_s': 0.392,          # Assume known ws
    'w_n': 0.0,
    'w_u': 0.0,
    'dt': 0.01,
    'max_time': 5.0
}
ss_option_fit = {'amount': 5, 'delay': 0}
ll_amount_fit = 10
ll_delays_fit = sorted(synthetic_data_df['ll_delay'].unique())

# Bounds for optimization [k, threshold]
# k > 0, maybe up to 1? Threshold > 0.1, maybe up to 2?
bounds = [(1e-6, 1.0), (0.1, 2.0)]

print(f"\nStarting parameter recovery for {len(all_subjects)} subjects...")

for subj_id in all_subjects:
    subject_data = synthetic_data_df[synthetic_data_df['subject'] == subj_id].copy()

    # Initial guess (use population mean or random?)
    # Using mean might bias recovery, maybe start slightly off?
    # x0 = [true_k_mean, true_thresh_mean]
    x0 = [0.05, 0.6] # Generic starting point

    # Define objective function for this subject
    objective_func = lambda p: calculate_nll(p, subject_data, fixed_params_for_fit, ss_option_fit, ll_amount_fit, ll_delays_fit)

    # Run optimization
    opt_result = minimize(
        objective_func,
        x0=x0,
        method='L-BFGS-B', # Method that supports bounds
        bounds=bounds,
        options={'maxiter': 100, 'ftol': 1e-6} # Limit iterations
    )

    if opt_result.success:
        recovered_k, recovered_thresh = opt_result.x
        recovered_params.append({
            'subject': subj_id,
            'recovered_k': recovered_k,
            'recovered_threshold': recovered_thresh,
            'nll': opt_result.fun,
            'success': True,
            'message': opt_result.message
        })
    else:
        recovered_params.append({
            'subject': subj_id,
            'recovered_k': np.nan,
            'recovered_threshold': np.nan,
            'nll': np.nan,
            'success': False,
            'message': opt_result.message
        })

    if (subj_id + 1) % 5 == 0:
         print(f"  Finished fitting subject {subj_id + 1}/{len(all_subjects)}")

recovered_params_df = pd.DataFrame(recovered_params)

print("\nParameter recovery finished.")

# --- Comparison ---
print("\n--- Sample Recovered Parameters (First 5 subjects) ---")
print(recovered_params_df.head().round(3))

# Merge true and recovered parameters
comparison_df = pd.merge(true_params_df, recovered_params_df, on='subject')

print("\n--- Recovery Success Rate ---")
success_rate = comparison_df['success'].mean()
print(f"{success_rate*100:.1f}% of fits converged successfully.")

print("\n--- Correlations (for successful fits) ---")
successful_fits = comparison_df[comparison_df['success']]
if not successful_fits.empty:
    corr_k = successful_fits['true_k'].corr(successful_fits['recovered_k'])
    corr_thresh = successful_fits['true_threshold'].corr(successful_fits['recovered_threshold'])
    print(f"Correlation(True K, Recovered K): {corr_k:.3f}")
    print(f"Correlation(True Thresh, Recovered Thresh): {corr_thresh:.3f}")
else:
    print("No successful fits to calculate correlations.")

# --- Plotting ---
import matplotlib.pyplot as plt

print("\n--- Plotting Instructions ---")
print("Generating scatter plots of parameter recovery...")

# Create scatter plots: Recovered vs True for k and threshold
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
if not successful_fits.empty:
    # Plot K
    axes[0].scatter(successful_fits['true_k'], successful_fits['recovered_k'], alpha=0.7)
    axes[0].plot([min(successful_fits['true_k']), max(successful_fits['true_k'])], [min(successful_fits['true_k']), max(successful_fits['true_k'])], 'r--', label='Identity')
    axes[0].set_xlabel("True k")
    axes[0].set_ylabel("Recovered k")
    axes[0].set_title(f"K Recovery (r={corr_k:.3f})")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Threshold
    axes[1].scatter(successful_fits['true_threshold'], successful_fits['recovered_threshold'], alpha=0.7)
    axes[1].plot([min(successful_fits['true_threshold']), max(successful_fits['true_threshold'])], [min(successful_fits['true_threshold']), max(successful_fits['true_threshold'])], 'r--', label='Identity')
    axes[1].set_xlabel("True Threshold")
    axes[1].set_ylabel("Recovered Threshold")
    axes[1].set_title(f"Threshold Recovery (r={corr_thresh:.3f})")
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.savefig('parameter_recovery_plots.png')
plt.close()  # Close the plot to free up memory

print("\nPlots saved to 'parameter_recovery_plots.png'")
