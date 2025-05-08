# Filename: explore_nes_parameter_space.py
# Purpose: Systematically explore how changes in NES parameters (w_n, a, w_s)
#          affect observable summary statistics from a simplified Go/No-Go task.
#          This is a forward simulation and sensitivity analysis, not fitting.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from functools import partial # For safe_stat if needed, but directly implemented

# --- 1. Robust Imports & Dependency Checks ---
try:
    # Dynamically add 'src' to path based on script location
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from agent_mvnes import MVNESAgent
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        print("Warning: Could not import agent_config. Using default simulation parameters.")
        T_NONDECISION = 0.1
        NOISE_STD_DEV = 0.2
        DT = 0.01
        MAX_TIME = 2.0
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules: {e}")
    sys.exit(1)

# --- 2. Configuration ---

# Reproducibility
SEED = 123
np.random.seed(SEED)

# Simulation Parameters
N_TRIALS_PER_SETTING = 500 # Increase for more stable summary statistics

# Parameter Grids to Explore
# Define a central value and then sweep around it
WN_GRID = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
A_GRID  = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
WS_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

# --- Default/Fixed Parameters (when not being swept) ---
DEFAULT_WN = 0.8
DEFAULT_A  = 1.0
DEFAULT_W_S = 0.7

BASE_SIM_PARAMS = {
    't': T_NONDECISION,
    'noise_std_dev': NOISE_STD_DEV,
    'dt': DT,
    'max_time': MAX_TIME,
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
}

# Simplified Task Parameters (same as SBC script)
P_HIGH_CONFLICT = 0.5
NEUTRAL_SALIENCE = 1.0
NEUTRAL_NORM = 0.0
CONFLICT_SALIENCE = 1.0
CONFLICT_NORM = 1.0

# --- 3. Helper Functions (from SBC script, slightly adapted) ---

def generate_trial_inputs(n_trials, p_conflict, seed=None):
    """Generates fixed arrays of salience and norm inputs for the simplified task."""
    rng = np.random.default_rng(seed)
    salience_inputs = np.zeros(n_trials)
    norm_inputs = np.zeros(n_trials)
    for i in range(n_trials):
        if rng.random() < p_conflict:
            salience_inputs[i] = CONFLICT_SALIENCE
            norm_inputs[i] = CONFLICT_NORM
        else:
            salience_inputs[i] = NEUTRAL_SALIENCE
            norm_inputs[i] = NEUTRAL_NORM
    return salience_inputs, norm_inputs

def simulate_trials_for_exploration(current_params_dict, salience_inputs, norm_inputs):
    """
    Simulates N trials for a given parameter set using fixed inputs.
    `current_params_dict` should contain 'w_n', 'a', 'w_s' and will be merged with BASE_SIM_PARAMS.
    Returns a DataFrame of trial results.
    """
    n_sim_trials = len(salience_inputs)
    results_list = []
    agent = MVNESAgent(config={})

    params_for_agent = {
        'w_n': current_params_dict['w_n'],
        'threshold_a': current_params_dict['a'], # Map 'a' to 'threshold_a'
        'w_s': current_params_dict['w_s'],
        **BASE_SIM_PARAMS
    }

    for i in range(n_sim_trials):
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_inputs[i],
                norm_input=norm_inputs[i],
                params=params_for_agent
            )
            results_list.append({
                'rt': trial_result.get('rt', np.nan),
                'choice': trial_result.get('choice', np.nan)
            })
        except Exception as e:
            print(f"Warning: Error in run_mvnes_trial (params={params_for_agent}): {e}")
            results_list.append({'rt': np.nan, 'choice': np.nan})
    return pd.DataFrame(results_list)

def get_summary_stat_keys():
    """Helper function to define the expected keys returned by calculate_summary_stats."""
    keys = ["n_choice_1", "n_choice_0", "choice_rate"]
    stat_names = ["rt_mean", "rt_median", "rt_var", "rt_skew", "rt_q10",
                  "rt_q30", "rt_q50", "rt_q70", "rt_q90", "rt_min",
                  "rt_max", "rt_range"]
    keys.extend(stat_names)
    keys.extend([f"choice_1_{s}" for s in stat_names])
    keys.extend([f"choice_0_{s}" for s in stat_names])
    keys.extend([f"rt_bin_{i}" for i in range(10)])
    return keys

def calculate_summary_stats_from_df(df_results):
    """Calculates summary statistics from a DataFrame of trial results."""
    all_keys = get_summary_stat_keys()
    summaries = {k: np.nan for k in all_keys}
    df_results = df_results.dropna(subset=['rt', 'choice'])
    n_sim_trials = len(df_results)

    if n_sim_trials == 0: return summaries

    rts = df_results['rt'].values
    choices = df_results['choice'].values
    choice_1_rts = rts[choices == 1]
    choice_0_rts = rts[choices == 0]
    n_choice_1 = len(choice_1_rts)
    n_choice_0 = len(choice_0_rts)

    summaries["n_choice_1"] = n_choice_1
    summaries["n_choice_0"] = n_choice_0
    summaries["choice_rate"] = n_choice_1 / n_sim_trials if n_sim_trials > 0 else np.nan

    def safe_stat(data, func, min_len=1, check_std=False):
        data = np.asarray(data)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < min_len: return np.nan
        std_val = np.std(valid_data) if len(valid_data) > 0 else 0
        if check_std and (np.isnan(std_val) or std_val == 0): return np.nan
        try:
            nan_func_name = func.__name__
            if not nan_func_name.startswith("nan") and hasattr(np, f"nan{nan_func_name}"):
                nan_func = getattr(np, f"nan{nan_func_name}")
                result = nan_func(data) # Use original data if nan-aware version exists
            elif func.__name__ == "<lambda>" and "nanpercentile" in str(func): # Handle nanpercentile lambdas
                result = func(data)
            else: # Fallback to using valid_data for other functions
                 result = func(valid_data)
            return result if np.isfinite(result) else np.nan
        except Exception: return np.nan

    stat_funcs = {
        "rt_mean": partial(safe_stat, func=np.mean), # Will use nanmean via safe_stat logic
        "rt_median": partial(safe_stat, func=np.median),
        "rt_var": partial(safe_stat, func=np.var),
        "rt_skew": partial(safe_stat, func=lambda x: np.mean(((x - np.nanmean(x))/np.nanstd(x))**3), min_len=3, check_std=True),
        "rt_q10": partial(safe_stat, func=lambda x: np.percentile(x, 10)), # np.nanpercentile for lambdas
        "rt_q30": partial(safe_stat, func=lambda x: np.percentile(x, 30)),
        "rt_q50": partial(safe_stat, func=lambda x: np.percentile(x, 50)),
        "rt_q70": partial(safe_stat, func=lambda x: np.percentile(x, 70)),
        "rt_q90": partial(safe_stat, func=lambda x: np.percentile(x, 90)),
        "rt_min": partial(safe_stat, func=np.min),
        "rt_max": partial(safe_stat, func=np.max),
        "rt_range": partial(safe_stat, func=lambda x: np.max(x) - np.min(x))
    }

    for name, func in stat_funcs.items(): summaries[name] = func(rts)
    for name, func in stat_funcs.items(): summaries[f"choice_1_{name}"] = func(choice_1_rts)
    for name, func in stat_funcs.items(): summaries[f"choice_0_{name}"] = func(choice_0_rts)

    try:
        valid_rts = rts[~np.isnan(rts)]
        if len(valid_rts) > 0:
            rt_min_val, rt_max_val = np.nanmin(valid_rts), np.nanmax(valid_rts)
            if np.isfinite(rt_min_val) and np.isfinite(rt_max_val):
                hist_range = (rt_min_val, rt_max_val) if rt_max_val > rt_min_val else (rt_min_val - 0.1, rt_max_val + 0.1)
                hist, _ = np.histogram(valid_rts, bins=10, range=hist_range, density=True)
                summaries.update({f"rt_bin_{i}": hist[i] for i in range(10)})
    except Exception as e:
        print(f"Warning: Error calculating histogram bins: {e}")
    final_summaries = {k: summaries.get(k, np.nan) for k in all_keys}
    return final_summaries

# --- 4. Main Exploration Logic ---
if __name__ == '__main__':
    print("="*60)
    print("Starting NES Parameter Space Exploration")
    print(f"Global Seed: {SEED}")
    print(f"Trials per setting: {N_TRIALS_PER_SETTING}")
    print(f"Base Fixed Params: a={DEFAULT_A}, w_s={DEFAULT_W_S}, w_n={DEFAULT_WN}")
    print(f"Base Sim Params: {BASE_SIM_PARAMS}")
    print("="*60)

    all_results = []

    # Generate fixed trial inputs ONCE
    print("Generating shared trial inputs...")
    salience_inputs, norm_inputs = generate_trial_inputs(N_TRIALS_PER_SETTING, P_HIGH_CONFLICT, seed=SEED)
    print(f"Generated {len(salience_inputs)} trial inputs.")

    # --- Sweep w_n (a, w_s fixed) ---
    print("\nSweeping w_n...")
    for wn_val in WN_GRID:
        current_params = {'w_n': wn_val, 'a': DEFAULT_A, 'w_s': DEFAULT_W_S}
        print(f"  Simulating for w_n={wn_val:.2f} (a={DEFAULT_A:.2f}, w_s={DEFAULT_W_S:.2f})")
        start_t = time.time()
        df_sim = simulate_trials_for_exploration(current_params, salience_inputs, norm_inputs)
        summary = calculate_summary_stats_from_df(df_sim)
        all_results.append({'param_swept': 'w_n', 'w_n': wn_val, 'a': DEFAULT_A, 'w_s': DEFAULT_W_S, **summary})
        print(f"    Done in {time.time() - start_t:.2f}s. Choice Rate: {summary.get('choice_rate', 'NaN'):.3f}")

    # --- Sweep a (w_n, w_s fixed) ---
    print("\nSweeping a (threshold)...")
    for a_val in A_GRID:
        current_params = {'w_n': DEFAULT_WN, 'a': a_val, 'w_s': DEFAULT_W_S}
        print(f"  Simulating for a={a_val:.2f} (w_n={DEFAULT_WN:.2f}, w_s={DEFAULT_W_S:.2f})")
        start_t = time.time()
        df_sim = simulate_trials_for_exploration(current_params, salience_inputs, norm_inputs)
        summary = calculate_summary_stats_from_df(df_sim)
        all_results.append({'param_swept': 'a', 'w_n': DEFAULT_WN, 'a': a_val, 'w_s': DEFAULT_W_S, **summary})
        print(f"    Done in {time.time() - start_t:.2f}s. Choice Rate: {summary.get('choice_rate', 'NaN'):.3f}")

    # --- Sweep w_s (w_n, a fixed) ---
    print("\nSweeping w_s...")
    for ws_val in WS_GRID:
        current_params = {'w_n': DEFAULT_WN, 'a': DEFAULT_A, 'w_s': ws_val}
        print(f"  Simulating for w_s={ws_val:.2f} (w_n={DEFAULT_WN:.2f}, a={DEFAULT_A:.2f})")
        start_t = time.time()
        df_sim = simulate_trials_for_exploration(current_params, salience_inputs, norm_inputs)
        summary = calculate_summary_stats_from_df(df_sim)
        all_results.append({'param_swept': 'w_s', 'w_n': DEFAULT_WN, 'a': DEFAULT_A, 'w_s': ws_val, **summary})
        print(f"    Done in {time.time() - start_t:.2f}s. Choice Rate: {summary.get('choice_rate', 'NaN'):.3f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("nes_param_exploration_results.csv", index=False, float_format='%.4f')
    print("\nFull exploration results saved to nes_param_exploration_results.csv")

    # --- 5. Plotting Selected Results ---
    print("\nGenerating plots...")

    # Key summary statistics to plot
    stats_to_plot = [
        "choice_rate",
        "rt_mean",
        "choice_1_rt_mean", "choice_1_rt_q50", "choice_1_rt_var",
        "choice_0_rt_mean", "choice_0_rt_q50", "choice_0_rt_var",
        "n_choice_1", "n_choice_0"
    ]

    for param_swept, param_grid, default_vals_str in [
        ('w_n', WN_GRID, f'a={DEFAULT_A}, w_s={DEFAULT_W_S}'),
        ('a', A_GRID, f'w_n={DEFAULT_WN}, w_s={DEFAULT_W_S}'),
        ('w_s', WS_GRID, f'w_n={DEFAULT_WN}, a={DEFAULT_A}')
    ]:
        subset_df = results_df[results_df['param_swept'] == param_swept]
        if subset_df.empty:
            print(f"No data to plot for swept parameter: {param_swept}")
            continue

        n_stats = len(stats_to_plot)
        n_cols = 3
        n_rows = (n_stats + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for i, stat_name in enumerate(stats_to_plot):
            if i < len(axes_flat):
                ax = axes_flat[i]
                if stat_name in subset_df.columns:
                    ax.plot(subset_df[param_swept], subset_df[stat_name], marker='o', linestyle='-')
                    ax.set_xlabel(f"Parameter: {param_swept}")
                    ax.set_ylabel(stat_name.replace("_", " ").title())
                    ax.set_title(f"{stat_name.replace('_', ' ').title()} vs. {param_swept}")
                    ax.grid(True, alpha=0.5)
                else:
                    ax.text(0.5, 0.5, f"Stat '{stat_name}' not found", ha='center', va='center')
                    ax.set_title(f"Stat '{stat_name}'")
            else: # Should not happen if n_stats <= n_rows * n_cols
                break
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        fig.suptitle(f"Sensitivity Analysis: Sweeping '{param_swept}' (Fixed: {default_vals_str})", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
        plt.show()

    print("\nParameter space exploration finished.")
    print("="*60)