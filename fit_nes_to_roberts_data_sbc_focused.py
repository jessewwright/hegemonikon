# Filename: fit_nes_to_roberts_data_sbc_focused.py
# Purpose: Train an NPE for the NES model on Roberts et al. data
#          and perform Simulation-Based Calibration (SBC).
# This version integrates core logic from the user's previous full script.

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for plotting

import argparse
import logging
import json
import sys
from pathlib import Path
import time
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns # For plotting if needed later, not strictly for SBC plot

# --- SBI Patches for standardization issues ---
from nflows.transforms.standard import IdentityTransform as NflowsIdentityTransform
import torch.nn as nn
import sbi.utils.sbiutils as sbi_utils_module_to_patch

# For standardizing_transform (expects nflows.Transform object)
sbi_utils_module_to_patch.standardizing_transform = lambda batch_x, structured_x=None: NflowsIdentityTransform()

# For standardizing_net (expects nn.Module returning single tensor)
sbi_utils_module_to_patch.standardizing_net = lambda data_for_stats, structured=False: nn.Identity()

# --- SBI Imports ---
try:
    import sbi 
    from sbi.inference import SNPE, simulate_for_sbi
    from sbi.utils import BoxUniform
    from scipy import stats as sp_stats # For KS test if sbc_rank_stats doesn't provide p-value
except ImportError as e:
    logging.error(f"Critical SBI/scipy import error: {e}. Please ensure libraries are installed correctly.")
    sys.exit(1)

# --- Local patch for sbi v0.22.0 SBC ---
def run_sbc(true_parameters, observations, posterior, num_posterior_samples):
    """
    Robust SBC implementation with better error handling and diagnostics.
    Args:
        true_parameters: Tensor of shape (num_datasets, num_parameters)
        observations: Tensor of shape (num_datasets, num_observation_features)
        posterior: SBI posterior object
        num_posterior_samples: Number of posterior samples per observation
    Returns:
        ranks: Tensor of shape (num_datasets, num_parameters)
    """
    import torch
    import logging
    import sys
    
    num_datasets, num_parameters = true_parameters.shape
    # Use -1 as placeholder for invalid/missing ranks (can't use NaN with integers)
    ranks_tensor = torch.full((num_datasets, num_parameters), -1, 
                             dtype=torch.int32, device=true_parameters.device)
    
    success = 0
    fail = 0
    error_messages = {}
    
    # Ensure posterior is not None before proceeding
    if posterior is None:
        logging.error("[SBC run_sbc] Posterior object is None. Cannot proceed.")
        raise ValueError("Posterior object cannot be None for SBC.")

    for i, (theta_i_original, x_i_original) in enumerate(zip(true_parameters, observations)):
        try:
            x_i_for_sampling = x_i_original.unsqueeze(0) # Ensure x_i is [1, num_summary_stats]

            # Logging:
            if i < 3 or i % (num_datasets // 10 or 1) == 0 : # Log first few and some spaced out
                logging.info(f"[SBC Loop {i}] Conditioning on x_i (shape {x_i_for_sampling.shape}): {x_i_for_sampling.flatten()[:5]}...")

            # Sample from posterior
            samples_i = posterior.sample(
                (num_posterior_samples,), 
                x=x_i_for_sampling, # Use the explicitly batched x_i
                show_progress_bars=False
            )
            
            if i < 3 or i % (num_datasets // 10 or 1) == 0 :
                logging.info(f"[SBC Loop {i}] Posterior sample means: {samples_i.mean(dim=0).cpu().numpy()}")

            # Calculate ranks
            # Ensure theta_i_original is comparable shape for broadcasting
            ranks_i = torch.sum(samples_i < theta_i_original.unsqueeze(0), dim=0) 
            
            # Store results
            ranks_tensor[success] = ranks_i
            success += 1
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == num_datasets:
                print(f"\r[SBC] Processed {i+1}/{num_datasets} datasets "
                      f"({success} success, {fail} failed)", 
                      end="", file=sys.stderr)
                
        except Exception as e:
            fail += 1
            error_type = type(e).__name__
            error_messages[error_type] = error_messages.get(error_type, 0) + 1
            if fail <= 5:  # Only show first few errors to avoid spam
                logging.warning(f"[SBC] Failed at simulation {i}: {e}")
            continue
    
    # Trim and validate results
    if success == 0:
        error_summary = ", ".join(f"{k}({v})" for k, v in error_messages.items())
        raise RuntimeError(
            f"SBC failed for all {num_datasets} datasets. Errors: {error_summary}"
        )
    
    if fail > 0:
        error_summary = ", ".join(f"{k}({v})" for k, v in error_messages.items())
        logging.warning(
            f"SBC had {fail}/{num_datasets} failures. Error types: {error_summary}"
        )
    
    # Trim to actual successful runs
    final_ranks = ranks_tensor[:success]

    if final_ranks.numel() == 0:
        logging.warning("[SBC Results] No successful SBC datasets to analyze. Returning empty ranks.")
    return final_ranks

def sbc_rank_plot(ranks, num_posterior_samples):
    import matplotlib.pyplot as plt
    import numpy as np
    num_datasets, num_parameters = ranks.shape
    fig, axes = plt.subplots(1, num_parameters, figsize=(4 * num_parameters, 4), squeeze=False)
    for j in range(num_parameters):
        ax = axes[0, j]
        ax.hist(ranks[:, j].cpu().numpy(), bins=np.arange(num_posterior_samples + 2) - 0.5, density=True)
        ax.set_title(f'Parameter {j}')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# --- Project-Specific Imports ---
script_dir = Path(__file__).resolve().parent
project_root_paths = [script_dir, script_dir.parent, script_dir.parent.parent]
agent_mvnes_found = False
try:
    for prp in project_root_paths:
        potential_src_dir = prp / 'src'
        if (potential_src_dir / 'agent_mvnes.py').exists():
            if str(potential_src_dir) not in sys.path:
                sys.path.insert(0, str(potential_src_dir))
            from agent_mvnes import MVNESAgent # ASSUMING THIS IS YOUR AGENT CLASS
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from: {potential_src_dir}")
            break
    if not agent_mvnes_found:
        # Fallback if 'src' is not found, try current dir if script is moved to project root
        if (Path('.') / 'agent_mvnes.py').exists():
            if str(Path('.')) not in sys.path: sys.path.insert(0, str(Path('.')))
            from agent_mvnes import MVNESAgent
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from current directory.")
        else:
            raise ImportError("Could not find agent_mvnes.py in typical project structures or current directory.")
except ImportError as e:
    logging.error(f"Error importing MVNESAgent: {e}. Check script location and 'src' directory.")
    sys.exit(1)

# --- Global Configurations ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s', force=True)
sbi_logger = logging.getLogger('sbi')
sbi_logger.setLevel(logging.WARNING)

PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain']  # Order must match prior bounds below
PRIOR_LOW = torch.tensor([0.1,  0.5,  0.2,  0.05, 0.5]) # Added 0.5 for alpha_gain
PRIOR_HIGH = torch.tensor([2.0,  2.5,  1.5,  0.7, 1.0]) # Added 1.0 for alpha_gain
BASE_SIM_PARAMS = {
    'noise_std_dev': 1.0, 'dt': 0.01, 'max_time': 10.0, 'veto_flag': False
}
CONDITIONS_ROBERTS = {
    'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
    'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
}

SUBJECT_TRIAL_STRUCTURE_TEMPLATE: Optional[pd.DataFrame] = None
# OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE is less critical for pure SBC sim if simulator is robust
# but calculate_summary_stats might use it if defined.
OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE: Optional[Dict[str, float]] = None


# --- Core Functions from User's Previous Script (Adapted) ---

def get_roberts_summary_stat_keys() -> List[str]:
    """Defines the keys for the summary statistics vector. MUST BE CONSISTENT."""
    # This is the enhanced version with targeted Gain vs Loss frame contrasts
    keys = [
        'prop_gamble_overall', 'mean_rt_overall',
        'rt_q10_overall', 'rt_q50_overall', 'rt_q90_overall',
    ]
    for cond_name_key in CONDITIONS_ROBERTS.keys(): # Use the keys from CONDITIONS_ROBERTS for consistency
        keys.append(f"prop_gamble_{cond_name_key}")
        keys.append(f"mean_rt_{cond_name_key}")
        keys.append(f"rt_q10_{cond_name_key}")
        keys.append(f"rt_q50_{cond_name_key}")
        keys.append(f"rt_q90_{cond_name_key}")
        for bin_idx in range(5): # 5 RT histogram bins
            keys.append(f'rt_hist_bin{bin_idx}_{cond_name_key}')
            
    # Core framing effect stats
    keys.extend(['framing_effect_ntc', 'framing_effect_tc', 
                'rt_framing_bias_ntc', 'rt_framing_bias_tc'])
    
    # RT distribution stats
    keys.append('rt_std_overall')
    for cond_name_key in CONDITIONS_ROBERTS.keys():
        keys.append(f'rt_std_{cond_name_key}')
    
    # New targeted Gain vs Loss frame contrasts
    keys.extend([
        'mean_rt_Gain_vs_Loss_TC',   # RT contrast in TC condition
        'mean_rt_Gain_vs_Loss_NTC',  # RT contrast in NTC condition
        'rt_median_Gain_vs_Loss_TC',  # Median RT contrast in TC
        'rt_median_Gain_vs_Loss_NTC', # Median RT contrast in NTC
        'framing_effect_rt_gain',     # RT effect within Gain frame (TC vs NTC)
        'framing_effect_rt_loss'      # RT effect within Loss frame (TC vs NTC)
    ])
    
    return keys

def calculate_summary_stats_roberts(df_trials: pd.DataFrame, 
                                   stat_keys: List[str],
                                   impute_rt_means: Optional[Dict[str, float]] = None
                                   ) -> Dict[str, float]:
    """Calculates summary statistics from trial data. Adapted from user's script."""
    summaries = {key: -999.0 for key in stat_keys} 

    if df_trials.empty or len(df_trials) < 5:
        logging.debug(f"Too few trials ({len(df_trials)}) for detailed summary stats, returning placeholders.")
        return summaries

    # Ensure auxiliary columns exist (frame and cond should be in simulated df_trials)
    if 'time_constrained' not in df_trials.columns:
        df_trials['time_constrained'] = df_trials['cond'] == 'tc'
    if 'is_gain_frame' not in df_trials.columns:
        df_trials['is_gain_frame'] = df_trials['frame'] == 'gain'
    
    # Impute NaNs in RT if impute_rt_means is provided (more relevant for observed data prep)
    # For simulated data, we expect RTs unless the sim failed.
    if impute_rt_means and df_trials['rt'].isna().any():
        for cond_key_enum, rt_mean_val in impute_rt_means.items():
            filters = CONDITIONS_ROBERTS[cond_key_enum]
            mask = (df_trials['frame'] == filters['frame']) & (df_trials['cond'] == filters['cond']) & df_trials['rt'].isna()
            df_trials.loc[mask, 'rt'] = rt_mean_val

    # Overall stats
    valid_choices = df_trials['choice'].dropna()
    if not valid_choices.empty:
        summaries['prop_gamble_overall'] = valid_choices.mean()
    
    rts_overall = df_trials['rt'].dropna()
    if not rts_overall.empty:
        summaries['mean_rt_overall'] = rts_overall.mean()
        try:
            quantiles = rts_overall.quantile([0.1, 0.5, 0.9])
            summaries['rt_q10_overall'] = quantiles.get(0.1, -999.0)
            summaries['rt_q50_overall'] = quantiles.get(0.5, -999.0)
            summaries['rt_q90_overall'] = quantiles.get(0.9, -999.0)
        except Exception: # Handle cases like all RTs being identical
            summaries['rt_q10_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q50_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q90_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
    
    cond_props = {}
    cond_rts_mean = {}
    cond_rts_median = {}  # Added for median RT tracking
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        if not subset.empty:
            valid_subset_choices = subset['choice'].dropna()
            if not valid_subset_choices.empty:
                prop_gamble = valid_subset_choices.mean()
                summaries[f'prop_gamble_{cond_key_enum}'] = prop_gamble
                cond_props[cond_key_enum] = prop_gamble
            
            rts_cond = subset['rt'].dropna()
            if not rts_cond.empty:
                mean_rt = rts_cond.mean()
                median_rt = rts_cond.median()
                summaries[f'mean_rt_{cond_key_enum}'] = mean_rt
                cond_rts_mean[cond_key_enum] = mean_rt
                cond_rts_median[cond_key_enum] = median_rt
                try:
                    q_cond = rts_cond.quantile([0.1, 0.5, 0.9])
                    summaries[f'rt_q10_{cond_key_enum}'] = q_cond.get(0.1, -999.0)
                    summaries[f'rt_q50_{cond_key_enum}'] = q_cond.get(0.5, -999.0)
                    summaries[f'rt_q90_{cond_key_enum}'] = q_cond.get(0.9, -999.0)
                except Exception:
                     summaries[f'rt_q10_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0
                     summaries[f'rt_q50_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0
                     summaries[f'rt_q90_{cond_key_enum}'] = rts_cond.iloc[0] if not rts_cond.empty else -999.0

                max_rt_val = 1.0 if 'TC' in cond_key_enum else 3.0 
                bin_edges = np.linspace(0, max_rt_val, 6) 
                if len(rts_cond) >= 1: 
                    hist, _ = np.histogram(rts_cond.clip(0, max_rt_val), bins=bin_edges, density=True)
                    for i_bin, bin_val in enumerate(hist):
                        summaries[f'rt_hist_bin{i_bin}_{cond_key_enum}'] = bin_val
    
    # Original framing effects (choice proportions)
    pg_ln = cond_props.get('Loss_NTC', np.nan); pg_gn = cond_props.get('Gain_NTC', np.nan)
    summaries['framing_effect_ntc'] = pg_ln - pg_gn if not (pd.isna(pg_ln) or pd.isna(pg_gn)) else -999.0
    
    pg_lt = cond_props.get('Loss_TC', np.nan); pg_gt = cond_props.get('Gain_TC', np.nan)
    summaries['framing_effect_tc'] = pg_lt - pg_gt if not (pd.isna(pg_lt) or pd.isna(pg_gt)) else -999.0

    # Original RT framing biases
    rt_ln = cond_rts_mean.get('Loss_NTC', np.nan); rt_gn = cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['rt_framing_bias_ntc'] = rt_ln - rt_gn if not (pd.isna(rt_ln) or pd.isna(rt_gn)) else -999.0

    rt_lt = cond_rts_mean.get('Loss_TC', np.nan); rt_gt = cond_rts_mean.get('Gain_TC', np.nan)
    summaries['rt_framing_bias_tc'] = rt_lt - rt_gt if not (pd.isna(rt_lt) or pd.isna(rt_gt)) else -999.0
    
    # RT standard deviations
    summaries['rt_std_overall'] = rts_overall.std() if not rts_overall.empty else -999.0
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        rts_cond = subset['rt'].dropna()
        summaries[f'rt_std_{cond_key_enum}'] = rts_cond.std() if not rts_cond.empty else -999.0

    # New targeted Gain vs Loss frame contrasts
    # Mean RT contrasts
    summaries['mean_rt_Gain_vs_Loss_TC'] = cond_rts_mean.get('Gain_TC', np.nan) - cond_rts_mean.get('Loss_TC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_NTC'] = cond_rts_mean.get('Gain_NTC', np.nan) - cond_rts_mean.get('Loss_NTC', np.nan)
    
    # Median RT contrasts
    summaries['rt_median_Gain_vs_Loss_TC'] = cond_rts_median.get('Gain_TC', np.nan) - cond_rts_median.get('Loss_TC', np.nan)
    summaries['rt_median_Gain_vs_Loss_NTC'] = cond_rts_median.get('Gain_NTC', np.nan) - cond_rts_median.get('Loss_NTC', np.nan)
    
    # RT effects within frames
    summaries['framing_effect_rt_gain'] = cond_rts_mean.get('Gain_TC', np.nan) - cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['framing_effect_rt_loss'] = cond_rts_mean.get('Loss_TC', np.nan) - cond_rts_mean.get('Loss_NTC', np.nan)

    # Final cleanup
    final_summaries = {key: summaries.get(key, -999.0) for key in stat_keys}
    for k, v in final_summaries.items():
        if pd.isna(v):
            final_summaries[k] = -999.0
    return final_summaries

def prepare_trial_template(roberts_data_path: Path, num_template_trials: int, seed: int) -> None:
    """Loads Roberts data and creates a global trial structure template for simulations."""
    global SUBJECT_TRIAL_STRUCTURE_TEMPLATE, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE
    
    try:
        df = pd.read_csv(roberts_data_path)
    except FileNotFoundError:
        logging.error(f"Roberts data file not found at {roberts_data_path} for template creation.")
        raise
    df = df[df['trialType'] == 'target'].copy()
    df.dropna(subset=['subject', 'frame', 'cond', 'prob', 'rt'], inplace=True) # Ensure RT is present for imputation source
    df['time_constrained'] = df['cond'] == 'tc'
    df['is_gain_frame'] = df['frame'] == 'gain'

    if df.empty:
        raise ValueError("No valid target trials found in Roberts data for template creation.")

    template_cols = ['frame', 'cond', 'prob', 'is_gain_frame', 'time_constrained']
    if len(df) < num_template_trials:
        SUBJECT_TRIAL_STRUCTURE_TEMPLATE = df[template_cols].copy()
    else:
        SUBJECT_TRIAL_STRUCTURE_TEMPLATE = df[template_cols].sample(
            n=num_template_trials, random_state=seed, replace=True # replace=True if N_template > N_available_unique_trial_configs
        ).reset_index(drop=True)
    logging.info(f"SUBJECT_TRIAL_STRUCTURE_TEMPLATE created with {len(SUBJECT_TRIAL_STRUCTURE_TEMPLATE)} trials.")

    OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE = {}
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
         subset = df[(df['frame'] == cond_filters['frame']) & (df['cond'] == cond_filters['cond'])]
         rts_for_impute = subset['rt'].dropna()
         if not rts_for_impute.empty:
            OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE[cond_key_enum] = rts_for_impute.mean()
         else: # Fallback if a condition is empty in the whole dataset
            overall_rts_for_impute = df['rt'].dropna()
            OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE[cond_key_enum] = overall_rts_for_impute.mean() if not overall_rts_for_impute.empty else 1.5 # Absolute fallback
    logging.info(f"Set OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE: {OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE}")


def nes_sbi_simulator(params_tensor: torch.Tensor, stat_keys: List[str]) -> torch.Tensor:
    """SBI wrapper for the NES model simulation for Roberts task. Simulates one parameter set."""
    global SUBJECT_TRIAL_STRUCTURE_TEMPLATE 
    if SUBJECT_TRIAL_STRUCTURE_TEMPLATE is None:
        # This should ideally not happen if prepare_trial_template is called first.
        raise RuntimeError("SUBJECT_TRIAL_STRUCTURE_TEMPLATE not initialized before simulation call.")

    # params_tensor is expected to be 1D [num_params]
    if params_tensor.ndim > 1: params_tensor = params_tensor.squeeze(0) # Handle potential [1, num_params]

    params_dict = {name: val.item() for name, val in zip(PARAM_NAMES, params_tensor)}
    agent = MVNESAgent(config={}) 
    
    sim_results_list = []
    for _, trial_info in SUBJECT_TRIAL_STRUCTURE_TEMPLATE.iterrows():
        salience_input = trial_info['prob'] 
        norm_input = 1.0 if trial_info['is_gain_frame'] else -1.0

        agent_run_params = {
            'w_n': params_dict['v_norm'], 
            'threshold_a': params_dict['a_0'], # Pass base threshold directly
            'w_s': params_dict['w_s_eff'], 
            't': params_dict['t_0'],
            'alpha_gain': params_dict['alpha_gain'], # Added alpha_gain
            **BASE_SIM_PARAMS 
        }
        
        try:
            trial_output = agent.run_mvnes_trial(
                salience_input=salience_input,
                norm_input=norm_input,
                params=agent_run_params 
            )
            sim_rt = trial_output.get('rt', np.nan) # This RT from MVNESAgent should already include t_0.

            if not pd.isna(sim_rt): # Only clip if RT is not NaN
                if trial_info['time_constrained']:
                    sim_rt = min(sim_rt, 1.0)
            
            sim_results_list.append({
                'rt': sim_rt, 'choice': trial_output.get('choice', np.nan),
                'frame': trial_info['frame'], 'cond': trial_info['cond']
                # 'time_constrained' and 'is_gain_frame' will be re-added by calculate_summary_stats if needed
            })
        except Exception as e_sim:
            logging.debug(f"Sim trial exception for params {params_dict}: {e_sim}")
            sim_results_list.append({'rt': np.nan, 'choice': np.nan, 'frame': trial_info['frame'], 'cond': trial_info['cond']})
            
    df_sim_batch = pd.DataFrame(sim_results_list)
    # calculate_summary_stats expects 'time_constrained', 'is_gain_frame' if not already present.
    # It will add them.
    summary_stats_dict = calculate_summary_stats_roberts(df_sim_batch, stat_keys, OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE)
    
    summary_stats_vector = [summary_stats_dict.get(k, -999.0) for k in stat_keys]
    return torch.tensor(summary_stats_vector, dtype=torch.float32)


# --- Main Script Logic ---
def setup_output_directory(output_base_name: str) -> Path:
    """Creates the main output directory and standard subdirectories."""
    base_dir = Path(output_base_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'data').mkdir(parents=True, exist_ok=True)
    (base_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (base_dir / 'models').mkdir(parents=True, exist_ok=True) # Add this line
    (base_dir / 'npe_cache').mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory setup at: {base_dir.resolve()}")
    return base_dir, base_dir / 'data', base_dir / 'plots', base_dir / 'models', base_dir / 'npe_cache'

def main():
    parser = argparse.ArgumentParser(description="Run SBC for NES model on Roberts et al. task.")
    parser.add_argument('--npe_train_sims', type=int, default=5000)  # Reduced from 30000 for rapid testing
    parser.add_argument('--template_trials', type=int, default=100)
    parser.add_argument('--sbc_datasets', type=int, default=100)  # Reduced from 300 for rapid testing
    parser.add_argument('--sbc_posterior_samples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None) 
    parser.add_argument('--sbc_debug_mode', action='store_true', help='Run SBC on only 10 datasets for debugging.') 
    parser.add_argument('--output_base_name', type=str, default="sbc_nes_roberts_rebuilt")
    parser.add_argument('--roberts_data_file', type=str, default="./roberts_framing_data/ftp_osf_data.csv") # Adjust path as needed
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--force_retrain_npe', action='store_true', help="Force retraining of NPE even if a checkpoint exists.") # Added for consistency, though loading is removed
    args = parser.parse_args()

    if args.seed is None: args.seed = int(time.time())
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}. Seed: {args.seed}")
    
    output_dir, output_dir_data, output_dir_plots, output_dir_models, output_dir_cache = setup_output_directory(args.output_base_name)
    summary_stat_keys = get_roberts_summary_stat_keys()
    num_summary_stats = len(summary_stat_keys)
    if num_summary_stats <=0: logging.error("Num summary stats is 0!"); sys.exit(1)
    logging.info(f"Number of summary stats defined: {num_summary_stats}")

    # Prepare trial template
    try:
        prepare_trial_template(Path(args.roberts_data_file), args.template_trials, args.seed)
    except Exception as e_template:
        logging.error(f"Failed to prepare trial template: {e_template}", exc_info=True); sys.exit(1)

    sbi_prior = BoxUniform(low=PRIOR_LOW.to(device), high=PRIOR_HIGH.to(device), device=device.type)
    def actual_simulator_for_sbi(parameter_sample_batch_tensor: torch.Tensor) -> torch.Tensor:
        # This function is called by simulate_for_sbi.
        # parameter_sample_batch_tensor can be [batch_size, num_params] or [num_params]
        if parameter_sample_batch_tensor.ndim == 1: # Single sample
             return nes_sbi_simulator(parameter_sample_batch_tensor, summary_stat_keys)
        else: # Batch of samples
            batch_results = []
            for i_sample in range(parameter_sample_batch_tensor.shape[0]):
                batch_results.append(nes_sbi_simulator(parameter_sample_batch_tensor[i_sample], summary_stat_keys))
            return torch.stack(batch_results)

    logging.info(f"Starting NPE training with {args.npe_train_sims} simulations...")

    theta_train, x_train = simulate_for_sbi(
        simulator=actual_simulator_for_sbi,
        proposal=sbi_prior,
        num_simulations=args.npe_train_sims,
        num_workers=1, 
        show_progress_bar=True,
        simulation_batch_size=1 # Kept at 1 as per previous findings, new wrapper handles potential batches if sbi sends them
    )

    logging.info(f"Generated training data: theta_train {theta_train.shape}, x_train {x_train.shape}")

    # Filter out simulations that resulted in NaNs or Infs or only placeholder values in summary stats
    # This can happen if a simulation run fails catastrophically for some parameter sets
    valid_sim_mask = ~(x_train == -999.0).all(dim=1) & ~torch.isnan(x_train).any(dim=1) & ~torch.isinf(x_train).any(dim=1)
    theta_train_valid = theta_train[valid_sim_mask]
    x_train_valid = x_train[valid_sim_mask]
    if len(theta_train_valid) < args.npe_train_sims * 0.5:
        logging.error(f"Less than 50% of simulations are valid ({len(theta_train_valid)}/{args.npe_train_sims}). Aborting."); sys.exit(1)
    logging.info(f"Using {len(theta_train_valid)} valid simulations for NPE training.")

    # For v0.22.0, rely on monkey-patched standardization
    npe = SNPE(prior=sbi_prior, density_estimator='maf', device=device.type)
    density_estimator = npe.append_simulations(theta_train_valid, x_train_valid).train(
        show_train_summary=True,
        force_first_round_loss=True
    )

    # --- Compute and save training summary-stat means and stds ---
    training_stat_means = x_train_valid.mean(dim=0).cpu().numpy()
    training_stat_stds = x_train_valid.std(dim=0).cpu().numpy()
    np.save(output_dir_data / 'training_stat_means.npy', training_stat_means)
    np.save(output_dir_data / 'training_stat_stds.npy', training_stat_stds)
    pd.DataFrame(training_stat_means).to_csv(output_dir_data / 'training_stat_means.csv', index=False, header=False)
    pd.DataFrame(training_stat_stds).to_csv(output_dir_data / 'training_stat_stds.csv', index=False, header=False)

    # --- Save summary_stat_keys and parameter names ---
    with open(output_dir_data / 'summary_stat_keys.json', 'w') as f:
        json.dump(summary_stat_keys, f, indent=2)
    with open(output_dir_data / 'parameter_names.json', 'w') as f:
        json.dump(PARAM_NAMES, f, indent=2)

    # --- Save prior bounds ---
    prior_bounds = {'low': PRIOR_LOW.cpu().numpy().tolist(), 'high': PRIOR_HIGH.cpu().numpy().tolist()}
    with open(output_dir_data / 'prior_bounds.json', 'w') as f:
        json.dump(prior_bounds, f, indent=2)
    torch.save(prior_bounds, output_dir_data / 'prior_low_high.pt')

    # --- Save all valid training data for reproducibility ---
    torch.save({'theta_train_valid': theta_train_valid.cpu(), 'x_train_valid': x_train_valid.cpu()}, output_dir_data / 'theta_x_train_valid.pt')
    np.save(output_dir_data / 'x_train_valid.npy', x_train_valid.cpu().numpy())
    np.save(output_dir_data / 'theta_train_valid.npy', theta_train_valid.cpu().numpy())
    pd.DataFrame(x_train_valid.cpu().numpy()).to_csv(output_dir_data / 'simulated_summary_stats.csv', index=False, header=False)
    training_stat_means = x_train_valid.mean(dim=0).cpu().numpy()
    training_stat_stds = x_train_valid.std(dim=0).cpu().numpy()
    np.save(output_dir_data / 'training_stat_means.npy', training_stat_means)
    np.save(output_dir_data / 'training_stat_stds.npy', training_stat_stds)
    pd.DataFrame(training_stat_means).to_csv(output_dir_data / 'training_stat_means.csv', index=False, header=False)
    pd.DataFrame(training_stat_stds).to_csv(output_dir_data / 'training_stat_stds.csv', index=False, header=False)

    # --- Save summary_stat_keys and parameter names ---
    with open(output_dir_data / 'summary_stat_keys.json', 'w') as f:
        json.dump(summary_stat_keys, f, indent=2)
    with open(output_dir_data / 'parameter_names.json', 'w') as f:
        json.dump(PARAM_NAMES, f, indent=2)

    # --- Save prior bounds ---
    prior_bounds = {'low': PRIOR_LOW.cpu().numpy().tolist(), 'high': PRIOR_HIGH.cpu().numpy().tolist()}
    with open(output_dir_data / 'prior_bounds.json', 'w') as f:
        json.dump(prior_bounds, f, indent=2)
    torch.save(prior_bounds, output_dir_data / 'prior_low_high.pt')

    # --- Save checkpoint with all critical metadata ---
    npe_save_path = output_dir_models / f"nes_npe_sims{args.npe_train_sims}_template{args.template_trials}_seed{args.seed}.pt"
    npe_checkpoint = {
        'density_estimator_state_dict': density_estimator.state_dict(),
        'prior_params': prior_bounds,
        'param_names': PARAM_NAMES,
        'num_summary_stats': num_summary_stats,
        'summary_stat_keys': summary_stat_keys,
        'npe_train_sims': args.npe_train_sims,
        'template_trials_for_training': args.template_trials,
        'sbi_version': sbi.__version__,
        'training_seed': args.seed,
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_stat_means': training_stat_means.tolist(),
        'training_stat_stds': training_stat_stds.tolist()
    }
    torch.save(npe_checkpoint, npe_save_path)
    logging.info(f"Saved trained NPE and all empirical fit metadata to {npe_save_path}")

    # --- Explicitly save essential artifacts for empirical fit ---
    # Save summary_stat_keys.json
    with open(output_dir_data / "summary_stat_keys.json", "w") as f:
        json.dump(summary_stat_keys, f, indent=2)
    # Save training_stat_means.pt
    torch.save(torch.tensor(training_stat_means), output_dir_data / "training_stat_means.pt")
    # Save parameter_names.json
    with open(output_dir_data / "parameter_names.json", "w") as f:
        json.dump(PARAM_NAMES, f, indent=2)
    # Save prior_bounds.json
    with open(output_dir_data / "prior_bounds.json", "w") as f:
        json.dump(prior_bounds, f, indent=2)
    # Save x_train_valid.pt and theta_train_valid.pt for debugging
    torch.save(x_train_valid.cpu(), output_dir_data / "x_train_valid.pt")
    torch.save(theta_train_valid.cpu(), output_dir_data / "theta_train_valid.pt")

    logging.info(f"Starting SBC with {args.sbc_datasets} datasets...")
    
    theta_sbc_gt = sbi_prior.sample((args.sbc_datasets,)).to(device) # Ensure on correct device

    x_sbc_obs_list = []
    for i in range(args.sbc_datasets):
        if (i + 1) % (args.sbc_datasets // 10 or 1) == 0: logging.info(f"Simulating SBC dataset {i+1}/{args.sbc_datasets}...")
        x_sbc_obs_list.append(actual_simulator_for_sbi(theta_sbc_gt[i])) 
    x_sbc_obs = torch.stack(x_sbc_obs_list).to(device) # Ensure results are on device
    logging.info(f"Generated SBC observations: theta_sbc_gt {theta_sbc_gt.shape}, x_sbc_obs {x_sbc_obs.shape}")

    valid_sbc_mask = ~(x_sbc_obs == -999.0).all(dim=1) & ~torch.isnan(x_sbc_obs).any(dim=1) & ~torch.isinf(x_sbc_obs).any(dim=1)
    theta_sbc_gt_valid = theta_sbc_gt[valid_sbc_mask].to(device)
    x_sbc_obs_valid = x_sbc_obs[valid_sbc_mask].to(device)
    num_valid_sbc_datasets = len(theta_sbc_gt_valid)
    if num_valid_sbc_datasets == 0: logging.error("No valid SBC datasets generated."); sys.exit(1)
    logging.info(f"Using {num_valid_sbc_datasets} valid datasets for SBC rank calculation.")

    posterior_object_for_sbc = npe.build_posterior(density_estimator)
    logging.info(f"Posterior object for SBC built from trained density_estimator.")
    logging.info(f"Proceeding with {num_valid_sbc_datasets} valid datasets for SBC rank calculation.")

    ranks = run_sbc(
        theta_sbc_gt_valid,
        x_sbc_obs_valid,
        posterior_object_for_sbc, 
        args.sbc_posterior_samples 
    ) 
    logging.info(f"SBC ranks calculated. Shape before potential transpose: {ranks.shape}")

    # Patch: Ensure ranks is 2D (num_datasets, num_parameters)
    if ranks.ndim == 1:
        logging.warning(f"Ranks tensor is 1D (shape: {ranks.shape}), unsqueezing to 2D for downstream compatibility.")
        ranks = ranks.unsqueeze(0)

    df_ranks_columns = [f"rank_{PARAM_NAMES[j]}" if j < len(PARAM_NAMES) else f"rank_param_{j}" for j in range(ranks.shape[1])]
    df_ranks = pd.DataFrame(ranks.cpu().numpy(), columns=df_ranks_columns)
    df_ranks.to_csv(output_dir_data / "sbc_ranks.csv", index=False)
    logging.info(f"SBC ranks saved to {output_dir_data / 'sbc_ranks.csv'}")

    ks_results = {}
    num_params_from_ranks = ranks.shape[1]
    for i in range(num_params_from_ranks):
        param_name = PARAM_NAMES[i] if i < len(PARAM_NAMES) else f"param_{i}"
        param_ranks_cpu = ranks[:, i].cpu().numpy() 
        normalized_ranks = param_ranks_cpu / args.sbc_posterior_samples
        if len(np.unique(normalized_ranks)) < 2:
            logging.warning(f"KS test for '{param_name}': All ranks are identical ({normalized_ranks[0]}). KS test result will be trivial (p=0 or p=1) and likely uninformative.")
            ks_stat, ks_pval = (1.0, 0.0) if np.all(normalized_ranks == normalized_ranks[0]) else sp_stats.kstest(normalized_ranks, 'uniform')
        else:
            ks_stat, ks_pval = sp_stats.kstest(normalized_ranks, 'uniform')
        ks_results[param_name] = {'ks_stat': ks_stat, 'ks_pval': ks_pval}
    logging.info(f"SBC KS test results: {ks_results}")
    with open(output_dir_data / "sbc_ks_test_results.json", 'w') as f_ks:
        json.dump(ks_results, f_ks, indent=4)
    logging.info(f"SBC KS test results saved to {output_dir_data / 'sbc_ks_test_results.json'}")

    # --- MANUAL SBC DIAGNOSTIC PLOTTING: ECDFs and Histograms ---
    try:
        # Ranks: shape (num_sbc_datasets, num_params)
        ranks_np = ranks.cpu().numpy()
        if ranks_np.shape[1] != len(PARAM_NAMES):
            param_names = [f'param_{i}' for i in range(ranks_np.shape[1])]
        else:
            param_names = PARAM_NAMES
        num_params = ranks_np.shape[1]
        num_datasets = ranks_np.shape[0]
        fig, axes = plt.subplots(num_params, 2, figsize=(10, 2.5*num_params))
        if num_params == 1:
            axes = np.array([axes])
        for i in range(num_params):
            # ECDF
            sorted_ranks = np.sort(ranks_np[:, i])
            ecdf = np.arange(1, num_datasets+1) / num_datasets
            axes[i, 0].plot(sorted_ranks / args.sbc_posterior_samples, ecdf, marker='.', linestyle='-', color='blue')
            axes[i, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i, 0].set_title(f"ECDF: {param_names[i]}")
            axes[i, 0].set_xlabel("Normalized Rank")
            axes[i, 0].set_ylabel("ECDF")
            axes[i, 0].set_xlim([0, 1])
            axes[i, 0].set_ylim([0, 1])
            # Histogram
            axes[i, 1].hist(ranks_np[:, i], bins=max(10, args.sbc_posterior_samples // 20), range=(0, args.sbc_posterior_samples), color='gray', alpha=0.7, density=True)
            axes[i, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5)
            axes[i, 1].set_title(f"Rank Histogram: {param_names[i]}")
            axes[i, 1].set_xlabel("Rank")
            axes[i, 1].set_ylabel("Density")
            axes[i, 1].set_xlim([0, args.sbc_posterior_samples])
        fig.tight_layout()
        fig.suptitle(f'SBC Manual Diagnostics ({num_datasets} Datasets, {args.sbc_posterior_samples} Posterior Samples)', y=1.02)
        sbc_plot_path = output_dir_plots / "sbc_manual_diagnostics_plot.png"
        fig.savefig(sbc_plot_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"SBC manual diagnostics plot saved to {sbc_plot_path}")
    except Exception as e:
        logging.error(f"Manual SBC diagnostics plotting failed: {e}")


    sbc_metadata = {
        'npe_train_sims': args.npe_train_sims, 'template_trials': args.template_trials,
        'sbc_datasets_attempted': args.sbc_datasets, 'sbc_datasets_valid': num_valid_sbc_datasets,
        'sbc_posterior_samples': args.sbc_posterior_samples, 'seed': args.seed,
        'output_dir': str(output_dir), 'npe_save_path': str(npe_save_path),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'ks_results': ks_results
    }
    with open(output_dir_data / "sbc_run_metadata.json", 'w') as f_meta:
        json.dump(sbc_metadata, f_meta, indent=4)
    logging.info(f"SBC metadata saved. Output directory: {output_dir}")
    logging.info("SBC run finished successfully.")

if __name__ == "__main__":
    main()