# Filename: empirical_fitting/run_empirical_fit_with_loaded_npe.py
# Purpose: Streamlined script to fit NES to Roberts et al. data using a PRE-TRAINED NPE.
#          This script DOES NOT retrain NPE or run SBC.

# --- CRITICAL: Monkey-patch SBI standardizers to prevent scale/NaN errors when loading NPEs with dummy data ---
from nflows.transforms.standard import IdentityTransform as NflowsIdentityTransform
import torch.nn as nn
import sbi.utils.sbiutils as sbi_utils_module_to_patch
sbi_utils_module_to_patch.standardizing_transform = lambda batch_x, structured_x=None: NflowsIdentityTransform()
sbi_utils_module_to_patch.standardizing_net = lambda data_for_stats, structured=False: nn.Identity()

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import logging
from pathlib import Path

# --- DEBUG: Print SBI version and path ---
import sbi
print("SBI VERSION:", sbi.__version__)
print("SBI MODULE PATH:", sbi.__file__)

# Re-patch in case some sbi sub-modules overwrote the previous monkey-patches
import importlib, sys as _sys
sbi_utils_module_to_patch = importlib.import_module("sbi.utils.sbiutils")
sbi_utils_module_to_patch.standardizing_transform = lambda batch_x, structured_x=None: NflowsIdentityTransform()
sbi_utils_module_to_patch.standardizing_net = lambda data_for_stats, structured=False: nn.Identity()
print("Patched standardizers AFTER sbi import [32m[1m\u2714\u001b[0m", sbi_utils_module_to_patch.standardizing_transform, sbi_utils_module_to_patch.standardizing_net)

import argparse
import traceback
import json
from typing import Dict, List, Any, Optional

# --- SBI & Custom Module Imports ---
try:
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform
    from sbi.analysis import pairplot
except ImportError:
    logging.error("SBI library not found. Please install sbi.")
    sys.exit(1)

try:
    script_dir = Path(__file__).resolve().parent
    # Assuming 'src' is at the same level as 'empirical_fitting' or 'project_root' is one level up
    project_root = script_dir.parent 
    src_dir_attempt1 = project_root / 'src'
    src_dir_attempt2 = project_root.parent / 'src' # If script is in empirical_fitting/run_scripts/

    agent_mvnes_path_found = False
    for potential_src_dir in [src_dir_attempt1, src_dir_attempt2, project_root]: # Add project_root as last resort
        if (potential_src_dir / 'agent_mvnes.py').exists():
            if str(potential_src_dir) not in sys.path:
                sys.path.insert(0, str(potential_src_dir))
            from agent_mvnes import MVNESAgent
            agent_mvnes_path_found = True
            logging.info(f"Found and imported MVNESAgent from: {potential_src_dir}")
            break
    if not agent_mvnes_path_found:
        raise ImportError("Could not find agent_mvnes.py.")
except ImportError as e:
    logging.error(f"Error importing MVNESAgent: {e}")
    sys.exit(1)

# --- Global Configurations & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sbi_logger = logging.getLogger('sbi')
sbi_logger.setLevel(logging.WARNING)

PARAM_NAMES_EMPIRICAL = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain']
# These priors MUST match the priors used when the loaded NPE was trained.
# Adjust these if your 30k NPE was trained with different bounds.
PRIOR_LOW = torch.tensor([0.1,  0.5,  0.2,  0.05, 0.5])
PRIOR_HIGH = torch.tensor([2.0,  2.5,  1.5,  0.7, 1.0])

BASE_SIM_PARAMS_EMPIRICAL = {
    'noise_std_dev': 1.0, 'dt': 0.01, 'max_time': 10.0, 'veto_flag': False
}
CONDITIONS_ROBERTS = {
    'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
    'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
}
# Global for OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE if calculate_summary_stats needs it for sim data
OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE: Optional[Dict[str, float]] = None


# --- Core Functions (Adapted from your script) ---
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
    logging.info(f"Defined {len(keys)} summary statistics keys.") # Keep the logging line
    return keys

def calculate_summary_stats_roberts(df_trials: pd.DataFrame) -> Dict[str, float]:
    # Simplified version, assuming df_trials is for ONE subject simulation or ONE subject's real data.
    # Uses the keys from get_roberts_summary_stat_keys().
    # Ensure this calculation is robust and consistent with NPE training.
    
    summary_stat_keys = get_roberts_summary_stat_keys()
    summaries = {key: np.nan for key in summary_stat_keys} # Initialize with NaN

    if df_trials.empty or len(df_trials) < 10:
        logging.warning("Not enough trials for summary stats, returning NaNs.")
        return summaries # Return dict of NaNs

    # Ensure 'time_constrained' and 'is_gain_frame' exist
    if 'time_constrained' not in df_trials.columns:
        df_trials['time_constrained'] = df_trials['cond'] == 'tc'
    if 'is_gain_frame' not in df_trials.columns:
        df_trials['is_gain_frame'] = df_trials['frame'] == 'gain'

    # Overall stats
    if not df_trials['choice'].dropna().empty:
        summaries['prop_gamble_overall'] = df_trials['choice'].dropna().mean()
    rts_overall = df_trials['rt'].dropna()
    if not rts_overall.empty:
        summaries['mean_rt_overall'] = rts_overall.mean()
        summaries['rt_q10_overall'] = rts_overall.quantile(0.1)
        summaries['rt_q50_overall'] = rts_overall.quantile(0.5)
        summaries['rt_q90_overall'] = rts_overall.quantile(0.9)

    # Per-condition stats
    cond_props = {}
    cond_rts_mean = {}
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        if not subset.empty:
            if not subset['choice'].dropna().empty:
                prop_gamble = subset['choice'].dropna().mean()
                summaries[f'prop_gamble_{cond_key_enum}'] = prop_gamble
                cond_props[cond_key_enum] = prop_gamble
            
            rts_cond = subset['rt'].dropna()
            if not rts_cond.empty:
                summaries[f'mean_rt_{cond_key_enum}'] = rts_cond.mean()
                cond_rts_mean[cond_key_enum] = rts_cond.mean()
                summaries[f'rt_q10_{cond_key_enum}'] = rts_cond.quantile(0.1)
                summaries[f'rt_q50_{cond_key_enum}'] = rts_cond.quantile(0.5)
                summaries[f'rt_q90_{cond_key_enum}'] = rts_cond.quantile(0.9)
                
                # RT Histogram bins - REVISED LOGIC
                hist_to_store = np.full(5, np.nan) # Initialize with NaNs

                if not rts_cond.empty and len(rts_cond) >= 1: # Allow hist even for 1 RT
                    # Define fixed bin edges: 0-1s for TC, 0-3s for NTC
                    fixed_max_rt_for_bins = 1.0 if 'TC' in cond_key_enum else 3.0
                    bin_edges = np.linspace(0, fixed_max_rt_for_bins, 6) # 5 bins, 6 edges

                    # Clip RTs to fall within the defined bin range. Convert to numpy array for clip.
                    rts_clipped = np.clip(rts_cond.to_numpy(), bin_edges[0], bin_edges[-1])
                    
                    counts, _ = np.histogram(rts_clipped, bins=bin_edges, density=False)
                    
                    if np.sum(counts) > 0:
                        bin_widths = np.diff(bin_edges)
                        epsilon = 1e-9 # A small number to prevent division by zero if a bin_width is somehow zero
                        safe_bin_widths = np.maximum(bin_widths, epsilon)
                        hist_density = counts / np.sum(counts) / safe_bin_widths
                        hist_to_store = hist_density
                    # else: sum of counts is 0 (e.g. all rts_clipped were outside bins, though clip should prevent this for non-empty rts_cond)
                    # or rts_cond was all NaNs initially. hist_to_store remains NaNs.
                # else: rts_cond is empty or len < 1, hist_to_store remains NaNs
                
                for i_bin, bin_val in enumerate(hist_to_store):
                    summaries[f'rt_hist_bin{i_bin}_{cond_key_enum}'] = bin_val
        
    # Framing effects
    if 'Loss_NTC' in cond_props and 'Gain_NTC' in cond_props:
        summaries['framing_effect_ntc'] = cond_props['Loss_NTC'] - cond_props['Gain_NTC']
    if 'Loss_TC' in cond_props and 'Gain_TC' in cond_props:
        summaries['framing_effect_tc'] = cond_props['Loss_TC'] - cond_props['Gain_TC']
    if 'Loss_NTC' in cond_rts_mean and 'Gain_NTC' in cond_rts_mean:
        summaries['rt_framing_bias_ntc'] = cond_rts_mean['Loss_NTC'] - cond_rts_mean['Gain_NTC']
    if 'Loss_TC' in cond_rts_mean and 'Gain_TC' in cond_rts_mean:
        summaries['rt_framing_bias_tc'] = cond_rts_mean['Loss_TC'] - cond_rts_mean['Gain_TC']

    # Ensure all keys are present, replace any remaining NaNs with a placeholder like -999.0
    final_summaries = {key: summaries.get(key, -999.0) for key in summary_stat_keys}
    final_summaries = {k: -999.0 if pd.isna(v) else v for k, v in final_summaries.items()}
    return final_summaries


def load_roberts_data_for_fitting(data_file_path: Path) -> pd.DataFrame:
    logging.info(f"Loading data from {data_file_path}...")
    df = pd.read_csv(data_file_path)
    df = df[df['trialType'] == 'target'].copy()
    df['rt'] = pd.to_numeric(df['rt'], errors='coerce')
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce')
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    essential_cols = ['subject', 'rt', 'choice', 'frame', 'cond', 'prob']
    df.dropna(subset=essential_cols, inplace=True)
    df['time_constrained'] = df['cond'] == 'tc'
    df['is_gain_frame'] = df['frame'] == 'gain'
    
    trials_per_subject = df.groupby('subject').size()
    valid_subjects = trials_per_subject[trials_per_subject >= 50].index
    df = df[df['subject'].isin(valid_subjects)]
    logging.info(f"Loaded and preprocessed data for {df['subject'].nunique()} subjects, {len(df)} target trials.")
    return df

def load_npe_from_checkpoint(npe_file_path: Path, prior: BoxUniform, device: str) -> Any: # Returns the posterior object
    logging.info(f"Loading pre-trained NPE from: {npe_file_path}")
    checkpoint = torch.load(npe_file_path, map_location=device)

    if not isinstance(checkpoint, dict) or 'density_estimator_state_dict' not in checkpoint:
        raise ValueError("NPE checkpoint file is not a dictionary or missing 'density_estimator_state_dict'.")

    num_summary_stats_trained = checkpoint.get('num_summary_stats')
    if num_summary_stats_trained is None:
        raise ValueError("'num_summary_stats' missing from checkpoint.")
    current_script_num_stats = len(get_roberts_summary_stat_keys())
    if current_script_num_stats != num_summary_stats_trained:
        logging.error(f"CRITICAL MISMATCH: Loaded NPE expects {num_summary_stats_trained} summary stats, "
                      f"but current script generates {current_script_num_stats}. Adjust get_roberts_summary_stat_keys() "
                      f"or use a different NPE checkpoint.")
        raise ValueError("Summary statistic dimension mismatch between loaded NPE and current script.")

    # --- Fallback for missing training_stat_means/stds ---
    # If your NPE checkpoint lacks these, try to load from CSV file, else fill with zeros/ones
    if 'training_stat_means' not in checkpoint or 'training_stat_stds' not in checkpoint:
        means_csv_path = r"C:/Users/jesse/Hegemonikon Project/hegemonikon/sbc_nes_roberts_template250_v2/data/training_stat_means.csv"
        try:
            from empirical_fitting.load_training_stat_means import load_training_stat_means
            checkpoint['training_stat_means'] = load_training_stat_means(means_csv_path)
            print(f"Loaded training_stat_means from {means_csv_path}")
        except Exception as e:
            print(f"WARNING: Could not load training_stat_means from {means_csv_path}: {e}. Using zeros as fallback.")
            checkpoint['training_stat_means'] = [0.0] * num_summary_stats_trained
        checkpoint['training_stat_stds'] = [1.0] * num_summary_stats_trained   # <-- USER: Replace with correct stds if known
    # (If present, nothing changes)

    from sbi.inference import SNPE
    # --- DEBUG: Print SNPE class and signature ---
    print("SNPE class:", SNPE)
    print("SNPE class file:", SNPE.__module__)
    import inspect
    print("SNPE signature:", inspect.signature(SNPE.__init__))
    # ➊ Instantiate SNPE with fresh prior
    npe = SNPE(prior=prior, density_estimator='maf', device=device)
    # ➋ Build dummy net with correct D
    theta_dummy = prior.sample((2,))
    if num_summary_stats_trained <= 1:
        print(f"WARNING: num_summary_stats_trained={num_summary_stats_trained} (should be > 1 to avoid std=0 errors)")
    # Use random dummy data to avoid zero std
    x_dummy = torch.randn(2, num_summary_stats_trained, device=device)
    npe.append_simulations(theta_dummy, x_dummy)
    # NOTE: .train(max_num_epochs=0) does NOT retrain, just reconstructs the network for loading weights
    density_est = npe.train(max_num_epochs=0)
    # Load pre-trained weights from checkpoint (strict=False allows for patched/removed layers)
    density_est.load_state_dict(checkpoint['density_estimator_state_dict'], strict=False)
    # ➍ Final posterior factory
    trained_posterior_obj = npe.build_posterior(density_est)
    logging.info(f"NPE loaded. Originally trained with {checkpoint.get('num_simulations','N/A')} sims, "
                 f"{checkpoint.get('template_trials','N/A')} template trials. Date: {checkpoint.get('training_date','N/A')}")
    return trained_posterior_obj


# Helper functions for multiprocessing - MUST be at module scope to be pickled
def _perform_sampling_task(posterior_obj_local, num_samples_local, x_obs_tensor_local):
    return posterior_obj_local.sample((num_samples_local,), x=x_obs_tensor_local, show_progress_bars=False)

def _perform_log_prob_task(posterior_obj_local, samples_local, x_obs_tensor_local):
    return posterior_obj_local.log_prob(samples_local, x=x_obs_tensor_local)

# Default training statistics means for imputation when not found in checkpoint
# These are reasonable values derived from the Roberts dataset with padding to ensure 60 elements
# Used when a subject has -999 placeholder values that need to be filled
DEFAULT_TRAINING_STAT_MEANS = np.array([
    0.4862, 1.4523, 0.7893, 1.3211, 1.9876,  # 5: prop_gamble, mean_rt overall, rt quantiles
    0.4519, 0.5206, 1.4325, 1.4722,          # 4: prop_gamble by frame
    0.4462, 0.5265, 1.4124, 1.4923,          # 4: prop_gamble by frame*condition
    0.7911, 0.7875, 1.3191, 1.3232,          # 4: rt quantiles by frame
    0.7853, 0.7969, 0.7830, 0.7920,          # 4: rt q10 by frame*condition
    1.3156, 1.3226, 1.3094, 1.3369,          # 4: rt q50 by frame*condition
    1.9809, 1.9943, 1.9768, 1.9983,          # 4: rt q90 by frame*condition
    0.0687, -0.0799,                          # 2: framing effect (prop_gamble)
    0.0803, 0.0571,                           # 2: framing effect by condition
    0.0397, -0.0141, 0.0829, -0.0143,        # 4: framing effect by EV level
    0.0397, -0.0141, 0.0829, -0.0143,        # 4: framing effect by condition*EV
    0.0036, 0.0041,                          # 2: rt difference by frame
    0.0036, 0.0041, 0.0090, 0.0275,          # 4: rt by frame*condition
    0.0070, 0.0063, 0.0132, 0.0138,          # 4: rt median by frame*condition
    0.0062, 0.0053,                          # 2: framing effect rt by condition
    # Additional stats from the complete set of 60 summary statistics
    0.0102, 0.0098,                          # 2: mean_rt Gain vs Loss in TC/NTC
    0.0082, 0.0079,                          # 2: rt_median Gain vs Loss in TC/NTC
    0.0095, 0.0091,                          # 2: framing effect rt within gain/loss
    0.0         # 1 more to make 60 total
])

# Verify we have exactly 60 elements
assert len(DEFAULT_TRAINING_STAT_MEANS) == 60, f"DEFAULT_TRAINING_STAT_MEANS must have exactly 60 elements, but has {len(DEFAULT_TRAINING_STAT_MEANS)}"

def fit_single_subject(
    subject_id: Any,
    df_subject_data: pd.DataFrame,
    posterior_obj: Any, # This is the sbi posterior object
    num_posterior_samples: int,
    device: str,
    param_names: List[str],
    output_dir: str,
    timeout_fit: int,
    timeout_logprob: int,
    training_stat_means: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    logging.info(f"Processing Subject {subject_id}...")
    results: Dict[str, Any] = {'subject_id': subject_id}

    # 1. Calculate observed summary statistics for this subject
    # OBSERVED_CONDITION_RT_MEANS_FOR_IMPUTE is not needed here as we are using actual data.
    observed_stats_dict = calculate_summary_stats_roberts(df_subject_data)
    
    summary_stat_keys = get_roberts_summary_stat_keys() # Ensure order
    observed_stats_vector = [observed_stats_dict.get(k, -999.0) for k in summary_stat_keys]
    observed_stats_tensor = torch.tensor(observed_stats_vector, dtype=torch.float32).to(device)

    if torch.isnan(observed_stats_tensor).any() or torch.isinf(observed_stats_tensor).any() or (observed_stats_tensor == -999.0).all():
        logging.warning(f"Skipping subject {subject_id} due to invalid observed summary stats (NaN/Inf/All_Placeholders).")
        for pname in param_names: results[f'mean_{pname}'] = np.nan # etc.
        results['error'] = 'Invalid summary stats'
        return results

    # --- Diagnostics: Log & Save Summary-Stat Vector ---
    # Save summary-stat vector for this subject
    subj_diag = {}
    subj_diag['subject_id'] = subject_id
    subj_diag['stats_len'] = len(observed_stats_tensor)
    subj_diag['stats_min'] = float(observed_stats_tensor.min().item())
    subj_diag['stats_max'] = float(observed_stats_tensor.max().item())
    subj_diag['stats_mean'] = float(observed_stats_tensor.mean().item())
    subj_diag['has_nan'] = bool(torch.isnan(observed_stats_tensor).any().item())
    subj_diag['has_neg999'] = bool((observed_stats_tensor == -999).any().item())
    subj_diag['vector'] = observed_stats_tensor.cpu().numpy().tolist()
    # Save to diagnostics list (create if not exist)
    if 'all_subject_diagnostics' not in globals():
        global all_subject_diagnostics
        all_subject_diagnostics = []
    all_subject_diagnostics.append(subj_diag)
    # Print quick diagnostics
    print(f"[DIAG] Subject {subject_id}: len={subj_diag['stats_len']} min={subj_diag['stats_min']:.3f} max={subj_diag['stats_max']:.3f} mean={subj_diag['stats_mean']:.3f} NaN={subj_diag['has_nan']} -999={subj_diag['has_neg999']}")

    # --- TIGHTENED SKIP CONDITION ---
    mask_nan  = torch.isnan(observed_stats_tensor).any()
    mask_inf  = torch.isinf(observed_stats_tensor).any()
    mask_plch = (observed_stats_tensor == -999.0).all()
    if mask_nan or mask_inf or mask_plch:
        logging.warning(
            f"Skipping subject {subject_id}: all_placeholder={mask_plch}, any_nan={mask_nan}, any_inf={mask_inf}"
        )
        results['error'] = f'Invalid summary stats (all_placeholder={mask_plch}, any_nan={mask_nan}, any_inf={mask_inf})'
        return results

    # --- OPTIONAL: IMPUTE ISOLATED -999 BINS ---
    arr = observed_stats_tensor.cpu().numpy()
    mask = arr == -999.0
    if mask.any():
        # Use training stat means from checkpoint if available
        if hasattr(posterior_obj, 'checkpoint') and 'training_stat_means' in posterior_obj.checkpoint:
            col_means = globals()['np'].asarray(posterior_obj.checkpoint['training_stat_means'])
        elif hasattr(posterior_obj, 'training_stat_means'):
            col_means = globals()['np'].asarray(posterior_obj.training_stat_means)
        else:
            # Use our pre-defined reasonable defaults instead of zeros
            print("INFO: Using pre-defined DEFAULT_TRAINING_STAT_MEANS for imputation since they were not found in checkpoint.")
            col_means = DEFAULT_TRAINING_STAT_MEANS
        if col_means.shape != arr.shape:
            raise ValueError(f"training_stat_means shape {col_means.shape} does not match observed stats shape {arr.shape}")
        arr[mask] = col_means[mask]
        observed_stats_tensor = torch.tensor(arr, dtype=torch.float32, device=observed_stats_tensor.device)
        print(f"[DIAG] Imputed -999 bins for subject {subject_id} using checkpoint means.")

    # --- DIAGNOSTICS FOR POST-IMPUTATION STATS ---
    post_imputation_stats_np = observed_stats_tensor.cpu().numpy().flatten()
    print(f"[POST-IMPUTE DIAG] Subject {subject_id}: len={len(post_imputation_stats_np)} "
          f"min={globals()['np'].min(post_imputation_stats_np):.3f} max={globals()['np'].max(post_imputation_stats_np):.3f} "
          f"mean={globals()['np'].mean(post_imputation_stats_np):.3f} NaN={globals()['np'].isnan(post_imputation_stats_np).any()} "
          f"Inf={globals()['np'].isinf(post_imputation_stats_np).any()} "
          f"-999={(post_imputation_stats_np == -999.0).any()}")
    
    # Uncomment to print the full vector if a specific subject is highly problematic:
    if subject_id == '136': # Or whatever subject ID is problematic
        print(f"[POST-IMPUTE FULL] Subject {subject_id}: {post_imputation_stats_np.tolist()}")

    # FINAL CHECK for NaNs or Infs before sampling
    if torch.isnan(observed_stats_tensor).any() or torch.isinf(observed_stats_tensor).any():
        logging.error(f"Subject {subject_id}: NaNs or Infs detected in observed_stats_tensor AFTER IMPUTATION and FINAL CHECKS, before sampling. Stats: {observed_stats_tensor.cpu().numpy().flatten()}")
        # Return empty/NaN results for this subject to allow script to continue
        nan_results = {'subject_id': subject_id}
        for param_name in PARAM_NAMES_EMPIRICAL:
            nan_results[param_name] = np.nan
        if hasattr(posterior_obj, 'log_prob'): # Check if log_prob is available
            nan_results['log_probability'] = np.nan
        if 'mean_log_posterior_prob' in pd.DataFrame().columns: # A bit of a hack to check, improve if needed
             nan_results['mean_log_posterior_prob'] = np.nan
        # Ensure all expected columns are present, even if log_prob related ones are not always computed
        # This part needs to align with how results_df is constructed in the main loop
        # For now, returning None signals to skip adding to results dataframe or handle appropriately
        return None, None 

    if (observed_stats_tensor == -999.0).any():
        logging.warning(f"Subject {subject_id}: -999.0 values still present in observed_stats_tensor AFTER IMPUTATION attempt (e.g. training_stat_means might be incomplete or also -999). This may cause issues. Stats: {observed_stats_tensor.cpu().numpy().flatten()}")

    logging.info(f"Starting posterior sampling for subject {subject_id} with {num_posterior_samples} samples (timeout: {timeout_fit}s)...")

    import argparse
    import pandas as pd
    # Use global imports instead of local imports to avoid scope issues
    # We already have numpy as np and os imported at the top level
    from tqdm import tqdm
    # Removed unused SBI imports that were causing errors in version 0.23.3

    # For timeout mechanism
    import pebble
    from concurrent.futures import TimeoutError as PebbleTimeoutError # Alias to avoid confusion

    # Define a global timeout for sampling (e.g., 60 seconds)
    SAMPLING_TIMEOUT_SECONDS = timeout_fit # Adjust as needed
    LOG_PROB_TIMEOUT_SECONDS = timeout_logprob # Shorter timeout for log_prob

    # Suppress specific PyTensor warning for missing g++
    import pytensor

    posterior_samples_np = None
    log_probabilities_np = None
    actual_log_probs_calculated = False

    # We're now using the module-level helper functions for pickling compatibility

    with pebble.ProcessPool(max_workers=1) as pool:
        # Ensure observed_stats_tensor is on CPU before sending to another process if it's not already
        # It should be on CPU from the imputation step, but double check if issues arise
        # Also, ensure it has the batch dimension for SBI
        x_for_sbi = observed_stats_tensor.unsqueeze(0) if observed_stats_tensor.ndim == 1 else observed_stats_tensor
        
        sampling_future = pool.schedule(_perform_sampling_task, 
                                      args=(posterior_obj, num_posterior_samples, x_for_sbi), 
                                      timeout=timeout_fit)
        try:
            posterior_samples_torch = sampling_future.result()  # Blocks until completion or timeout
            posterior_samples_np = posterior_samples_torch.cpu().numpy()

            if hasattr(posterior_obj, 'log_prob') and posterior_samples_torch is not None:
                log_prob_future = pool.schedule(_perform_log_prob_task,
                                                args=(posterior_obj, posterior_samples_torch, x_for_sbi),
                                                timeout=timeout_logprob)
                try:
                    log_probabilities = log_prob_future.result()
                    log_probabilities_np = log_probabilities.cpu().numpy()
                    actual_log_probs_calculated = True
                except PebbleTimeoutError:
                    logging.warning(f"Subject {subject_id}: log_prob calculation timed out after {timeout_logprob} seconds.")
                    log_probabilities_np = np.full(num_posterior_samples, np.nan)
                except Exception as e_logprob:
                    logging.error(f"Subject {subject_id}: Error during log_prob calculation: {e_logprob}")
                    log_probabilities_np = np.full(num_posterior_samples, np.nan)
            else:
                log_probabilities_np = np.full(num_posterior_samples, np.nan)

        except PebbleTimeoutError:
            logging.error(f"Subject {subject_id}: Posterior sampling timed out after {timeout_fit} seconds. Skipping this subject.")
            return None, None # Indicates failure to the main loop
        except Exception as e:
            logging.error(f"Subject {subject_id}: An error occurred during posterior sampling task: {e}")
            return None, None # Indicates failure

    if posterior_samples_np is None:
        logging.warning(f"Subject {subject_id}: No posterior samples obtained (likely due to timeout or error). Returning NaN results.")
        # Create a DataFrame with NaNs for this subject
        nan_metrics = {f'{param}_mean': np.nan for param in PARAM_NAMES_EMPIRICAL}
        nan_metrics.update({f'{param}_median': np.nan for param in PARAM_NAMES_EMPIRICAL})
        nan_metrics.update({f'{param}_std': np.nan for param in PARAM_NAMES_EMPIRICAL})
        # Add subject_id and error message
        nan_metrics['subject_id'] = subject_id
        nan_metrics['error'] = 'No posterior samples obtained'
        return nan_metrics
        
    # If we get here, we have posterior samples - process them
    logging.info(f"Subject {subject_id}: Successfully obtained {posterior_samples_np.shape[0]} posterior samples. Computing metrics...")
    
    # Calculate statistics for each parameter dimension
    results = {'subject_id': subject_id}
    
    # Save posterior samples to CSV for later analysis if needed
    output_dir_path = Path(output_dir)
    posterior_samples_df = pd.DataFrame(posterior_samples_np, columns=param_names)
    posterior_samples_df.to_csv(output_dir_path / 'data' / f'subject_{subject_id}_posterior_samples.csv', index=False)
    
    # Compute parameter means, medians, std devs
    for i, param in enumerate(param_names):
        values = posterior_samples_np[:, i]
        results[f'{param}_mean'] = np.mean(values)
        results[f'{param}_median'] = np.median(values)
        results[f'{param}_std'] = np.std(values)
    
    # Add log probs if calculated
    if actual_log_probs_calculated:
        results['log_prob_mean'] = np.mean(log_probabilities_np)
        results['log_prob_median'] = np.median(log_probabilities_np)
    else:
        results['log_prob_mean'] = np.nan
        results['log_prob_median'] = np.nan
        
    logging.info(f"Subject {subject_id}: Fitting complete. Results summary:")
    for param in param_names:
        logging.info(f"  {param}: mean={results[f'{param}_mean']:.4f}, median={results[f'{param}_median']:.4f}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run empirical fitting using a loaded NPE.")
    parser.add_argument('--npe_file', type=str, required=True, help='Path to pre-trained NPE checkpoint (.pt file).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fitting results.')
    parser.add_argument('--n_subjects', type=int, default=0, help='Number of subjects to fit (0 for all).')
    parser.add_argument('--posterior_samples', type=int, default=1000, help='Number of posterior samples per subject.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--timeout_fit', type=int, default=60, help='Timeout for posterior sampling (fit) per subject in seconds.')
    parser.add_argument('--timeout_logprob', type=int, default=30, help='Timeout for log_prob calculation per subject in seconds.')
    args = parser.parse_args()

    # Setup
    # Configure basic logging FIRST to catch early issues
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'data').mkdir(exist_ok=True) # For individual posterior csvs
    (output_path / 'plots').mkdir(exist_ok=True)
    logging.info(f"Output directory prepared: {output_path}")

    prior_obj = BoxUniform(low=PRIOR_LOW.to(device), high=PRIOR_HIGH.to(device), device=device)

    # 1. Load Pre-trained NPE
    try:
        loaded_posterior_object = load_npe_from_checkpoint(Path(args.npe_file), prior_obj, device)
    except Exception as e:
        logging.error(f"Failed to load and prepare NPE from {args.npe_file}. Exiting. Error: {e}")
        return

    # --- Attempt to load and attach training_stat_means for imputation ---
    training_stat_means = None # Default to None
    npe_file_path = Path(args.npe_file)
    sbc_output_dir = npe_file_path.parent.parent 
    stat_means_file = sbc_output_dir / 'data' / 'training_stat_means.npy'

    if stat_means_file.exists():
        try:
            training_stat_means_np = np.load(stat_means_file)
            training_stat_means = torch.from_numpy(training_stat_means_np).float().to(device) 
            if hasattr(loaded_posterior_object, 'set_default_x'): # Check if this method exists
                 # For some SBI versions, this might be how you provide default x for imputation
                 # Or attach it as an attribute if that's how fit_single_subject expects it
                 pass # Placeholder for actual mechanism if needed
            # For now, we assume fit_single_subject will take training_stat_means as an argument
            logging.info(f"Successfully loaded training_stat_means from {stat_means_file}")
        except Exception as e_load_means:
            logging.warning(f"Found {stat_means_file} but failed to load or convert it: {e_load_means}. Imputation may use fallbacks.")
            training_stat_means = None # Ensure it's None if loading failed
    else:
        logging.warning(f"Training stat means file not found at {stat_means_file}. Imputation will use fallbacks.")
    # --- End of loading training_stat_means ---

    # 2. Load and Preprocess Empirical Data (Reverted to robust relative path logic)
    current_script_dir = Path(__file__).resolve().parent
    # Expected primary location: ../roberts_framing_data/ftp_osf_data.csv
    data_file = current_script_dir.parent / 'roberts_framing_data' / 'ftp_osf_data.csv'
    if not data_file.exists():
        # Fallback: ../../roberts_framing_data/ftp_osf_data.csv (if script is one level deeper)
        data_file = current_script_dir.parent.parent / 'roberts_framing_data' / 'ftp_osf_data.csv'
        if not data_file.exists():
            logging.error(f"Cannot find ftp_osf_data.csv. Tried: "
                          f"{current_script_dir.parent / 'roberts_framing_data'} and "
                          f"{current_script_dir.parent.parent / 'roberts_framing_data'}. Exiting.")
            return
            
    df_all_subjects = load_roberts_data_for_fitting(data_file)
    if df_all_subjects is None or df_all_subjects.empty:
        logging.error(f"Failed to load or processed data from {data_file} is empty. Exiting.")
        return
    logging.info(f"Successfully loaded empirical data from {data_file} for {df_all_subjects['subject'].nunique()} unique subjects.")

    # 3. Iterate and Fit Subjects
    subject_ids = df_all_subjects['subject'].unique()
    if args.n_subjects > 0:
        subject_ids = subject_ids[:args.n_subjects]
    logging.info(f"Starting fitting for {len(subject_ids)} subjects.")

    all_subject_results = []
    # Try to use tqdm if available for a progress bar
    try:
        from tqdm import tqdm
        subject_iter = tqdm(subject_ids, total=len(subject_ids), desc="Fitting subjects")
        use_tqdm = True
    except ImportError:
        subject_iter = subject_ids
        use_tqdm = False
        logging.info("tqdm not found, will log progress per subject.")

    for idx, subj_id in enumerate(subject_iter): # Changed to enumerate for index if not using tqdm's own enumeration
        if not use_tqdm:
            # Manually log progress if tqdm is not used
            logging.info(f"Processing subject {idx + 1}/{len(subject_ids)} (ID: {subj_id})")
        
        df_subj = df_all_subjects[df_all_subjects['subject'] == subj_id]
        
        subject_fit_results = fit_single_subject(
            subject_id=subj_id, 
            df_subject_data=df_subj, 
            posterior_obj=loaded_posterior_object, 
            num_posterior_samples=args.posterior_samples, 
            device=device, 
            param_names=PARAM_NAMES_EMPIRICAL,
            output_dir=args.output_dir,             # Pass explicitly
            timeout_fit=args.timeout_fit,          # Pass explicitly
            timeout_logprob=args.timeout_logprob,    # Pass explicitly
            training_stat_means=training_stat_means # Pass explicitly (was already there)
        )
        all_subject_results.append(subject_fit_results)

    # 4. Combine and Save Results
    df_final_results = pd.DataFrame(all_subject_results)
    df_final_results.to_csv(output_path / 'empirical_fitting_results.csv', index=False)
    logging.info(f"Results saved to {output_path / 'empirical_fitting_results.csv'}")

if __name__ == "__main__":
    main()