# Filename: empirical_fitting/run_ppc_nes_roberts.py
# Purpose: Perform Posterior Predictive Checks (PPCs) for the NES model 
#          fitted to the Roberts et al. (2022) empirical data.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Must be set before numpy/pytorch import
import concurrent.futures
import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

try:
    import pebble
    from pebble import ProcessPool
    has_pebble = True
except ImportError:
    has_pebble = False
    print("Pebble library not found. Timeout functionality will be disabled.")

# Define helper functions for multiprocessing (must be at module level for pickling)
def sample_from_posterior(posterior_obj, num_samples, obs_tensor):
    """Helper function for sampling from posterior (used with timeout)"""
    return posterior_obj.sample(
        (num_samples,), 
        x=obs_tensor.unsqueeze(0),
        show_progress_bars=False
    ).cpu()

def run_simulation(params_dict, subject_data, stat_keys):
    """Helper function for running simulations (used with timeout)"""
    return simulate_dataset_for_ppc(params_dict, subject_data, stat_keys)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- SBI & Custom Module Imports ---
try:
    from sbi.inference import SNPE # Only SNPE needed if loading a posterior object directly or density estimator
    from sbi.utils import BoxUniform
    # For PPC, we might not need sbi.analysis functions if doing manual plotting
except ImportError as e:
    logging.error(f"Critical SBI import error: {e}. Please ensure SBI is installed correctly.")
    sys.exit(1)

try:
    script_dir = Path(__file__).resolve().parent
    project_root_paths = [script_dir, script_dir.parent, script_dir.parent.parent]
    agent_mvnes_found = False
    for prp in project_root_paths:
        potential_src_dir = prp / 'src'
        if (potential_src_dir / 'agent_mvnes.py').exists():
            if str(potential_src_dir) not in sys.path:
                sys.path.insert(0, str(potential_src_dir))
            from agent_mvnes import MVNESAgent
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from: {potential_src_dir}")
            break
    if not agent_mvnes_found:
        if (Path('.') / 'agent_mvnes.py').exists(): # Fallback for current dir
            if str(Path('.')) not in sys.path: sys.path.insert(0, str(Path('.')))
            from agent_mvnes import MVNESAgent
            agent_mvnes_found = True
            logging.info(f"Found and imported MVNESAgent from current directory.")
        else:
            raise ImportError("Could not find agent_mvnes.py.")
except ImportError as e:
    logging.error(f"Error importing MVNESAgent: {e}. Check script location and 'src' directory.")
    sys.exit(1)

# --- Global Configurations & Constants (should match training/fitting) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s', force=True)
sbi_logger = logging.getLogger('sbi')
sbi_logger.setLevel(logging.WARNING)

# Modified for 5-parameter model (added alpha_gain)
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain']
PRIOR_LOW = torch.tensor([0.1,  0.5,  0.2,  0.05, 0.5]) # Must match prior of loaded NPE
PRIOR_HIGH = torch.tensor([2.0,  2.5,  1.5,  0.7, 1.0]) # Must match prior of loaded NPE

BASE_SIM_PARAMS = {
    'noise_std_dev': 1.0, 'dt': 0.01, 'max_time': 10.0, 'veto_flag': False
}
CONDITIONS_ROBERTS = {
    'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
    'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
}

# --- Core Functions (some can be imported from your fitting script if refactored) ---

def get_roberts_summary_stat_keys() -> List[str]:
    """Defines the keys for the summary statistics vector. MUST MATCH WHAT THE NPE WAS TRAINED WITH."""
    # This is the enhanced version with 60 statistics (including Gain vs Loss frame contrasts)
    # matching the version used to train the 5-parameter model
    keys = [
        'prop_gamble_overall', 'mean_rt_overall',
        'rt_q10_overall', 'rt_q50_overall', 'rt_q90_overall',
    ]
    for cond_name_key in CONDITIONS_ROBERTS.keys():
        keys.append(f"prop_gamble_{cond_name_key}")
        keys.append(f"mean_rt_{cond_name_key}")
        keys.append(f"rt_q10_{cond_name_key}")
        keys.append(f"rt_q50_{cond_name_key}")
        keys.append(f"rt_q90_{cond_name_key}")
        for bin_idx in range(5): 
            keys.append(f'rt_hist_bin{bin_idx}_{cond_name_key}')
            
    # Core framing effect stats
    keys.extend(['framing_effect_ntc', 'framing_effect_tc', 
                 'rt_framing_bias_ntc', 'rt_framing_bias_tc'])
    
    # RT distribution stats
    keys.append('rt_std_overall')
    for cond_name_key in CONDITIONS_ROBERTS.keys():
        keys.append(f'rt_std_{cond_name_key}')
    
    # Targeted Gain vs Loss frame contrasts - CRITICAL for 5-parameter model
    keys.extend([
        'mean_rt_Gain_vs_Loss_TC',   # RT contrast in TC condition
        'mean_rt_Gain_vs_Loss_NTC',  # RT contrast in NTC condition
        'rt_median_Gain_vs_Loss_TC',  # Median RT contrast in TC
        'rt_median_Gain_vs_Loss_NTC', # Median RT contrast in NTC
        'framing_effect_rt_gain',     # RT effect within Gain frame (TC vs NTC)
        'framing_effect_rt_loss'      # RT effect within Loss frame (TC vs NTC)
    ])
    
    # Verify we have exactly 60 statistics
    if len(keys) != 60:
        logging.warning(f"Expected 60 summary statistics but found {len(keys)}")
        
    logging.debug(f"Defined {len(keys)} summary statistics keys for PPC.")
    return keys

def calculate_summary_stats_for_ppc(df_trials: pd.DataFrame, stat_keys: List[str]) -> Dict[str, float]:
    """Calculates summary statistics for a single dataset (empirical or simulated)."""
    summaries = {}
    # Handle completely empty datasets
    if df_trials.empty: return {k: -999.0 for k in stat_keys}
    
    # Filter to valid trials if needed
    df_valid = df_trials.dropna(subset=['choice', 'rt'])
    if len(df_valid) == 0: return {k: -999.0 for k in stat_keys}
    
    # Overall stats across conditions
    choices = df_valid['choice']
    rts = df_valid['rt']
    prop_gamble = (choices == 1).mean() if 'choice' in df_valid.columns else -999.0
    summaries['prop_gamble_overall'] = prop_gamble
    summaries['mean_rt_overall'] = rts.mean()
    try:
        q = rts.quantile([0.1,0.5,0.9])
        summaries['rt_q10_overall']=q.get(0.1,-999.0); summaries['rt_q50_overall']=q.get(0.5,-999.0); summaries['rt_q90_overall']=q.get(0.9,-999.0)
    except: pass
    
    # Per-condition stats
    cond_props, cond_rts_mean, cond_rts_std, cond_rts_median = {}, {}, {}, {}
    for cond_key, cond_info in CONDITIONS_ROBERTS.items():
        cond_key_enum = cond_key # e.g. 'Gain_TC'
        cond_mask = (df_valid['frame'] == cond_info['frame']) & (df_valid['cond'] == cond_info['cond'])
        choices_cond = df_valid.loc[cond_mask, 'choice'] if 'choice' in df_valid.columns else pd.Series([])
        rts_cond = df_valid.loc[cond_mask, 'rt']
        
        if not choices_cond.empty:
            prop_gamble_cond = (choices_cond == 1).mean()
            summaries[f'prop_gamble_{cond_key_enum}'] = prop_gamble_cond
            cond_props[cond_key_enum] = prop_gamble_cond
        
        if not rts_cond.empty:
            # Calculate and store mean RTs per condition
            summaries[f'mean_rt_{cond_key_enum}'] = rts_cond.mean() 
            cond_rts_mean[cond_key_enum] = rts_cond.mean()
            summaries[f'rt_std_{cond_key_enum}'] = rts_cond.std() 
            cond_rts_std[cond_key_enum] = rts_cond.std()
            
            # Calculate and store median RTs per condition
            median_rt = rts_cond.median()
            cond_rts_median[cond_key_enum] = median_rt
            
            try:
                q_c = rts_cond.quantile([0.1,0.5,0.9])
                summaries[f'rt_q10_{cond_key_enum}']=q_c.get(0.1,-999.0) 
                summaries[f'rt_q50_{cond_key_enum}']=q_c.get(0.5,-999.0) 
                summaries[f'rt_q90_{cond_key_enum}']=q_c.get(0.9,-999.0)
            except: pass
            
            max_rt = 1.0 if 'TC' in cond_key_enum else 3.0 
            edges = np.linspace(0,max_rt,6)
            if len(rts_cond) >= 1:
                hist, _ = np.histogram(rts_cond.clip(0,max_rt), bins=edges, density=True) 
                for i, h in enumerate(hist):
                    summaries[f'rt_hist_bin{i}_{cond_key_enum}'] = h
    
    # Framing effects (choice proportions)
    pg_ln = cond_props.get('Loss_NTC', np.nan)
    pg_gn = cond_props.get('Gain_NTC', np.nan) 
    summaries['framing_effect_ntc'] = pg_ln - pg_gn if not(pd.isna(pg_ln) or pd.isna(pg_gn)) else -999.0
    
    pg_lt = cond_props.get('Loss_TC', np.nan)
    pg_gt = cond_props.get('Gain_TC', np.nan) 
    summaries['framing_effect_tc'] = pg_lt - pg_gt if not(pd.isna(pg_lt) or pd.isna(pg_gt)) else -999.0
    
    # RT framing effects (mean RT differences between loss and gain)
    rt_ln = cond_rts_mean.get('Loss_NTC', np.nan)
    rt_gn = cond_rts_mean.get('Gain_NTC', np.nan) 
    summaries['rt_framing_bias_ntc'] = rt_ln - rt_gn if not(pd.isna(rt_ln) or pd.isna(rt_gn)) else -999.0
    
    rt_lt = cond_rts_mean.get('Loss_TC', np.nan)
    rt_gt = cond_rts_mean.get('Gain_TC', np.nan) 
    summaries['rt_framing_bias_tc'] = rt_lt - rt_gt if not(pd.isna(rt_lt) or pd.isna(rt_gt)) else -999.0
    
    # NEW: Mean RT contrasts for Gain vs Loss (added for 5-parameter model)
    mrt_gtc = cond_rts_mean.get('Gain_TC', np.nan) 
    mrt_ltc = cond_rts_mean.get('Loss_TC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_TC'] = mrt_gtc - mrt_ltc if not (pd.isna(mrt_gtc) or pd.isna(mrt_ltc)) else -999.0

    mrt_gntc = cond_rts_mean.get('Gain_NTC', np.nan) 
    mrt_lntc = cond_rts_mean.get('Loss_NTC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_NTC'] = mrt_gntc - mrt_lntc if not (pd.isna(mrt_gntc) or pd.isna(mrt_lntc)) else -999.0

    # NEW: Median RT contrasts (based on stored median values)
    medrt_gtc = cond_rts_median.get('Gain_TC', np.nan) 
    medrt_ltc = cond_rts_median.get('Loss_TC', np.nan)
    summaries['rt_median_Gain_vs_Loss_TC'] = medrt_gtc - medrt_ltc if not (pd.isna(medrt_gtc) or pd.isna(medrt_ltc)) else -999.0

    medrt_gntc = cond_rts_median.get('Gain_NTC', np.nan) 
    medrt_lntc = cond_rts_median.get('Loss_NTC', np.nan)
    summaries['rt_median_Gain_vs_Loss_NTC'] = medrt_gntc - medrt_lntc if not (pd.isna(medrt_gntc) or pd.isna(medrt_lntc)) else -999.0

    # NEW: RT effects within frames (TC vs NTC mean RTs)
    summaries['framing_effect_rt_gain'] = mrt_gtc - mrt_gntc if not (pd.isna(mrt_gtc) or pd.isna(mrt_gntc)) else -999.0
    summaries['framing_effect_rt_loss'] = mrt_ltc - mrt_lntc if not (pd.isna(mrt_ltc) or pd.isna(mrt_lntc)) else -999.0
    
    return {k: summaries.get(k, -999.0) if not pd.isna(summaries.get(k, -999.0)) else -999.0 for k in stat_keys}


def simulate_dataset_for_ppc(
    params_dict: Dict[str, float], 
    subject_trial_structure: Any,  # numpy record array for speed
    stat_keys: List[str],
    agent: Any = None
) -> Dict[str, float]:
    """Simulates one full dataset for a subject given one parameter set and their trial structure."""
    if agent is None:
        agent = MVNESAgent(config={})
    sim_results_list = []

    for trial_info in subject_trial_structure:
        salience_input = trial_info['prob']
        norm_input = 1.0 if trial_info['is_gain_frame'] else -1.0
        agent_run_params = {
            'w_n': params_dict['v_norm'], 'threshold_a': params_dict['a_0'],
            'w_s': params_dict['w_s_eff'], 't': params_dict['t_0'],
            'alpha_gain': params_dict.get('alpha_gain', 1.0),
            **BASE_SIM_PARAMS
        }
        try:
            # Always run the DDM simulation, let run_mvnes_trial handle all drift rates
            trial_output = agent.run_mvnes_trial(salience_input, norm_input, agent_run_params)
            sim_rt = trial_output.get('rt', np.nan)
            sim_choice = trial_output.get('choice', np.nan)
            trace = trial_output.get('trace', [np.nan])
            timeout = trial_output.get('timeout', False)
            # Clamp RT if time constrained
            if not pd.isna(sim_rt) and trial_info['time_constrained']:
                sim_rt = min(sim_rt, 1.0)
            sim_results_list.append({
                'rt': sim_rt, 'choice': sim_choice,
                'frame': trial_info['frame'], 'cond': trial_info['cond'], 'prob': trial_info['prob']
            })
        except Exception:
            sim_results_list.append({'rt': np.nan, 'choice': np.nan, 'frame': trial_info['frame'], 'cond': trial_info['cond'], 'prob': trial_info['prob']})
    df_sim_dataset = pd.DataFrame(sim_results_list)
    return calculate_summary_stats_for_ppc(df_sim_dataset, stat_keys)


def load_npe_posterior_object(npe_file_path: Path, prior: BoxUniform, device: str) -> Any:
    """Loads a pre-trained NPE density estimator and builds a posterior object."""
    logging.info(f"Loading pre-trained NPE from: {npe_file_path}")
    checkpoint = torch.load(npe_file_path, map_location=device) # Add weights_only=True for safety if applicable
    if not isinstance(checkpoint, dict) or 'density_estimator_state_dict' not in checkpoint:
        raise ValueError("NPE checkpoint error: Expected dict with 'density_estimator_state_dict'.")

    num_summary_stats_trained = checkpoint.get('num_summary_stats')
    if num_summary_stats_trained is None:
        raise ValueError("'num_summary_stats' missing from checkpoint. Cannot verify consistency.")
    
    current_script_num_stats = len(get_roberts_summary_stat_keys())
    if current_script_num_stats != num_summary_stats_trained:
        logging.error(f"CRITICAL MISMATCH: Loaded NPE expects {num_summary_stats_trained} summary stats, "
                      f"but current script defines {current_script_num_stats}. Adjust get_roberts_summary_stat_keys().")
        raise ValueError("Summary statistic dimension mismatch.")

    inference_algorithm = SNPE(prior=prior, density_estimator='maf', device=device)
    batch_theta = prior.sample((2,))
    batch_x = torch.randn(2, num_summary_stats_trained, device=device)
    density_estimator_net = inference_algorithm._build_neural_net(batch_theta, batch_x)
    density_estimator_net.load_state_dict(checkpoint['density_estimator_state_dict'], strict=False)
    posterior_object = inference_algorithm.build_posterior(density_estimator_net)
    logging.info(f"NPE loaded and posterior object built. Trained with {checkpoint.get('npe_train_sims','N/A')} sims.")
    return posterior_object

# --- Main PPC Function ---
def run_ppc_for_subject(
    subject_id: Any,
    df_subject_empirical_data: pd.DataFrame,
    fitted_params_mean: Dict[str, float],
    npe_posterior_obj: Any,
    observed_subject_summary_stats_tensor: torch.Tensor,
    num_ppc_simulations: int,
    device: str,
    output_plots_dir: Path
):
    logging.info(f"--- Starting PPC for Subject {subject_id} ---")
    stat_keys = get_roberts_summary_stat_keys()
    # 1. Get actual observed summary statistics for this subject
    actual_obs_stats = calculate_summary_stats_for_ppc(df_subject_empirical_data, stat_keys)
    # 2. Prepare trial structure as numpy record array for simulation speed
    trial_struct = df_subject_empirical_data[['prob','is_gain_frame','frame','cond','time_constrained']].to_records(index=False)
    # 3. Only sample as many posterior parameter sets as num_ppc_simulations
    logging.info(f"Drawing {num_ppc_simulations} posterior parameter samples for subject {subject_id}...")
    x_condition = observed_subject_summary_stats_tensor.unsqueeze(0) if observed_subject_summary_stats_tensor.dim() == 1 else observed_subject_summary_stats_tensor
    x_condition = x_condition.to(device)
    subject_posterior_param_samples = npe_posterior_obj.sample(
        (num_ppc_simulations,),
        x=x_condition,
        show_progress_bars=False
    ).cpu()
    # 4. Reuse a single agent for all PPC sims
    agent = MVNESAgent(config={})
    logging.info(f"Simulating {num_ppc_simulations} datasets for PPC using posterior parameter samples...")
    ppc_simulated_stats_list = []
    for i in range(num_ppc_simulations):
        if (i + 1) % (num_ppc_simulations // 10 or 1) == 0:
            logging.info(f"  PPC simulation {i+1}/{num_ppc_simulations}")
        selected_params_tensor = subject_posterior_param_samples[i]
        params_dict_for_sim = {name: val.item() for name, val in zip(PARAM_NAMES, selected_params_tensor)}
        sim_stats = simulate_dataset_for_ppc(params_dict_for_sim, trial_struct, stat_keys, agent=agent)
        ppc_simulated_stats_list.append(sim_stats)
    df_ppc_stats = pd.DataFrame(ppc_simulated_stats_list)
    # 5. Plotting (unchanged)
    plot_stat_keys = []
    for cond_name_key in CONDITIONS_ROBERTS.keys():
        plot_stat_keys.append(f"prop_gamble_{cond_name_key}")
        plot_stat_keys.append(f"mean_rt_{cond_name_key}")
    plot_stat_keys.extend(['framing_effect_ntc', 'framing_effect_tc'])
    num_plot_stats = len(plot_stat_keys)
    n_cols = 3
    n_rows = (num_plot_stats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    for i, key in enumerate(plot_stat_keys):
        ax = axes[i]
        sim_values = df_ppc_stats[key].dropna()
        if not sim_values.empty:
            sns.histplot(sim_values, ax=ax, kde=True, stat="density", label="Simulated")
            actual_val = actual_obs_stats.get(key)
            if actual_val is not None and not pd.isna(actual_val) and actual_val != -999.0 :
                ax.axvline(actual_val, color='r', linestyle='--', label=f"Observed: {actual_val:.3f}")
            ax.set_title(key.replace("_", " ").title())
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No sim data", ha='center', va='center')
            ax.set_title(key.replace("_", " ").title())
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Posterior Predictive Checks - Subject {subject_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_plots_dir / f"ppc_subject_{subject_id}.png")
    plt.close(fig)
    logging.info(f"PPC plot saved for subject {subject_id}")
    # Return a summary for a table
    ppc_summary = {'subject_id': subject_id}
    for key in plot_stat_keys:
        ppc_summary[f'obs_{key}'] = actual_obs_stats.get(key)
        sim_vals = df_ppc_stats[key].dropna()
        ppc_summary[f'sim_mean_{key}'] = sim_vals.mean() if not sim_vals.empty else np.nan
        ppc_summary[f'sim_median_{key}'] = sim_vals.median() if not sim_vals.empty else np.nan
        if not sim_vals.empty and actual_obs_stats.get(key) is not None and not pd.isna(actual_obs_stats.get(key)):
            obs_val = actual_obs_stats.get(key)
            p_val = np.mean(np.abs(sim_vals - sim_vals.mean()) >= np.abs(obs_val - sim_vals.mean()))
            ppc_summary[f'sim_pval_{key}'] = p_val
    return ppc_summary


# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPCs for NES model on Roberts et al. data.")
    parser.add_argument('--npe_file', type=str, required=True, help='Path to PRE-TRAINED NPE checkpoint (.pt file).')
    parser.add_argument('--fitted_params_file', type=str, required=True, help='Path to CSV file with fitted empirical parameters (e.g., from empirical fitting run).')
    parser.add_argument('--roberts_data_file', type=str, default="./roberts_framing_data/ftp_osf_data.csv", help='Path to the Roberts data CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save PPC results and plots.')
    parser.add_argument('--subject_ids', type=str, default=None, help='Comma-separated list of subject IDs for PPC (e.g., "114,122,165"). Default: 3-5 representative subjects.')
    parser.add_argument('--num_ppc_simulations', type=int, default=200, help='Number of datasets to simulate from posterior for PPC.')
    parser.add_argument('--num_posterior_samples_for_ppc', type=int, default=500, help='Number of parameter sets to draw from each subject posterior for PPC sims.')
    parser.add_argument('--seed', type=int, default=int(time.time()))
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--quantify_coverage', action='store_true', help='If set, quantify PPC coverage for all summary stats and all fitted subjects.')
    parser.add_argument('--timeout_subject', type=int, default=600, help='Timeout in seconds for processing each subject (default: 600s = 10min)')
    args = parser.parse_args()

    if args.quantify_coverage:
        # --- PPC Coverage Quantification Routine ---
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
        logging.info(f"Using device: {device}. Seed: {args.seed}")

        output_path = Path(args.output_dir)
        output_path_data = output_path / 'data'; output_path_data.mkdir(parents=True, exist_ok=True)
        # Load prior, NPE, fitted params, empirical data
        sbi_prior_for_loading = BoxUniform(low=PRIOR_LOW.to(device), high=PRIOR_HIGH.to(device), device=device)
        loaded_npe_posterior_object = load_npe_posterior_object(Path(args.npe_file), sbi_prior_for_loading, device)
        df_fitted_params = pd.read_csv(args.fitted_params_file)
        df_all_empirical_data = pd.read_csv(args.roberts_data_file)
        if 'trialType' in df_all_empirical_data.columns:
            df_all_empirical_data = df_all_empirical_data[df_all_empirical_data['trialType'] == 'target'].copy()
        if 'rt' in df_all_empirical_data.columns:
            df_all_empirical_data['rt'] = pd.to_numeric(df_all_empirical_data['rt'], errors='coerce')
        if 'choice' in df_all_empirical_data.columns:
            df_all_empirical_data['choice'] = pd.to_numeric(df_all_empirical_data['choice'], errors='coerce')
        if 'prob' in df_all_empirical_data.columns:
            df_all_empirical_data['prob'] = pd.to_numeric(df_all_empirical_data['prob'], errors='coerce')
        df_all_empirical_data['time_constrained'] = df_all_empirical_data['cond'] == 'tc'
        df_all_empirical_data['is_gain_frame'] = df_all_empirical_data['frame'] == 'gain'
        stat_keys = get_roberts_summary_stat_keys()
        coverage_records = {k: [] for k in stat_keys}
        n_subjects = 0
        for idx, subj_row in tqdm(df_fitted_params.iterrows(), total=len(df_fitted_params), desc="Coverage: Subjects"):
            subj_id = subj_row['subject_id'] if 'subject_id' in subj_row else subj_row['subject']
            try:
                subj_id_int = int(subj_id)
            except:
                subj_id_int = subj_id
            df_subj = df_all_empirical_data[df_all_empirical_data['subject'] == subj_id_int]
            if df_subj.empty:
                continue
            obs_stats = calculate_summary_stats_for_ppc(df_subj, stat_keys)
            obs_stats_tensor = torch.tensor([obs_stats[k] for k in stat_keys], dtype=torch.float32, device=device)
            # Convert subject trials to numpy record array ONCE for all PPC sims
            subj_trial_struct = df_subj[['prob','is_gain_frame','frame','cond','time_constrained']].to_records(index=False)
            # Process subjects with timeout protection
            if has_pebble:
                try:
                    # Create a process pool with timeout
                    with ProcessPool(max_workers=1) as pool:
                        # Sample from posterior with timeout
                        try:
                            future = pool.schedule(
                                sample_from_posterior,
                                args=(loaded_npe_posterior_object, args.num_posterior_samples_for_ppc, obs_stats_tensor),
                                timeout=args.timeout_subject
                            )
                            posterior_samples = future.result()
                        except (TimeoutError, concurrent.futures.TimeoutError):
                            logging.warning(f"Timeout during posterior sampling for subject {subj_id}. obs_stats_tensor: {obs_stats_tensor.tolist()}")
                            # Insert placeholder stats for all PPC sims for this subject
                            sim_stats = [{k: -999.0 for k in stat_keys} for _ in range(args.num_ppc_simulations)]
                            df_sim_stats = pd.DataFrame(sim_stats)
                            for k in stat_keys:
                                coverage_records[k].append({'covered_90': np.nan, 'covered_95': np.nan})
                            n_subjects += 1
                            continue
                        except Exception as e:
                            logging.error(f"Error during posterior sampling for subject {subj_id}: {e}\nobs_stats_tensor: {obs_stats_tensor.tolist()}")
                            sim_stats = [{k: -999.0 for k in stat_keys} for _ in range(args.num_ppc_simulations)]
                            df_sim_stats = pd.DataFrame(sim_stats)
                            for k in stat_keys:
                                coverage_records[k].append({'covered_90': np.nan, 'covered_95': np.nan})
                            n_subjects += 1
                            continue
                        # Simulate PPC datasets with timeout for each
                        from tqdm import tqdm as tqdm_inner
                        sim_stats = []
                        for i in tqdm_inner(range(args.num_ppc_simulations), desc=f"Subject {subj_id} PPC Sims", leave=False):
                            param_sample_idx = torch.randint(len(posterior_samples), (1,)).item()
                            params_dict_for_sim = {name: val.item() for name, val in zip(PARAM_NAMES, posterior_samples[param_sample_idx])}
                            try:
                                future = pool.schedule(
                                    run_simulation,
                                    args=(params_dict_for_sim, subj_trial_struct, stat_keys),
                                    timeout=max(30, args.timeout_subject // args.num_ppc_simulations)
                                )
                                sim_result = future.result()
                                sim_stats.append(sim_result)
                            except (TimeoutError, concurrent.futures.TimeoutError):
                                logging.warning(f"Timeout during PPC simulation for subject {subj_id}, params: {params_dict_for_sim}")
                                sim_stats.append({k: -999.0 for k in stat_keys})
                            except Exception as e:
                                logging.error(f"Error during PPC simulation for subject {subj_id}, params: {params_dict_for_sim}: {e}")
                                sim_stats.append({k: -999.0 for k in stat_keys})
                        df_sim_stats = pd.DataFrame(sim_stats)
                except Exception as e:
                    logging.error(f"Fatal error in ProcessPool for subject {subj_id}: {e}")
                    sim_stats = [{k: -999.0 for k in stat_keys} for _ in range(args.num_ppc_simulations)]
                    df_sim_stats = pd.DataFrame(sim_stats)
                    for k in stat_keys:
                        coverage_records[k].append({'covered_90': np.nan, 'covered_95': np.nan})
                    n_subjects += 1
                    continue
            else:
                # Fall back to original code without timeout protection
                posterior_samples = loaded_npe_posterior_object.sample(
                    (args.num_posterior_samples_for_ppc,), x=obs_stats_tensor.unsqueeze(0), show_progress_bars=False
                ).cpu()
                from tqdm import tqdm as tqdm_inner
                sim_stats = []
                for i in tqdm_inner(range(args.num_ppc_simulations), desc=f"Subject {subj_id} PPC Sims", leave=False):
                    param_sample_idx = torch.randint(len(posterior_samples), (1,)).item()
                    params_dict_for_sim = {name: val.item() for name, val in zip(PARAM_NAMES, posterior_samples[param_sample_idx])}
                    try:
                        sim_stats.append(simulate_dataset_for_ppc(params_dict_for_sim, subj_trial_struct, stat_keys))
                    except Exception as e:
                        logging.error(f"Error in fallback PPC simulation for subject {subj_id}, params: {params_dict_for_sim}: {e}")
                        sim_stats.append({k: -999.0 for k in stat_keys})
                df_sim_stats = pd.DataFrame(sim_stats)
            for k in stat_keys:
                sim_vals = df_sim_stats[k].dropna()
                if sim_vals.empty or obs_stats[k] == -999.0 or pd.isna(obs_stats[k]):
                    coverage_records[k].append({'covered_90': np.nan, 'covered_95': np.nan})
                    continue
                q05, q95 = np.percentile(sim_vals, [5, 95])
                q025, q975 = np.percentile(sim_vals, [2.5, 97.5])
                covered_90 = (obs_stats[k] >= q05) and (obs_stats[k] <= q95)
                covered_95 = (obs_stats[k] >= q025) and (obs_stats[k] <= q975)
                coverage_records[k].append({'covered_90': covered_90, 'covered_95': covered_95})
            n_subjects += 1
        # Summarize coverage
        rows = []
        for k in stat_keys:
            vals = [rec for rec in coverage_records[k] if rec['covered_90'] is not np.nan]
            n_cov_90 = sum([rec['covered_90'] for rec in vals if rec['covered_90'] is not np.nan])
            n_cov_95 = sum([rec['covered_95'] for rec in vals if rec['covered_95'] is not np.nan])
            n = len(vals)
            pct_90 = 100.0 * n_cov_90 / n if n > 0 else np.nan
            pct_95 = 100.0 * n_cov_95 / n if n > 0 else np.nan
            rows.append({'stat_key': k, 'n_subjects': n, 'n_covered_90': n_cov_90, 'pct_covered_90': pct_90, 'n_covered_95': n_cov_95, 'pct_covered_95': pct_95})
        df_cov = pd.DataFrame(rows)
        out_cov_path = output_path_data / f"ppc_coverage_summary_seed{args.seed}.csv"
        df_cov.to_csv(out_cov_path, index=False)
        print(f"PPC coverage summary saved to: {out_cov_path}")
        logging.info(f"PPC coverage summary saved to: {out_cov_path}")
        sys.exit(0)
        pass

    # Setup
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}. Seed: {args.seed}")

    output_path = Path(args.output_dir)
    output_path_data = output_path / 'data'; output_path_data.mkdir(parents=True, exist_ok=True)
    output_path_plots = output_path / 'plots'; output_path_plots.mkdir(parents=True, exist_ok=True)
    
    # Define prior on the correct device (must match the prior used for training the loaded NPE)
    sbi_prior_for_loading = BoxUniform(low=PRIOR_LOW.to(device), high=PRIOR_HIGH.to(device), device=device)

    # 1. Load Pre-trained NPE
    try:
        loaded_npe_posterior_object = load_npe_posterior_object(Path(args.npe_file), sbi_prior_for_loading, device)
    except Exception as e:
        logging.error(f"Failed to load NPE. Exiting. Error: {e}", exc_info=True)
        sys.exit(1)

    # 2. Load Empirical Data and Fitted Parameters
    df_all_empirical_data = pd.read_csv(args.roberts_data_file)
    # Filter to target trials and basic preprocessing
    if 'trialType' in df_all_empirical_data.columns:
        df_all_empirical_data = df_all_empirical_data[df_all_empirical_data['trialType'] == 'target'].copy()
    if 'rt' in df_all_empirical_data.columns:
        df_all_empirical_data['rt'] = pd.to_numeric(df_all_empirical_data['rt'], errors='coerce')
    if 'choice' in df_all_empirical_data.columns:
        df_all_empirical_data['choice'] = pd.to_numeric(df_all_empirical_data['choice'], errors='coerce')
    if 'prob' in df_all_empirical_data.columns:
        df_all_empirical_data['prob'] = pd.to_numeric(df_all_empirical_data['prob'], errors='coerce')
    df_all_empirical_data['time_constrained'] = df_all_empirical_data['cond'] == 'tc'
    df_all_empirical_data['is_gain_frame'] = df_all_empirical_data['frame'] == 'gain'
    try:
        df_fitted_params = pd.read_csv(args.fitted_params_file)
    except FileNotFoundError:
        logging.error(f"Fitted parameters file not found: {args.fitted_params_file}"); sys.exit(1)

    # 3. Select Subjects for PPC
    if args.subject_ids:
        subject_ids_for_ppc = [s.strip() for s in args.subject_ids.split(',')]
        # Convert to int if your subject IDs in df_all_empirical_data are integers
        try:
            subject_ids_for_ppc = [int(sid) for sid in subject_ids_for_ppc]
        except ValueError:
            logging.warning("Subject IDs provided are not all integers. Using as strings if that matches data.")
    else: # Select a few representative subjects if none specified
        # Example: one low, one mid, one high framing effect if 'framing_effect_avg_obs' is in df_fitted_params
        if 'framing_effect_avg_obs' in df_fitted_params.columns and not df_fitted_params['framing_effect_avg_obs'].dropna().empty:
            df_sorted_by_fe = df_fitted_params.dropna(subset=['framing_effect_avg_obs']).sort_values(by='framing_effect_avg_obs')
            num_subjects_total = len(df_sorted_by_fe)
            if num_subjects_total >= 3:
                subject_ids_for_ppc = [
                    df_sorted_by_fe.iloc[0]['subject_id'],
                    df_sorted_by_fe.iloc[num_subjects_total // 2]['subject_id'],
                    df_sorted_by_fe.iloc[-1]['subject_id']
                ]
            elif num_subjects_total > 0:
                 subject_ids_for_ppc = [df_sorted_by_fe.iloc[0]['subject_id']]
            else:
                 subject_ids_for_ppc = [df_all_empirical_data['subject'].unique()[0]] if not df_all_empirical_data.empty else []
        else: # Fallback if framing effect not available in fitted params file
            unique_subj = df_all_empirical_data['subject'].unique()
            subject_ids_for_ppc = unique_subj[:min(3, len(unique_subj))] if len(unique_subj)>0 else []
        logging.info(f"No specific subject IDs provided. Selected representative subjects: {subject_ids_for_ppc}")
    
    if not subject_ids_for_ppc:
        logging.error("No subjects selected or available for PPC. Exiting.")
        sys.exit(1)

    # 4. Run PPCs
    all_ppc_summaries = []
    stat_keys_for_calc = get_roberts_summary_stat_keys() # Get once

    for subj_id in subject_ids_for_ppc:
        df_subj_emp_data = df_all_empirical_data[df_all_empirical_data['subject'] == subj_id]
        if df_subj_emp_data.empty:
            logging.warning(f"No empirical data for subject {subj_id} in loaded file. Skipping PPC."); continue
        # Get this subject's observed summary statistics (needed to draw from their posterior)
        subj_obs_stats_dict = calculate_summary_stats_for_ppc(df_subj_emp_data, stat_keys_for_calc)
        subj_obs_stats_vector = [subj_obs_stats_dict.get(k, -999.0) for k in stat_keys_for_calc]
        subj_obs_stats_tensor = torch.tensor(subj_obs_stats_vector, dtype=torch.float32)
        # Get mean fitted parameters (not directly used for sampling, but could be for reference)
        subj_fitted_row = df_fitted_params[df_fitted_params['subject_id'] == subj_id]
        if subj_fitted_row.empty:
            logging.warning(f"No fitted parameters found for subject {subj_id}. Skipping PPC."); continue
        mean_params = {}
        for p in PARAM_NAMES:
            if f"mean_{p}" in subj_fitted_row.columns:
                mean_params[p] = subj_fitted_row.iloc[0][f"mean_{p}"]
            elif p in subj_fitted_row.columns:
                mean_params[p] = subj_fitted_row.iloc[0][p]
            elif f"{p}_mean" in subj_fitted_row.columns:
                mean_params[p] = subj_fitted_row.iloc[0][f"{p}_mean"]
            else:
                raise KeyError(f"Could not find parameter column for '{p}' (tried 'mean_{p}', '{p}', and '{p}_mean') in fitted params file")
        ppc_summary = run_ppc_for_subject(
            subj_id, df_subj_emp_data, mean_params,
            loaded_npe_posterior_object,
            subj_obs_stats_tensor,
            args.num_ppc_simulations,
            device, output_path_plots
        )
        all_ppc_summaries.append(ppc_summary)

    # 5. Save PPC Summary Table
    if all_ppc_summaries:
        df_ppc_summary_table = pd.DataFrame(all_ppc_summaries)
        ppc_summary_path = output_path_data / f"ppc_summary_table_seed{args.seed}.csv"
        df_ppc_summary_table.to_csv(ppc_summary_path, index=False, float_format='%.4f')
        logging.info(f"PPC summary table saved to: {ppc_summary_path}")

    logging.info("PPC script finished.")