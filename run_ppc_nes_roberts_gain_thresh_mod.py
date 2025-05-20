# This script is a modified version of run_ppc_nes_roberts.py
# It implements the advisor's suggestion: for Gain-frame trials, threshold a_0 is multiplied by 0.75
# All other logic is unchanged except for output naming and subject selection

# ---- BEGIN COPY ----
import argparse
import logging
import sys
import os
from pathlib import Path
import time
import traceback
import json
from typing import Dict, List, Any, Optional

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
except ImportError as e:
    logging.error(f"Critical SBI import error: {e}. Please ensure SBI is installed correctly.")
    sys.exit(1)

# --- Global Configurations & Constants (should match training/fitting) ---
PARAM_NAMES = ['v_norm', 'a_0', 'w_s_eff', 't_0']
PRIOR_LOW = torch.tensor([0.1,  0.5,  0.2,  0.05]) # Must match prior of loaded NPE
PRIOR_HIGH = torch.tensor([2.0,  2.5,  1.5,  0.7]) # Must match prior of loaded NPE
BASE_SIM_PARAMS = {
    'noise_std_dev': 1.0, 'dt': 0.01, 'max_time': 10.0, 'veto_flag': False
}
CONDITIONS_ROBERTS = {
    'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
    'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
}

# Import the actual MVNESAgent from the src directory
import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
from agent_mvnes import MVNESAgent

def get_roberts_summary_stat_keys() -> List[str]:
    # THIS MUST MATCH THE SUMMARY STATS USED TO TRAIN THE LOADED NPE
    keys = [
        'prop_gamble_overall', 'mean_rt_overall',
        'rt_q10_overall', 'rt_q50_overall', 'rt_q90_overall', 'rt_std_overall'
    ]
    for cond_name_key in CONDITIONS_ROBERTS.keys():
        keys.append(f"prop_gamble_{cond_name_key}")
        keys.append(f"mean_rt_{cond_name_key}")
        keys.append(f"rt_std_{cond_name_key}")
        keys.append(f"rt_q10_{cond_name_key}")
        keys.append(f"rt_q50_{cond_name_key}")
        keys.append(f"rt_q90_{cond_name_key}")
        for bin_idx in range(5): 
            keys.append(f'rt_hist_bin{bin_idx}_{cond_name_key}')
    keys.extend(['framing_effect_ntc', 'framing_effect_tc', 
                 'rt_framing_bias_ntc', 'rt_framing_bias_tc'])
    return keys

def calculate_summary_stats_roberts(df_trials: pd.DataFrame, 
                                   stat_keys: List[str],
                                   impute_rt_means: Optional[Dict[str, float]] = None
                                   ) -> Dict[str, float]:
    """Calculates summary statistics from trial data. Adapted from fit_nes_to_roberts_data_sbc_focused.py."""
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
        CONDITIONS_ROBERTS = {
            'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
            'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
        }
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
        except Exception:
            summaries['rt_q10_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q50_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
            summaries['rt_q90_overall'] = rts_overall.iloc[0] if not rts_overall.empty else -999.0
    
    CONDITIONS_ROBERTS = {
        'Gain_NTC': {'frame': 'gain', 'cond': 'ntc'}, 'Gain_TC': {'frame': 'gain', 'cond': 'tc'},
        'Loss_NTC': {'frame': 'loss', 'cond': 'ntc'}, 'Loss_TC': {'frame': 'loss', 'cond': 'tc'},
    }
    cond_props = {}
    cond_rts_mean = {}
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
                summaries[f'mean_rt_{cond_key_enum}'] = rts_cond.mean()
                cond_rts_mean[cond_key_enum] = rts_cond.mean()
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
    
    pg_ln = cond_props.get('Loss_NTC', np.nan); pg_gn = cond_props.get('Gain_NTC', np.nan)
    summaries['framing_effect_ntc'] = pg_ln - pg_gn if not (pd.isna(pg_ln) or pd.isna(pg_gn)) else -999.0
    
    pg_lt = cond_props.get('Loss_TC', np.nan); pg_gt = cond_props.get('Gain_TC', np.nan)
    summaries['framing_effect_tc'] = pg_lt - pg_gt if not (pd.isna(pg_lt) or pd.isna(pg_gt)) else -999.0

    rt_ln = cond_rts_mean.get('Loss_NTC', np.nan); rt_gn = cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['rt_framing_bias_ntc'] = rt_ln - rt_gn if not (pd.isna(rt_ln) or pd.isna(rt_gn)) else -999.0

    rt_lt = cond_rts_mean.get('Loss_TC', np.nan); rt_gt = cond_rts_mean.get('Gain_TC', np.nan)
    summaries['rt_framing_bias_tc'] = rt_lt - rt_gt if not (pd.isna(rt_lt) or pd.isna(rt_gt)) else -999.0
    
    # --- RT std summary stats ---
    summaries['rt_std_overall'] = rts_overall.std() if not rts_overall.empty else -999.0
    for cond_key_enum, cond_filters in CONDITIONS_ROBERTS.items():
        subset = df_trials[
            (df_trials['frame'] == cond_filters['frame']) & 
            (df_trials['cond'] == cond_filters['cond'])
        ]
        rts_cond = subset['rt'].dropna()
        summaries[f'rt_std_{cond_key_enum}'] = rts_cond.std() if not rts_cond.empty else -999.0
    final_summaries = {key: summaries.get(key, -999.0) for key in stat_keys}
    for k, v in final_summaries.items():
        if pd.isna(v):
            final_summaries[k] = -999.0
    return final_summaries

def simulate_dataset_for_ppc(
    params_dict: Dict[str, float], 
    subject_trial_structure: pd.DataFrame, 
    stat_keys: List[str]
) -> Dict[str, float]:
    """Simulates one full dataset for a subject given one parameter set and their trial structure.
    GAIN THRESH MOD: For Gain-frame trials, threshold_a is reduced to a_0 * 0.75.
    """
    agent = MVNESAgent(config={})
    sim_results_list = []

    for _, trial_info in subject_trial_structure.iterrows():
        salience_input = trial_info['prob']
        norm_input = 1.0 if trial_info['is_gain_frame'] else -1.0
        # GAIN THRESH MOD: Reduce threshold for gain-frame trials
        if trial_info['is_gain_frame']:
            effective_a = params_dict['a_0'] * 0.75
        else:
            effective_a = params_dict['a_0']
        agent_run_params = {
            'w_n': params_dict['v_norm'], 'threshold_a': effective_a,
            'w_s': params_dict['w_s_eff'], 't': params_dict['t_0'],
            **BASE_SIM_PARAMS
        }
        try:
            trial_output = agent.run_mvnes_trial(salience_input, norm_input, agent_run_params)
            sim_rt = trial_output.get('rt', np.nan)
            if not pd.isna(sim_rt) and trial_info['time_constrained']:
                sim_rt = min(sim_rt, 1.0)
            sim_results_list.append({
                'rt': sim_rt, 'choice': trial_output.get('choice', np.nan),
                'frame': trial_info['frame'], 'cond': trial_info['cond'], 'prob': trial_info['prob']
            })
        except Exception:
            sim_results_list.append({'rt': np.nan, 'choice': np.nan, 'frame': trial_info['frame'], 'cond': trial_info['cond'], 'prob': trial_info['prob']})
            
    df_sim_dataset = pd.DataFrame(sim_results_list)
    return calculate_summary_stats_roberts(df_sim_dataset, stat_keys)

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

def run_ppc_for_subject(
    subject_id: Any,
    df_subject_empirical_data: pd.DataFrame,
    fitted_params_mean: Dict[str, float], # Could also pass posterior_samples directly
    npe_posterior_obj: Any, # The loaded posterior object from sbi
    observed_subject_summary_stats_tensor: torch.Tensor, # For regenerating posterior samples
    num_ppc_simulations: int,
    num_posterior_samples_for_ppc: int,
    device: str,
    output_plots_dir: Path
):
    logging.info(f"--- Starting PPC for Subject {subject_id} ---")
    stat_keys = get_roberts_summary_stat_keys()
    
    # 1. Draw posterior samples for this subject
    with torch.no_grad():
        npe_posterior_obj.set_default_x(observed_subject_summary_stats_tensor)
        posterior_samples = npe_posterior_obj.sample((num_posterior_samples_for_ppc,), show_progress_bars=False).cpu().numpy()
    
    # 2. Simulate datasets for each posterior sample
    sim_stats_list = []
    for i in tqdm(range(num_ppc_simulations), desc=f"Sim PPC {subject_id}"):
        # Optionally, sample a posterior sample each time, or cycle
        sample_idx = i % num_posterior_samples_for_ppc
        params_dict = {k: float(posterior_samples[sample_idx, j]) for j, k in enumerate(PARAM_NAMES)}
        sim_stats = simulate_dataset_for_ppc(params_dict, df_subject_empirical_data, stat_keys)
        sim_stats_list.append(sim_stats)
    df_sim_stats = pd.DataFrame(sim_stats_list)
    actual_obs_stats = {k: float(observed_subject_summary_stats_tensor[j].cpu().numpy()) for j, k in enumerate(stat_keys)}
    # 3. Plotting (optional, can be expanded)
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    for key in stat_keys:
        plt.figure(figsize=(5,3))
        plt.hist(df_sim_stats[key], bins=20, alpha=0.7, label='Simulated')
        plt.axvline(actual_obs_stats[key], color='red', linestyle='--', label='Observed')
        plt.title(f'Subj {subject_id}: {key}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plots_dir / f"ppc_subj{subject_id}_{key}.png")
        plt.close()
    # 4. Summary stats for PPC
    ppc_summary = {'subject_id': subject_id}
    for key in stat_keys:
        sim_vals = df_sim_stats[key]
        ppc_summary[f'sim_mean_{key}'] = sim_vals.mean() if not sim_vals.empty else np.nan
        ppc_summary[f'sim_median_{key}'] = sim_vals.median() if not sim_vals.empty else np.nan
        # P-value: proportion of simulated stats more extreme than observed
        if not sim_vals.empty and actual_obs_stats.get(key) is not None and not pd.isna(actual_obs_stats.get(key)):
            obs_val = actual_obs_stats.get(key)
            # Two-sided p-value (simplified)
            p_val = np.mean(np.abs(sim_vals - sim_vals.mean()) >= np.abs(obs_val - sim_vals.mean()))
            ppc_summary[f'sim_pval_{key}'] = p_val
    # Return full simulated stats and actual obs stats for coverage analysis
    return ppc_summary, df_sim_stats, actual_obs_stats

def main():
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
    args = parser.parse_args()

    output_path_data = os.path.join(args.output_dir, 'data')
    os.makedirs(output_path_data, exist_ok=True)

    ppc_summary_path = os.path.join(output_path_data, f"ppc_summary_table_gain_thresh_mod_seed{args.seed}.csv")
    ppc_param_grid_path = os.path.join(output_path_data, f"ppc_param_grid_gain_thresh_mod_seed{args.seed}.csv")

    # --- Load fitted params and empirical data before subject selection ---
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

    # Determine subject IDs to run PPC on
    if args.subject_ids is not None:
        subject_ids_for_ppc = [int(sid.strip()) for sid in args.subject_ids.split(',') if sid.strip()]
        logging.info(f"[gain_thresh_mod] Running PPC for user-specified subjects: {subject_ids_for_ppc}")
    else:
        # Use all subjects in the fitted params file
        subject_ids_for_ppc = sorted(df_fitted_params['subject_id'].unique().tolist())
        logging.info(f"[gain_thresh_mod] Running PPC for ALL fitted subjects: {subject_ids_for_ppc}")
    if not subject_ids_for_ppc:
        logging.error("No subjects selected or available for PPC. Exiting.")
        sys.exit(1)

    # --- Begin full PPC workflow from original script ---
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}. Seed: {args.seed}")

    output_path = Path(args.output_dir)
    output_path_data = output_path / 'data'; output_path_data.mkdir(parents=True, exist_ok=True)
    output_plots_dir = output_path / 'plots'; output_plots_dir.mkdir(parents=True, exist_ok=True)

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

    ppc_summary_records = []
    for subj_id in subject_ids_for_ppc:
        df_subj = df_all_empirical_data[df_all_empirical_data['subject'] == subj_id]
        if df_subj.empty:
            logging.warning(f"No data found for subject {subj_id}. Skipping.")
            continue
        # Use mean fitted params for subject (or you could use full posterior samples)
        subj_row = df_fitted_params[df_fitted_params['subject_id'] == subj_id]
        if subj_row.empty:
            logging.warning(f"No fitted params found for subject {subj_id}. Skipping.")
            continue
        subj_row = subj_row.iloc[0]
        # Use the correct column names from the fitted params file (mean values)
        fitted_params_mean = {k: subj_row[f'mean_{k}'] for k in PARAM_NAMES}
        obs_stats = calculate_summary_stats_roberts(df_subj, stat_keys)
        obs_stats_tensor = torch.tensor([obs_stats[k] for k in stat_keys], dtype=torch.float32, device=device)
        # Run PPC for this subject
        summary, df_sim_stats, actual_obs_stats = run_ppc_for_subject(
            subject_id=subj_id,
            df_subject_empirical_data=df_subj,
            fitted_params_mean=fitted_params_mean,
            npe_posterior_obj=loaded_npe_posterior_object,
            observed_subject_summary_stats_tensor=obs_stats_tensor,
            num_ppc_simulations=args.num_ppc_simulations,
            num_posterior_samples_for_ppc=args.num_posterior_samples_for_ppc,
            device=device,
            output_plots_dir=output_plots_dir
        )
        # Attach sim stats and obs stats for coverage summary
        summary['df_sim_stats'] = df_sim_stats
        summary['obs_stats'] = actual_obs_stats
        ppc_summary_records.append(summary)
    # Save summary table
    df_ppc_summary = pd.DataFrame(ppc_summary_records)
    # Save PPC summary table
    df_ppc_summary = pd.DataFrame(ppc_summary_records)
    summary_path = output_path_data / f"ppc_summary_gain_thresh_mod_seed{args.seed}.csv"
    df_ppc_summary.to_csv(summary_path, index=False)
    print(f"PPC summary saved to: {summary_path}")

    # --- Compute PPC Coverage Summary ---
    # Modify run_ppc_for_subject to return df_sim_stats and obs_stats
    # Collect all simulated stats and observed stats for each subject/stat
    all_sim_stats = {k: [] for k in stat_keys}
    all_obs_stats = {k: [] for k in stat_keys}
    for subj_idx, summary in enumerate(ppc_summary_records):
        subj_sim_stats = summary.get('df_sim_stats', None)
        subj_obs_stats = summary.get('obs_stats', None)
        if subj_sim_stats is None or subj_obs_stats is None:
            continue
        for k in stat_keys:
            # Defensive: Only append if key exists and values are not all NaN
            if k in subj_sim_stats and k in subj_obs_stats:
                sim_vals = subj_sim_stats[k].values
                obs_val = subj_obs_stats[k]
                if np.all(np.isnan(sim_vals)) or pd.isna(obs_val):
                    continue
                all_sim_stats[k].append(sim_vals)
                all_obs_stats[k].append(obs_val)
    coverage_rows = []
    for k in stat_keys:
        covered_90 = 0
        covered_95 = 0
        total = len(all_obs_stats[k])
        if total == 0:
            print(f"[COVERAGE] Stat '{k}': No valid subjects, writing NaN.")
        else:
            print(f"[COVERAGE] Stat '{k}': Calculating coverage over {total} subjects.")
        for obs, sim_vals in zip(all_obs_stats[k], all_sim_stats[k]):
            if len(sim_vals) == 0 or pd.isna(obs):
                continue
            q05, q95 = np.percentile(sim_vals, [5, 95])
            q025, q975 = np.percentile(sim_vals, [2.5, 97.5])
            if q05 <= obs <= q95:
                covered_90 += 1
            if q025 <= obs <= q975:
                covered_95 += 1
        pct_covered_90 = 100.0 * covered_90 / total if total > 0 else np.nan
        pct_covered_95 = 100.0 * covered_95 / total if total > 0 else np.nan
        coverage_rows.append({'stat_name': k, 'pct_covered_90': pct_covered_90, 'pct_covered_95': pct_covered_95})
    df_coverage = pd.DataFrame(coverage_rows)
    coverage_path = output_path_data / f"ppc_coverage_summary_gain_thresh_mod_seed{args.seed}.csv"
    df_coverage.to_csv(coverage_path, index=False)
    print(f"PPC coverage summary saved to: {coverage_path}")
    logging.info(f"PPC summary saved to: {summary_path}")
    # --- End full PPC workflow ---
    subject_trial_structure = pd.DataFrame({
        'prob': [0.5, 0.7, 0.3],
        'is_gain_frame': [True, False, True],
        'time_constrained': [True, False, True],
        'frame': [1, 2, 3],
        'cond': ['A', 'B', 'A'],
        'prob': [0.5, 0.7, 0.3]
    })

    params_dict = {
        'a_0': 1.0,
        'v_norm': 1.0,
        'w_s_eff': 1.0,
        't_0': 1.0
    }

    stat_keys = ['mean_rt', 'accuracy']

    result = simulate_dataset_for_ppc(params_dict, subject_trial_structure, stat_keys)
    print(result)

    # Save PPC summary to CSV
    ppc_summary_df = pd.DataFrame([result])
    ppc_summary_df.to_csv(ppc_summary_path, index=False)

if __name__ == '__main__':
    main()
