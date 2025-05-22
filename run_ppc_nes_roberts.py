# Filename: empirical_fitting/run_ppc_nes_roberts.py
# Purpose: Perform Posterior Predictive Checks (PPCs) for the NES model 
#          fitted to the Roberts et al. (2022) empirical data.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Must be set before numpy/import logging
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging

# Maximize CPU utilization for PyTorch
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    logging.info(f"[Startup] Detected total system RAM: {ram_gb:.1f} GB")
except ImportError:
    logging.info("[Startup] psutil not installed; cannot display RAM info.")

cpu_threads = os.cpu_count() or 1
logging.info(f"[Startup] Setting torch.set_num_threads({cpu_threads}) for maximal CPU utilization.")
torch.set_num_threads(cpu_threads)

from typing import Dict, List, Tuple, Any, Optional, Union

# Define helper functions for PPC (no multiprocessing, no timeouts)
def sample_from_posterior(posterior_obj, num_samples, obs_tensor):
    """Helper function for sampling from posterior (used in main process)"""
    logging.info("[Main process] Entered sample_from_posterior.")
    return posterior_obj.sample(
        (num_samples,),
        x=obs_tensor.unsqueeze(0),
        show_progress_bars=False
    ).cpu()

def run_simulation(params_dict, subject_data, stat_keys):
    """Helper function for running simulations (used in main process)"""
    import torch, numpy as np
    # Derive deterministic seed from global_seed, subject_id, sim_idx if available
    subject_id = params_dict.get('subject_id', 0)
    sim_idx = params_dict.get('sim_idx', 0)
    seed = int(0) + int(subject_id) * 100_000 + int(sim_idx)
    torch.manual_seed(seed)
    np.random.seed(seed)
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
    """
    Returns the keys for the summary statistics vector.
    
    Returns:
        List[str]: List of summary statistic keys in the exact order expected by the NPE.
        
    Note:
        The statistics are defined in stats_schema.py to ensure consistency across the codebase.
        Any changes to the statistics must be made there, not here.
    """
    from stats_schema import ROBERTS_SUMMARY_STAT_KEYS, validate_summary_stats
    
    try:
        # This will raise ValueError if the stats don't match expectations
        validate_summary_stats()
        return list(ROBERTS_SUMMARY_STAT_KEYS)  # Return a copy to prevent modification
    except ImportError as e:
        logging.error("Failed to import stats_schema. Make sure it's in your PYTHONPATH.")
        raise
    except ValueError as e:
        logging.error(f"Summary statistics validation failed: {e}")
        raise

def safe_diff(a, b):
    return np.nan if (np.isnan(a) or np.isnan(b)) else a - b

def calculate_summary_stats_for_ppc(df_trials: pd.DataFrame, stat_keys: List[str]) -> Dict[str, float]:
    """Calculates summary statistics for a single dataset (empirical or simulated)."""
    import numpy as np
    summaries = {k: np.nan for k in stat_keys}  # Initialize all stats as NaN
    
    # Handle completely empty datasets
    if df_trials.empty:
        return summaries
    
    # Filter to valid trials if needed
    df_valid = df_trials.dropna(subset=['choice', 'rt'])
    if len(df_valid) == 0:
        return summaries
    
    # Overall stats across conditions
    choices = df_valid['choice']
    rts = df_valid['rt']
    
    if 'choice' in df_valid.columns and len(choices) > 0:
        prop_gamble = (choices == 1).mean()
        if not np.isnan(prop_gamble):
            summaries['prop_gamble_overall'] = float(prop_gamble)
    
    if len(rts) > 0:
        rt_mean = rts.mean()
        if not np.isnan(rt_mean):
            summaries['mean_rt_overall'] = float(rt_mean)
        
        try:
            q = rts.quantile([0.1, 0.5, 0.9])
            if not q.empty:
                for q_name, q_val in zip(['rt_q10', 'rt_q50', 'rt_q90'], [0.1, 0.5, 0.9]):
                    val = q.get(q_val, np.nan)
                    if not np.isnan(val):
                        summaries[f'{q_name}_overall'] = float(val)
        except Exception as e:
            logging.debug(f"Error calculating overall RT quantiles: {e}")
    
    # Per-condition stats
    cond_props, cond_rts_mean, cond_rts_std, cond_rts_median = {}, {}, {}, {}
    for cond_key, cond_info in CONDITIONS_ROBERTS.items():
        cond_key_enum = cond_key  # e.g. 'Gain_TC'
        cond_mask = (df_valid['frame'] == cond_info['frame']) & (df_valid['cond'] == cond_info['cond'])
        choices_cond = df_valid.loc[cond_mask, 'choice'] if 'choice' in df_valid.columns else pd.Series([])
        rts_cond = df_valid.loc[cond_mask, 'rt']
        
        # Calculate choice proportions if we have valid choice data
        if not choices_cond.empty and len(choices_cond) > 0:
            prop_gamble_cond = (choices_cond == 1).mean()
            if not np.isnan(prop_gamble_cond):
                summaries[f'prop_gamble_{cond_key_enum}'] = float(prop_gamble_cond)
                cond_props[cond_key_enum] = prop_gamble_cond
        
        # Calculate RT statistics if we have valid RT data
        if not rts_cond.empty and len(rts_cond) > 0:
            # Mean and std RTs
            mean_rt = rts_cond.mean()
            std_rt = rts_cond.std()
            
            if not np.isnan(mean_rt):
                summaries[f'mean_rt_{cond_key_enum}'] = float(mean_rt)
                cond_rts_mean[cond_key_enum] = mean_rt
            
            if not np.isnan(std_rt):
                summaries[f'rt_std_{cond_key_enum}'] = float(std_rt)
                cond_rts_std[cond_key_enum] = std_rt
            
            # Calculate and store median RTs
            median_rt = rts_cond.median()
            if not np.isnan(median_rt):
                cond_rts_median[cond_key_enum] = median_rt
            
            # Calculate RT quantiles
            try:
                q_c = rts_cond.quantile([0.1, 0.5, 0.9])
                if not q_c.empty:
                    for q_name, q_val in zip(['rt_q10', 'rt_q50', 'rt_q90'], [0.1, 0.5, 0.9]):
                        val = q_c.get(q_val, np.nan)
                        if not np.isnan(val):
                            summaries[f'{q_name}_{cond_key_enum}'] = float(val)
            except Exception as e:
                logging.debug(f"Error calculating RT quantiles for {cond_key_enum}: {e}")
            
            # Calculate RT histogram bins (proportion of trials in each bin)
            max_rt = 1.0 if 'TC' in cond_key_enum else 3.0
            edges = np.linspace(0, max_rt, 6)
            try:
                hist, _ = np.histogram(rts_cond.clip(0, max_rt), bins=edges, density=False)
                if len(rts_cond) > 0:
                    hist = hist / len(rts_cond)
                for i, h in enumerate(hist):
                    if not np.isnan(h):
                        summaries[f'rt_hist_bin{i}_{cond_key_enum}'] = float(h)
            except Exception as e:
                logging.debug(f"Error calculating RT histogram for {cond_key_enum}: {e}")
        else:
            # Log dropped/empty RT subset for this condition
            logging.warning(f"Empty RT subset for dataset (unknown subj), condition {cond_key_enum}")
    
    # Framing effects (choice proportions)
    pg_ln = cond_props.get('Loss_NTC', np.nan)
    pg_gn = cond_props.get('Gain_NTC', np.nan)
    summaries['framing_effect_ntc'] = safe_diff(pg_ln, pg_gn)
    
    pg_lt = cond_props.get('Loss_TC', np.nan)
    pg_gt = cond_props.get('Gain_TC', np.nan)
    summaries['framing_effect_tc'] = safe_diff(pg_lt, pg_gt)
    
    # RT framing effects (mean RT differences between loss and gain)
    rt_ln = cond_rts_mean.get('Loss_NTC', np.nan)
    rt_gn = cond_rts_mean.get('Gain_NTC', np.nan)
    summaries['rt_framing_bias_ntc'] = safe_diff(rt_ln, rt_gn)
    
    rt_lt = cond_rts_mean.get('Loss_TC', np.nan)
    rt_gt = cond_rts_mean.get('Gain_TC', np.nan)
    summaries['rt_framing_bias_tc'] = safe_diff(rt_lt, rt_gt)
    
    # Mean RT contrasts for Gain vs Loss (added for 5-parameter model)
    mrt_gtc = cond_rts_mean.get('Gain_TC', np.nan)
    mrt_ltc = cond_rts_mean.get('Loss_TC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_TC'] = safe_diff(mrt_gtc, mrt_ltc)

    mrt_gntc = cond_rts_mean.get('Gain_NTC', np.nan)
    mrt_lntc = cond_rts_mean.get('Loss_NTC', np.nan)
    summaries['mean_rt_Gain_vs_Loss_NTC'] = safe_diff(mrt_gntc, mrt_lntc)

    # Median RT contrasts (based on stored median values)
    medrt_gtc = cond_rts_median.get('Gain_TC', np.nan)
    medrt_ltc = cond_rts_median.get('Loss_TC', np.nan)
    summaries['rt_median_Gain_vs_Loss_TC'] = safe_diff(medrt_gtc, medrt_ltc)

    medrt_gntc = cond_rts_median.get('Gain_NTC', np.nan)
    medrt_lntc = cond_rts_median.get('Loss_NTC', np.nan)
    summaries['rt_median_Gain_vs_Loss_NTC'] = safe_diff(medrt_gntc, medrt_lntc)

    # RT effects within frames (TC vs NTC mean RTs)
    summaries['framing_effect_rt_gain'] = safe_diff(mrt_gtc, mrt_gntc)
    summaries['framing_effect_rt_loss'] = safe_diff(mrt_ltc, mrt_lntc)
    
    # Ensure we only return the requested stats in the correct order
    return {k: summaries[k] for k in stat_keys}


def simulate_dataset_for_ppc(
    params_dict: Dict[str, float], 
    subject_trial_structure: Any,  # numpy record array for speed
    stat_keys: List[str],
    agent: Any = None
) -> Dict[str, float]:
    # --- Safety check: enforce parameter name consistency ---
    # This prevents silent bugs if upstream parameter names change or are refactored.
    expected_param_names = {'w_n', 'a_0', 'w_s_eff', 't_0', 'alpha_gain'}
    assert set(params_dict.keys()) >= expected_param_names, (
        f"params_dict missing expected keys: {expected_param_names - set(params_dict.keys())}\n"
        f"Current keys: {set(params_dict.keys())}"
    )
    # If you refactor parameter names upstream, update this assertion and mapping accordingly.

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
        # Set max_time based on trial type (before calling agent)
        if trial_info['time_constrained']:
            agent_run_params['max_time'] = 1.0
        else:
            agent_run_params['max_time'] = 3.0
        try:
            # Always run the DDM simulation, let run_mvnes_trial handle all drift rates and RT bounds
            trial_output = agent.run_mvnes_trial(salience_input, norm_input, agent_run_params)
            sim_rt = trial_output.get('rt', np.nan)
            sim_choice = trial_output.get('choice', np.nan)
            trace = trial_output.get('trace', [np.nan])
            timeout = trial_output.get('timeout', False)
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
    checkpoint = torch.load(npe_file_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'density_estimator_state_dict' not in checkpoint:
        raise ValueError("NPE checkpoint error: Expected dict with 'density_estimator_state_dict'.")

    num_summary_stats_trained = checkpoint.get('num_summary_stats')
    logging.info(f"Checkpoint num_summary_stats: {num_summary_stats_trained}")
    if num_summary_stats_trained is None or num_summary_stats_trained <= 0:
        logging.warning(f"num_summary_stats_trained is {num_summary_stats_trained}. Trying to use len(get_roberts_summary_stat_keys()).")
        num_summary_stats_trained = len(get_roberts_summary_stat_keys())
    if num_summary_stats_trained <= 0:
        raise ValueError("Number of summary statistics for dummy_x is non-positive. Cannot build dummy network.")

    dummy_theta = prior.sample((10,))
    # Use non-degenerate dummy_x: each row is random and offset
    dummy_x = torch.randn(10, num_summary_stats_trained, device=device) + torch.arange(10, device=device).unsqueeze(1)
    logging.info(f"Dummy_x shape: {dummy_x.shape}, should be (10, D>0) and non-degenerate")
    
    current_script_num_stats = len(get_roberts_summary_stat_keys())
    if current_script_num_stats != num_summary_stats_trained:
        logging.error(f"CRITICAL MISMATCH: Loaded NPE expects {num_summary_stats_trained} summary stats, "
                      f"but current script defines {current_script_num_stats}. Adjust get_roberts_summary_stat_keys().")
        raise ValueError("Summary statistic dimension mismatch.")

    try:
        inference_algorithm = SNPE(prior=prior, density_estimator='maf', device=device, z_score_x='none', z_score_y=False)
        logging.info("Instantiated SNPE with z_score_x='none' and z_score_y=False.")
    except TypeError:
        inference_algorithm = SNPE(prior=prior, density_estimator='maf', device=device)
        logging.info("SNPE does not accept z_score_x/z_score_y in this version; falling back to default.")

    # To load state_dict, the network must be built first.
    # We use dummy data to trigger the network build via append_simulations and a 0-epoch train.
    # Use 10 non-degenerate dummy samples
    dummy_theta = prior.sample((10,))
    dummy_x = torch.randn(10, num_summary_stats_trained, device=device) + torch.arange(10, device=device).unsqueeze(1)
    density_estimator_build_kwargs = checkpoint.get('density_estimator_build_kwargs', {})
    filtered_build_kwargs = {k: v for k, v in density_estimator_build_kwargs.items() if k not in ('z_score_x', 'z_score_y')}
    if len(filtered_build_kwargs) != len(density_estimator_build_kwargs):
        logging.info(f"Filtered out unsupported keys from build kwargs: {[k for k in density_estimator_build_kwargs if k in ('z_score_x', 'z_score_y')]}")
    logging.info(f"Building network structure with dummy data. Build kwargs from checkpoint: {filtered_build_kwargs}")
    density_estimator_net = inference_algorithm.append_simulations(
        dummy_theta,
        dummy_x
    ).train(
        max_num_epochs=1,
        show_train_summary=False,
        **filtered_build_kwargs
    )
    logging.info(f"density_estimator_net after train: {density_estimator_net}")
    logging.info(f"inference_algorithm.neural_net after train: {getattr(inference_algorithm, 'neural_net', None)}")

    # Use whichever is not None
    net_to_use = density_estimator_net if density_estimator_net is not None else getattr(inference_algorithm, 'neural_net', None)
    if net_to_use is None:
        logging.error(f"Both density_estimator_net and inference_algorithm.neural_net are None after train. Types: {type(density_estimator_net)}, {type(getattr(inference_algorithm, 'neural_net', None))}")
        raise RuntimeError("Failed to build the neural network structure within the SNPE object.")
    logging.info("Loading state_dict into the built network...")
    net_to_use.load_state_dict(checkpoint['density_estimator_state_dict'], strict=False)

    loaded_net = net_to_use
    if hasattr(loaded_net, 'num_flows') and 'num_flows' in checkpoint:
        if loaded_net.num_flows != checkpoint.get('num_flows'):
            logging.warning(f"num_flows mismatch: net={loaded_net.num_flows}, checkpoint={checkpoint.get('num_flows')}. strict=False used.")
    if hasattr(loaded_net, 'hidden_features') and 'hidden_features' in checkpoint:
        if loaded_net.hidden_features != checkpoint.get('hidden_features'):
            logging.warning(f"hidden_features mismatch: net={loaded_net.hidden_features}, checkpoint={checkpoint.get('hidden_features')}. strict=False used.")

    # Try both build_posterior() signatures
    try:
        posterior_object = inference_algorithm.build_posterior(net_to_use)
    except Exception:
        posterior_object = inference_algorithm.build_posterior()
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
    import time as _time
    _t0 = _time.time()
    logging.info(f"--- Starting PPC for Subject {subject_id} ---")
    print(f"[PPC] Starting subject {subject_id}", flush=True)
    stat_keys = get_roberts_summary_stat_keys()
    # 1. Get actual observed summary statistics for this subject
    actual_obs_stats = calculate_summary_stats_for_ppc(df_subject_empirical_data, stat_keys)
    # 2. Prepare trial structure as numpy record array for simulation speed
    trial_struct = df_subject_empirical_data[['prob','is_gain_frame','frame','cond','time_constrained']].to_records(index=False)
    # 3. Draw a larger set of posterior samples for better global coverage
    num_posterior_samples = 2000
    logging.info(f"Drawing {num_posterior_samples} posterior parameter samples for subject {subject_id}...")
    with torch.no_grad():
        x_condition = observed_subject_summary_stats_tensor.unsqueeze(0) if observed_subject_summary_stats_tensor.dim() == 1 else observed_subject_summary_stats_tensor
        x_condition = x_condition.to(device)
        subject_posterior_param_samples = npe_posterior_obj.sample(
            (num_posterior_samples,),
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
            print(f"[PPC] Subject {subject_id}: sim {i+1}/{num_ppc_simulations}", flush=True)
        selected_params_tensor = subject_posterior_param_samples[i]
        params_dict_for_sim = {name: val.item() for name, val in zip(PARAM_NAMES, selected_params_tensor)}
        params_dict_for_sim['subject_id'] = subject_id
        params_dict_for_sim['sim_idx'] = i
        try:
            sim_stats = simulate_dataset_for_ppc(params_dict_for_sim, trial_struct, stat_keys, agent=agent)
        except Exception as e:
            import traceback
            logging.error(f"Exception in simulate_dataset_for_ppc for subject {subject_id}, sim {i}: {e}")
            traceback.print_exc()
            print(f"[PPC][ERROR] Subject {subject_id} sim {i}: {e}", flush=True)
            sim_stats = {k: float('nan') for k in stat_keys}
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
        sim_values = df_ppc_stats[key].replace(-999.0, np.nan).dropna()
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
    _t1 = _time.time()
    elapsed = _t1 - _t0
    print(f"[PPC] Finished subject {subject_id} in {elapsed:.1f} sec", flush=True)
    logging.info(f"--- Finished PPC for Subject {subject_id} in {elapsed:.1f} sec ---")
    # Return a summary for a table
    ppc_summary = {'subject_id': subject_id}
    for key in plot_stat_keys:
        ppc_summary[f'obs_{key}'] = actual_obs_stats.get(key)
        sim_vals = df_ppc_stats[key].replace(-999.0, np.nan).dropna()
        ppc_summary[f'sim_mean_{key}'] = sim_vals.mean() if not sim_vals.empty else np.nan
        ppc_summary[f'sim_median_{key}'] = sim_vals.median() if not sim_vals.empty else np.nan
        if not sim_vals.empty and actual_obs_stats.get(key) is not None and not pd.isna(actual_obs_stats.get(key)):
            obs_val = actual_obs_stats.get(key)
            p_val = np.mean(np.abs(sim_vals - sim_vals.mean()) >= np.abs(obs_val - sim_vals.mean()))
            ppc_summary[f'sim_pval_{key}'] = p_val
    return ppc_summary


# --- Main Script Execution ---
if __name__ == "__main__":
    # Quick pytest-style smoke tests
    from stats_schema import validate_summary_stats
    import pandas as pd
    assert validate_summary_stats()           # schema length check
    subj_dummy = pd.DataFrame({               # 4 fake trials
        'frame':['gain','gain','loss','loss'],
        'cond':['tc','ntc','tc','ntc'],
        'prob':[.75,.25,.75,.25],
        'choice':[1,0,1,0],
        'rt':[0.5,0.8,0.6,1.2]
    })
    stats = calculate_summary_stats_for_ppc(subj_dummy, get_roberts_summary_stat_keys())
    assert not any(v is None for v in stats.values())
    print("[Smoke test] Summary stats schema and calculation: PASS")

    parser = argparse.ArgumentParser(description="Run PPCs for NES model on Roberts et al. data.")
    parser.add_argument('--npe_file', type=str, required=True, help='Path to PRE-TRAINED NPE checkpoint (.pt file).')
    parser.add_argument('--fitted_params_file', type=str, required=True, help='Path to CSV file with fitted empirical parameters (e.g., from empirical fitting run).')
    parser.add_argument('--roberts_data_file', type=str, default="./roberts_framing_data/ftp_osf_data.csv", help='Path to the Roberts data CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save PPC results and plots.')
    parser.add_argument('--subject_ids', type=str, default=None, help='Comma-separated list of subject IDs for PPC (e.g., "114,122,165"). Default: 3-5 representative subjects.')
    # ... (rest of the code remains the same)
    parser.add_argument('--num_ppc_simulations', type=int, default=50, help='Number of datasets to simulate from posterior for PPC.')
    parser.add_argument('--num_posterior_samples_for_ppc', type=int, default=200, help='Number of parameter sets to draw from each subject posterior for PPC sims.')
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
            logging.info(f"\n========== Starting PPC Coverage for Subject {subj_id} (row {idx}) ==========")
            logging.debug(f"Subject {subj_id} (row {idx}): DataFrame shape: {df_all_empirical_data.shape}, Subject trials: {df_all_empirical_data[df_all_empirical_data['subject'] == subj_id_int].shape}")
            print(f"[Coverage] Starting subject {subj_id} (row {idx})", flush=True)
            import time as _time
            _t0 = _time.time()
            df_subj = df_all_empirical_data[df_all_empirical_data['subject'] == subj_id_int]
            if df_subj.empty:
                logging.warning(f"Subject {subj_id} (row {idx}): No data found in empirical data, skipping.")
                continue
            obs_stats = calculate_summary_stats_for_ppc(df_subj, stat_keys)
            logging.debug(f"Subject {subj_id} (row {idx}): Observed summary stats: {obs_stats}")
            obs_stats_tensor = torch.tensor([obs_stats[k] for k in stat_keys], dtype=torch.float32, device=device)
            subj_trial_struct = df_subj[['prob','is_gain_frame','frame','cond','time_constrained']].to_records(index=False)
            logging.debug(f"Subject {subj_id} (row {idx}): Trial structure sample: {subj_trial_struct[:2] if len(subj_trial_struct) > 1 else subj_trial_struct}")
            # --- Single-core, main-process PPC (no multiprocessing, no timeouts) ---
            try:
                logging.info(f"Subject {subj_id} (row {idx}): Sampling {args.num_posterior_samples_for_ppc} posterior parameter sets in main process...")
                t_sample_start = _time.time()
                posterior_samples = loaded_npe_posterior_object.sample(
                    (args.num_posterior_samples_for_ppc,), x=obs_stats_tensor.unsqueeze(0), show_progress_bars=False
                ).cpu()
                logging.info(f"Subject {subj_id} (row {idx}): Posterior sampling finished in {(_time.time() - t_sample_start):.2f}s. Shape: {getattr(posterior_samples, 'shape', type(posterior_samples))}")
                from tqdm import tqdm as tqdm_inner
                sim_stats = []
                for i in tqdm_inner(range(args.num_ppc_simulations), desc=f"Subject {subj_id} PPC Sims", leave=False):
                    t_sim_start = _time.time()
                    param_sample_idx = torch.randint(len(posterior_samples), (1,)).item()
                    params_dict_for_sim = {name: val.item() for name, val in zip(PARAM_NAMES, posterior_samples[param_sample_idx])}
                    if (i + 1) % (args.num_ppc_simulations // 10 or 1) == 0:
                        logging.info(f"Subject {subj_id} (row {idx}): PPC simulation {i+1}/{args.num_ppc_simulations}")
                    try:
                        sim_stats.append(simulate_dataset_for_ppc(params_dict_for_sim, subj_trial_struct, stat_keys))
                        logging.debug(f"Subject {subj_id} (row {idx}): PPC sim {i+1} finished in {(_time.time() - t_sim_start):.2f}s")
                    except Exception as e:
                        logging.error(f"Error in PPC simulation for subject {subj_id} (row {idx}), params: {params_dict_for_sim}: {e}", exc_info=True)
                        sim_stats.append({k: -999.0 for k in stat_keys})
                df_sim_stats = pd.DataFrame(sim_stats)
            except Exception as e:
                logging.error(f"Error in PPC for subject {subj_id} (row {idx}): {e}", exc_info=True)
                continue

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
            _t1 = _time.time()
            elapsed = _t1 - _t0
            logging.info(f"========== Finished PPC Coverage for Subject {subj_id} (row {idx}) in {elapsed:.1f} sec ==========")
            logging.debug(f"Subject {subj_id} (row {idx}): PPC stats: obs_stats={obs_stats}, sim_stats_head={df_sim_stats.head(2).to_dict() if not df_sim_stats.empty else 'empty'}")
            print(f"[Coverage] Finished subject {subj_id} (row {idx}) in {elapsed:.1f} sec", flush=True)

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