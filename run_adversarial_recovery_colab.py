# Filename: run_adversarial_recovery_colab.py
# Purpose: Colab-friendly version of adversarial model recovery script with GPU support.
#          1. Install required packages if missing.
#          2. Use GPU if available (PyTorch, sbi, etc.).
#          3. Adapt file paths for Colab (e.g., Google Drive or file uploads).
#          4. Run the same workflow as run_adversarial_recovery.py.

# --- 0. Colab Setup ---
# Uncomment the following lines if running in Colab to install required packages:
# !pip install sbi arviz matplotlib pandas scipy

import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os

# --- Colab & CUDA Diagnostics ---
try:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA device name: None")
except Exception as e:
    print(f"Error checking CUDA devices: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- (If running in Colab, ensure PyTorch with CUDA is installed) ---
# Uncomment and run this cell at the top of your Colab notebook if you have issues:
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# import importlib; importlib.reload(torch)

from pathlib import Path
import argparse
import logging
import traceback
from functools import partial
from scipy import stats as sp_stats

# --- 1. File I/O for Colab ---
# If using Google Drive, uncomment and run:
# from google.colab import drive
# drive.mount('/content/drive')
# BASE_PATH = '/content/drive/MyDrive/YOUR_PROJECT_FOLDER/'
# Or use file upload widgets:
# from google.colab import files
# uploaded = files.upload()

# For now, use current working directory (can be changed by user)
BASE_PATH = os.getcwd()

# --- 2. Robust Imports & Dependency Checks ---
try:
    script_dir = Path(BASE_PATH)
    src_dir = script_dir / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from agent_mvnes import MVNESAgent
    try:
        from agent_config import T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME
    except ImportError:
        T_NONDECISION, NOISE_STD_DEV, DT, MAX_TIME = 0.1, 1.0, 0.01, 10.0
except ImportError as e:
    print(f"ERROR: Failed to import MVNESAgent: {e}")
    sys.exit(1)

try:
    import sbi
    from sbi.inference import SNPE_C as SNPE
    from sbi.utils import BoxUniform
    from sbi.analysis import pairplot
    import arviz as az
    print(f"Successfully imported sbi version: {sbi.__version__}")
    print(f"Successfully imported arviz version: {az.__version__}")
except ImportError:
    print("ERROR: sbi or arviz library not found. Please install: pip install sbi arviz")
    sys.exit(1)

# --- 3. Device Selection ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 4. All required constants and functions from run_adversarial_recovery.py ---
import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from functools import partial

# --- Simulation and Model Constants ---
DEFAULT_N_SUBJECTS = 10
DEFAULT_N_TRIALS_PER_SUB = 300
N_TRIALS_PER_DATASET = DEFAULT_N_TRIALS_PER_SUB
DEFAULT_NPE_TRAINING_SIMS_ADVERSARIAL = 5000
DEFAULT_NPE_POSTERIOR_SAMPLES = 2000
DEFAULT_SEED = 42
SIMPLE_DDM_A_TRUE = 1.2
SIMPLE_DDM_W_S_TRUE = 0.5
SIMPLE_DDM_T_TRUE = 0.25
FIXED_W_S_FOR_NES_FIT = 0.7
PRIOR_NES_LOW = torch.tensor([0.1, 0.4, 0.05])
PRIOR_NES_HIGH = torch.tensor([2.0, 1.5, 0.5])
PARAM_NAMES_NES = ['w_n_eff', 'a_nes', 't_nes']
PRIOR_SIMPLE_DDM_LOW = torch.tensor([0.4])
PRIOR_SIMPLE_DDM_HIGH = torch.tensor([2.0])
PARAM_NAMES_SIMPLE_DDM = ['a_simple']
FIXED_V_SIMPLE = 0.7
FIXED_T_SIMPLE = 0.25
BASE_SIM_PARAMS_ADV = {
    'noise_std_dev': 1.0,
    'dt': 0.01,
    'max_time': 10.0,
    'affect_stress_threshold_reduction': -0.3,
    'veto_flag': False
}
CONFLICT_LEVELS_ADV = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
CONFLICT_PROPORTIONS_ADV  = np.array([0.2] * 5)

# --- Helper: Generate conflict levels for trials ---
def generate_stroop_conflict_levels(n_trials, conflict_levels_arr, conflict_proportions_arr, seed=None):
    rng = np.random.default_rng(seed)
    level_indices = rng.choice(np.arange(len(conflict_levels_arr)), size=n_trials, p=conflict_proportions_arr)
    return conflict_levels_arr[level_indices]

# --- Simulate DDM Trials ---
def simulate_ddm_trials_from_params(params_dict, n_trials, conflict_levels_arr, conflict_proportions_arr, base_sim_params_dict, is_nes_model=False, fixed_w_s_nes_val=None):
    from agent_mvnes import MVNESAgent
    results_list = []
    agent = MVNESAgent(config={})
    conflict_level_sequence = generate_stroop_conflict_levels(n_trials, conflict_levels_arr, conflict_proportions_arr)
    a_sim = params_dict['a']
    if is_nes_model:
        t0_sim = params_dict['t']
    else:
        t0_sim = FIXED_T_SIMPLE
    sim_params_for_agent = {
        'threshold_a': a_sim,
        't': t0_sim,
        **{k: v for k, v in base_sim_params_dict.items() if k != 't'}
    }
    for i in range(n_trials):
        conflict_lvl = conflict_level_sequence[i]
        if is_nes_model:
            w_n_eff_sim = params_dict['w_n_eff']
            w_s_sim = fixed_w_s_nes_val
            drift_rate = w_s_sim * (1.0 - conflict_lvl) - w_n_eff_sim * conflict_lvl
            sim_params_for_agent['w_s'] = w_s_sim
            sim_params_for_agent['w_n'] = w_n_eff_sim
            salience_input_trial = 1.0 - conflict_lvl
            norm_input_trial = conflict_lvl
        else:
            v_simple_sim = FIXED_V_SIMPLE
            t0_sim = FIXED_T_SIMPLE
            sim_params_for_agent['w_s'] = v_simple_sim
            sim_params_for_agent['w_n'] = 0.0
            salience_input_trial = 1.0
            norm_input_trial = 0.0
        try:
            trial_result = agent.run_mvnes_trial(
                salience_input=salience_input_trial,
                norm_input=norm_input_trial,
                params=sim_params_for_agent
            )
            results_list.append({
                'rt': trial_result.get('rt', np.nan),
                'choice': trial_result.get('choice', np.nan),
                'conflict_level': conflict_lvl
            })
        except Exception:
            results_list.append({'rt': np.nan, 'choice': np.nan, 'conflict_level': conflict_lvl})
    df_simulated = pd.DataFrame(results_list)
    df_simulated.dropna(subset=['rt', 'choice'], inplace=True)
    return df_simulated

# --- Summary Statistics ---
def get_summary_stat_keys():
    # Only two robust stats for debugging
    return ["overall_choice_rate", "overall_mean_rt"]

def calculate_summary_stats(df_trials):
    keys = get_summary_stat_keys()
    summaries = {k: -999.0 for k in keys}
    df_results = df_trials.dropna(subset=['rt', 'choice'])
    if len(df_results) == 0:
        return summaries
    # Only compute the two robust stats
    summaries["overall_choice_rate"] = float(np.mean(df_results['choice'] == 1)) if len(df_results) > 0 else -999.0
    summaries["overall_mean_rt"] = float(np.nanmean(df_results['rt'])) if len(df_results['rt']) > 0 else -999.0
    # Replace any remaining NaNs/Infs with -999.0 for robustness
    for k in summaries:
        if not np.isfinite(summaries[k]):
            summaries[k] = -999.0
    return summaries

# --- sbi Simulator Wrapper ---
def sbi_simulator_wrapper(parameter_set_tensor, is_nes_model_flag, fixed_w_s_for_nes=None):
    raw = parameter_set_tensor.cpu().numpy().flatten()
    if is_nes_model_flag:
        params_dict = {'w_n_eff': raw[0], 'a': raw[1], 't': raw[2]}
        stats = list(calculate_summary_stats(
            simulate_ddm_trials_from_params(
                params_dict, N_TRIALS_PER_DATASET, CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
                is_nes_model=True, fixed_w_s_nes_val=fixed_w_s_for_nes
            )
        ).values())
    else:
        params_dict = {'a': raw[0]}
        stats = list(calculate_summary_stats(
            simulate_ddm_trials_from_params(
                params_dict, N_TRIALS_PER_DATASET, CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
                is_nes_model=False
            )
        ).values())
    stats = np.nan_to_num(stats, nan=-999.0, posinf=-999.0, neginf=-999.0)
    return torch.tensor(stats, dtype=torch.float32)

# --- NPE Training with tqdm Progress Bar ---
def train_npe(prior_dist, is_nes_model_flag, fixed_w_s_for_nes_train=None, device='cpu'):
    from sbi.inference import SNPE_C as SNPE
    model_name = "NES" if is_nes_model_flag else "SimpleDDM"
    print(f"\n--- Training NPE for {model_name} model ---")
    print(f"Using {NPE_TRAINING_SIMS_ADVERSARIAL} simulations for training.")
    start_train_time = time.time()
    inference_obj = SNPE(prior=prior_dist, density_estimator='maf', device=device)
    wrapped_simulator = partial(sbi_simulator_wrapper,
                                is_nes_model_flag=is_nes_model_flag,
                                fixed_w_s_for_nes=fixed_w_s_for_nes_train)
    theta_train = prior_dist.sample((NPE_TRAINING_SIMS_ADVERSARIAL,))
    x_train_list = []
    try:
        from tqdm import trange
        pbar = trange(NPE_TRAINING_SIMS_ADVERSARIAL, desc=f"Simulating for {model_name} NPE training")
    except ImportError:
        pbar = range(NPE_TRAINING_SIMS_ADVERSARIAL)
    for idx in pbar:
        x_train_list.append(wrapped_simulator(theta_train[idx]))
    x_train = torch.stack(x_train_list)
    valid_training_mask = ~torch.all(torch.isnan(x_train) | (x_train == -999.0), dim=1)
    theta_train_valid = theta_train[valid_training_mask]
    x_train_valid = x_train[valid_training_mask].to(device)
    print(f"Using {len(theta_train_valid)} valid simulations for training {model_name} NPE.")
    if len(theta_train_valid) < NPE_TRAINING_SIMS_ADVERSARIAL * 0.1 :
        print(f"ERROR: Too few valid training simulations for {model_name}. Stopping.")
        return None, None
    try:
        density_estimator = inference_obj.append_simulations(theta_train_valid, x_train_valid).train()
        print(f"NPE training for {model_name} took: {time.time() - start_train_time:.2f}s")
    except Exception as e:
        print(f"WARNING: Exception during NPE training for {model_name}: {e}")
        density_estimator = None
    return inference_obj, density_estimator

# --- 5. Main Routine ---
if __name__ == "__main__":
    # Add argparse for command-line arguments
    import argparse
    try:
        parser = argparse.ArgumentParser(description="Adversarial Model Recovery (Colab/GPU)")
        parser.add_argument('--n_subjects', type=int, default=3, help='Number of subjects (default: 3)')
        parser.add_argument('--n_trials_per_sub', type=int, default=50, help='Number of trials per subject (default: 50)')
        parser.add_argument('--npe_training_sims_adversarial', type=int, default=200, help='Number of simulations for NPE training (default: 200)')
        parser.add_argument('--npe_num_posterior_samples', type=int, default=10, help='Number of posterior samples (default: 10)')
        parser.add_argument('--global_seed', type=int, default=42, help='Random seed (default: 42)')
        args = parser.parse_args()

        N_SUBJECTS = args.n_subjects
        N_TRIALS_PER_SUB = args.n_trials_per_sub
        NPE_TRAINING_SIMS_ADVERSARIAL = args.npe_training_sims_adversarial
        NPE_NUM_POSTERIOR_SAMPLES = args.npe_num_posterior_samples
        GLOBAL_SEED = args.global_seed

        tag = f"colab_subj{N_SUBJECTS}_trials{N_TRIALS_PER_SUB}_{int(time.time())}"
        output_directory = Path(BASE_PATH) / f"adversarial_recovery_results_colab_{tag}"
        output_directory.mkdir(parents=True, exist_ok=True)

        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

        print("="*60)
        print("Starting Adversarial Model Recovery (Colab/GPU)")
        print("="*60)

        # --- 1. Generate "Observed" Data from Simple DDM ---
        print("\n--- Generating 'Observed' Data from Simple DDM ---")
        simple_ddm_true_params_dict = {'a': 1.2}
        all_simple_ddm_data = []
        for s_idx in range(N_SUBJECTS):
            np.random.seed(GLOBAL_SEED + s_idx + 100)
            df_subj_simple = simulate_ddm_trials_from_params(
                simple_ddm_true_params_dict, N_TRIALS_PER_SUB,
                CONFLICT_LEVELS_ADV, CONFLICT_PROPORTIONS_ADV, BASE_SIM_PARAMS_ADV,
                is_nes_model=False
            )
            df_subj_simple['subj_idx'] = s_idx
            all_simple_ddm_data.append(df_subj_simple)
        observed_data_simple_ddm = pd.concat(all_simple_ddm_data).reset_index(drop=True)
        observed_summary_stats_simple_ddm = calculate_summary_stats(observed_data_simple_ddm)
        obs_sumstats_simple_ddm_tensor = torch.tensor(
            [observed_summary_stats_simple_ddm.get(k, np.nan) for k in get_summary_stat_keys()],
            dtype=torch.float32
        ).to(device)
        print("OBSERVED SUMMARY STATS (simple DDM):", obs_sumstats_simple_ddm_tensor.cpu().numpy())
        assert torch.isfinite(obs_sumstats_simple_ddm_tensor).all(), f"Non-finite observed summary stats: {obs_sumstats_simple_ddm_tensor.cpu().numpy()}"
        print(f"Generated {len(observed_data_simple_ddm)} trials from Simple DDM.")
        observed_data_simple_ddm.to_csv(output_directory / f"observed_data_from_simple_ddm_{tag}.csv", index=False)

        # --- 2. Train NPE for Simple DDM & NES Model ---
        sbi_prior_simple_ddm = BoxUniform(low=PRIOR_SIMPLE_DDM_LOW.to(device), high=PRIOR_SIMPLE_DDM_HIGH.to(device), device=device)
        npe_simple_ddm, density_simple_ddm = train_npe(sbi_prior_simple_ddm, is_nes_model_flag=False, device=device)
        sbi_prior_nes = BoxUniform(low=PRIOR_NES_LOW.to(device), high=PRIOR_NES_HIGH.to(device), device=device)
        npe_nes, density_nes = train_npe(sbi_prior_nes, is_nes_model_flag=True, fixed_w_s_for_nes_train=FIXED_W_S_FOR_NES_FIT, device=device)

        # --- Debug: NES Posterior Sampling with Dummy x ---
        if npe_nes is not None and density_nes is not None:
            try:
                print("Testing NES posterior sampling with REAL observed summary stats...")
                posterior_nes = npe_nes.build_posterior(density_nes)
                # Use actual observed data instead of zeros
                x_for_test = obs_sumstats_simple_ddm_tensor
                print("[DEBUG] x_for_test summary stats tensor before NES posterior sampling:", x_for_test.cpu().numpy())
                # Start with a small number of samples to verify it works
                test_samples = 10
                samples = posterior_nes.sample((test_samples,), x=x_for_test)
                print(f"  Got {test_samples} NES samples:", samples.shape)
                # If successful, try with the full number
                if NPE_NUM_POSTERIOR_SAMPLES > test_samples:
                    print(f"  Now sampling the full {NPE_NUM_POSTERIOR_SAMPLES} samples...")
                    samples_full = posterior_nes.sample((NPE_NUM_POSTERIOR_SAMPLES,), x=x_for_test)
                    print(f"  Got full {NPE_NUM_POSTERIOR_SAMPLES} NES samples:", samples_full.shape)
            except Exception as e:
                print("Exception during NES posterior debug sampling:", e)

        print("Workflow complete. Check output directory for results.")
    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc()

# --- 6. Colab Tips ---
# - To upload files: from google.colab import files; files.upload()
# - To download files: from google.colab import files; files.download('filename')
# - To use Google Drive: from google.colab import drive; drive.mount('/content/drive')
