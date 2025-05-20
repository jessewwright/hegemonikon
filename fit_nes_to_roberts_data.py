# reconstructed_fit_nes_to_roberts_data.py

# 1. STANDARD IMPORTS
import argparse
import logging
import sys
from pathlib import Path
import pickle # For saving/loading objects like NPE

import numpy as np
import pandas as pd
import torch
import seaborn as sns # For plotting
import matplotlib.pyplot as plt # For plotting

# 2. SBI Patches (MUST be before other sbi imports, based on memories)
# Memory [ad6ee20a-2ccc-4588-a10c-f62e11c32cc2] and [6a495677-249f-4782-909c-ba3fe2b11af5]
try:
    import sbi.utils.sbiutils as sbi_utils_module_to_patch
    from nflows.transforms.standard import IdentityTransform as NflowsIdentityTransform
    import torch.nn as nn # For nn.Identity
    
    # For standardizing_transform (expects nflows.Transform object)
    sbi_utils_module_to_patch.standardizing_transform = lambda batch_x, structured_x=None: NflowsIdentityTransform()
    logging.info("Applied monkey-patch to sbi.utils.sbiutils.standardizing_transform")

    # For standardizing_net (expects nn.Module returning single tensor)
    sbi_utils_module_to_patch.standardizing_net = lambda data_for_stats, structured=False: nn.Identity()
    logging.info("Applied monkey-patch to sbi.utils.sbiutils.standardizing_net")

    # Patch for build_maf internal standardizing_net, Memory [ad6ee20a-2ccc-4588-a10c-f62e11c32cc2]
    import sbi.neural_nets.net_builders.flow as sbi_flow_module_for_patching
    sbi_flow_module_for_patching.standardizing_net = lambda data, structured=False: nn.Identity()
    logging.info("Applied monkey-patch to sbi.neural_nets.net_builders.flow.standardizing_net")

except ImportError as e:
    logging.warning(f"Could not apply SBI monkey-patches due to import error: {e}. SBI behavior might be unexpected if using older versions or specific configurations.")
except AttributeError as e:
    logging.warning(f"Could not apply SBI monkey-patches due to attribute error: {e}. SBI behavior might be unexpected.")


# 3. SBI IMPORTS
from sbi.inference import SNPE
from sbi.utils import RestrictedPrior, MultipleIndependent, BoxUniform 
from sbi import utils as sbi_utils
from sbi.analysis import get_sbc_rank_plot, sbc_rank_stats
# from sbi.types import Shape, Array # These might be older sbi types, or internal.

# 4. PROJECT-SPECIFIC IMPORTS (Assumed based on function names, adjust as needed)
# Example: from your_project_utils import get_roberts_summary_stat_keys, calculate_summary_stats_roberts_final, fit_single_subject
# Example: from your_model_definitions import create_prior, dd_model_simulation_wrapper_for_sbi

# --- Placeholder for your actual function definitions if not imported ---
# Ensure these functions are defined or imported before they are called.
def get_roberts_summary_stat_keys():
    # Replace with your actual summary stat keys logic
    logging.info("Using placeholder get_roberts_summary_stat_keys() - REPLACE WITH ACTUAL IMPLEMENTATION")
    # Example: return ['stat1', 'stat2', ..., 'statN'] based on your ~49-51 stats
    # This needs to be accurate for D_for_model_x
    return [f'stat_{i}' for i in range(50)] # Placeholder for ~50 stats

def calculate_summary_stats_roberts_final(df_subject_data, param_names_for_stats, summary_stat_keys):
    logging.info("Using placeholder calculate_summary_stats_roberts_final() - REPLACE WITH ACTUAL IMPLEMENTATION")
    # param_names_for_stats is passed but might not be used if stats are purely data-driven
    # Should return: obs_sumstats_df_subj, n_subj_trials
    num_stats = len(summary_stat_keys)
    if not df_subject_data.empty:
        # Create a DataFrame with one row of random numbers for placeholder stats.
        stats_data = {key: [np.random.rand()] for key in summary_stat_keys}
        obs_sumstats_df_subj = pd.DataFrame(stats_data)
        n_subj_trials = len(df_subject_data)
    else:
        obs_sumstats_df_subj = pd.DataFrame(columns=summary_stat_keys) # Empty df with correct columns
        n_subj_trials = 0
    return obs_sumstats_df_subj, n_subj_trials


def fit_single_subject(subject_id_str, density_estimator_obj, obs_sumstats_tensor_subj, num_posterior_samples, device, param_names_for_fitting):
    logging.info("Using placeholder fit_single_subject() - REPLACE WITH ACTUAL IMPLEMENTATION")
    # Should return a dictionary of results
    num_params = len(param_names_for_fitting)
    # Dummy posterior samples
    posterior_samples = np.random.rand(num_posterior_samples, num_params) 
    results = {'subject_id': subject_id_str, 'posterior_samples': posterior_samples}
    for i, p_name in enumerate(param_names_for_fitting):
        results[f'mean_{p_name}'] = np.mean(posterior_samples[:, i])
        results[f'median_{p_name}'] = np.median(posterior_samples[:, i])
        results[f'std_{p_name}'] = np.std(posterior_samples[:, i])
    # Add placeholders for framing effects if those keys are expected by downstream processing
    results['framing_effect_ntc'] = np.nan 
    results['framing_effect_tc'] = np.nan
    results['framing_effect_avg'] = np.nan
    return results

def sbi_simulator_roberts(params_set_numpy, template_trials_for_sim, current_summary_stat_keys, current_param_names):
    logging.info("Using placeholder sbi_simulator_roberts() - REPLACE WITH ACTUAL IMPLEMENTATION")
    # params_set_numpy is (N_sims, N_params)
    # Should return a tensor of summary statistics (N_sims, D_for_model_x).
    num_simulations = params_set_numpy.shape[0]
    num_summary_stats = len(current_summary_stat_keys)
    dummy_summary_stats = np.random.randn(num_simulations, num_summary_stats)
    # Simulate some NaNs occasionally for robustness checks if desired, but not by default
    # if np.random.rand() < 0.01: dummy_summary_stats[np.random.randint(num_simulations), np.random.randint(num_summary_stats)] = np.nan
    return torch.tensor(dummy_summary_stats, dtype=torch.float32)

# --- END Placeholder for function definitions ---


# 5. GLOBAL CONSTANTS AND CONFIGURATIONS
DEFAULT_TEMPLATE_TRIALS = 100 # As requested for SBC consistency and filenames
BASE_OUTPUT_DIR = Path("./hpc_results_nes_roberts") 
OUTPUT_DIR_SBC = BASE_OUTPUT_DIR / "sbc"
OUTPUT_DIR_EMPIRICAL = BASE_OUTPUT_DIR / "empirical"
DEFAULT_NPE_FILENAME_TEMPLATE = "{model_type}_npe_sims{sims}_template{template}.pt"

# Create output directories if they don't exist
OUTPUT_DIR_SBC.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR_SBC / 'plots').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR_SBC / 'data').mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_EMPIRICAL.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR_EMPIRICAL / 'plots').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR_EMPIRICAL / 'data').mkdir(parents=True, exist_ok=True)


def main():
    # 6. ARGUMENT PARSING
    parser = argparse.ArgumentParser(description="Fit NES to Roberts data and/or run SBC for DD model.")
    
    # SBC arguments
    parser.add_argument('--run_sbc', action='store_true', help='Run Simulation-Based Calibration.')
    parser.add_argument('--sbc_only', action='store_true', help='Run SBC and then exit (implies --run_sbc).')
    parser.add_argument('--sbc_datasets', type=int, default=300, help='Number of datasets (ground truth parameters) for SBC.')
    parser.add_argument('--sbc_npe_posterior_samples', type=int, default=2000, help='Number of posterior samples for SBC.')
    
    # NPE Training arguments
    parser.add_argument('--retrain_npe', action='store_true', help='Retrain the NPE, even if a file exists.')
    parser.add_argument('--npe_train_sims', type=int, default=30000, help='Number of simulations for NPE training.')
    parser.add_argument('--npe_file', type=str, default=None, help='Path to a pre-trained NPE file to load/save. If None, a default name is generated.')
    parser.add_argument('--npe_model_type', type=str, default='maf', choices=['maf', 'nsf', 'mdn'], help='Type of density estimator for NPE.')
    parser.add_argument('--maf_hidden_features', type=int, default=50, help='Number of hidden features for MAF.')
    parser.add_argument('--maf_num_transforms', type=int, default=5, help='Number of transforms for MAF.')

    # Empirical Fitting arguments
    parser.add_argument('--fit_empirical', action='store_true', help='Fit the model to empirical Roberts data.')
    parser.add_argument('--roberts_data_file', type=str, default="roberts_data.csv", help='Path to the Roberts data CSV file.')
    parser.add_argument('--subject_ids', type=str, default=None, help='Comma-separated list of subject IDs to fit (e.g., "1,2,5"). Default all.')
    parser.add_argument('--npe_posterior_samples', type=int, default=10000, help='Number of posterior samples for empirical fitting.')

    # General arguments
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if CUDA is available.')

    args = parser.parse_args()

    if args.sbc_only:
        args.run_sbc = True # sbc_only implies run_sbc
    
    if not args.run_sbc and not args.fit_empirical and not args.retrain_npe and args.npe_file is None:
        print("No action specified. Please use --run_sbc, --fit_empirical, --retrain_npe, or provide an --npe_file to load for fitting.")
        parser.print_help()
        sys.exit(1)
        
    # 7. INITIAL SETUP BASED ON ARGS
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_numeric, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s', force=True)
    logger = logging.getLogger(__name__)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available() and not args.force_cpu:
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info(f"Using device: {device}")

    npe_file_path_str = args.npe_file if args.npe_file else DEFAULT_NPE_FILENAME_TEMPLATE.format(
        model_type=args.npe_model_type, 
        sims=args.npe_train_sims, 
        template=DEFAULT_TEMPLATE_TRIALS
    )
    npe_file_path = BASE_OUTPUT_DIR / npe_file_path_str
    logger.info(f"NPE file path set to: {npe_file_path}")

    # 8. INITIALIZE KEY VARIABLES
    prior_sbc_dist = None
    prior_empirical_dist = None 
    param_names_sbc = []
    param_names_empirical = []
    density_estimator_empirical_obj = None 
    
    SUMMARY_STAT_KEYS = get_roberts_summary_stat_keys()
    D_for_model_x = len(SUMMARY_STAT_KEYS)
    logger.info(f"Number of summary statistics (D_for_model_x): {D_for_model_x}")
    if D_for_model_x <= 0:
        logger.error("CRITICAL: Number of summary statistics is <= 0. Cannot proceed.")
        sys.exit(1)

    # --- Define Priors (Example structure) ---
    # Replace with your actual parameter names and ranges for Roberts data fitting
    # These names must be consistent with your sbi_simulator_roberts and stat calculations
    roberts_param_details = {
        'v': (0.0, 5.0), 'a': (0.5, 3.0), 't0': (0.1, 0.5),
        # 'k': (0.0, 1.0), 'beta':(0.1, 0.9) # Add other params as needed
    }
    roberts_param_names = list(roberts_param_details.keys())
    roberts_lows = torch.tensor([roberts_param_details[p][0] for p in roberts_param_names], dtype=torch.float32, device=device)
    roberts_highs = torch.tensor([roberts_param_details[p][1] for p in roberts_param_names], dtype=torch.float32, device=device)

    if not roberts_param_names:
        logger.error("Prior parameters for Roberts fitting not defined. Please define 'roberts_param_details'.")
        sys.exit(1)
        
    prior_empirical_dist = BoxUniform(low=roberts_lows, high=roberts_highs, device=device.type) 
    param_names_empirical = roberts_param_names[:]
    logger.info(f"Empirical Prior defined for parameters: {param_names_empirical} with ranges on device {device.type}")

    # Prior for SBC might be the same or different. For now, assume same for simplicity.
    prior_sbc_dist = BoxUniform(low=roberts_lows, high=roberts_highs, device=device.type)
    param_names_sbc = roberts_param_names[:]
    logger.info(f"SBC Prior defined for parameters: {param_names_sbc} with ranges on device {device.type}")
    
    density_estimator_build_kwargs = {
        'z_score_x': 'none', 
        'z_score_y': False, 
    }
    if args.npe_model_type == 'maf':
        density_estimator_build_kwargs['hidden_features'] = args.maf_hidden_features
        density_estimator_build_kwargs['num_transforms'] = args.maf_num_transforms

    # --- NPE Training or Loading ---
    # The NPE used for empirical fitting and for SBC rank calculation is the same one.
    if args.retrain_npe or not npe_file_path.exists():
        if not args.retrain_npe and not npe_file_path.exists():
            logger.info(f"NPE file {npe_file_path} not found and --retrain_npe not specified. Will train a new NPE.")
        
        logger.info(f"Training new NPE with {args.npe_train_sims} simulations. Model: {args.npe_model_type}")
        training_prior_to_use = prior_empirical_dist # Use the empirical prior for training
        training_param_names_to_use = param_names_empirical

        if training_prior_to_use is None or not training_param_names_to_use:
            logger.error("Training prior or parameter names not set up for NPE training. Exiting.")
            sys.exit(1)

        def sbi_training_simulator_wrapper(parameter_sample_batch_tensor):
            return sbi_simulator_roberts(
                params_set_numpy=parameter_sample_batch_tensor.cpu().numpy(), 
                template_trials_for_sim=DEFAULT_TEMPLATE_TRIALS,
                current_summary_stat_keys=SUMMARY_STAT_KEYS,
                current_param_names=training_param_names_to_use
            )

        logger.info(f"Generating {args.npe_train_sims} simulations for NPE training...")
        theta_train, x_train = sbi_utils.simulate_for_sbi(
            simulation_wrapper=sbi_training_simulator_wrapper,
            proposal=training_prior_to_use,
            num_simulations=args.npe_train_sims,
            num_workers=1, 
            show_progress_bar=True,
            simulation_batch_size=1000 # Example, adjust as needed
        )
        logger.info(f"Generated training data: theta_train shape {theta_train.shape}, x_train shape {x_train.shape}")

        if torch.isnan(theta_train).any() or torch.isinf(theta_train).any():
            logger.error("NaNs or Infs found in training parameters (theta_train). Check prior and simulator."); sys.exit(1)
        if torch.isnan(x_train).any() or torch.isinf(x_train).any():
            logger.error("NaNs or Infs found in training summary stats (x_train). Check simulator and summary stat calculation."); sys.exit(1)

        density_estimator_empirical_obj = SNPE(
            prior=training_prior_to_use,
            density_estimator=args.npe_model_type,
            device=device.type, 
            logging_level=args.log_level.upper(),
            density_estimator_build_kwargs=density_estimator_build_kwargs
        )
        
        logger.info(f"Starting NPE training with {args.npe_model_type}...")
        _ = density_estimator_empirical_obj.append_simulations(theta_train, x_train).train(show_train_summary=True)
        logger.info("NPE training finished.")

        npe_checkpoint = {
            'model_state_dict': density_estimator_empirical_obj.neural_net.state_dict(),
            'prior_params': {'low': training_prior_to_use.support.base_dist.low.cpu().numpy(), 
                             'high': training_prior_to_use.support.base_dist.high.cpu().numpy()},
            'param_names': training_param_names_to_use,
            'num_summary_stats': D_for_model_x, 
            'summary_stat_keys': SUMMARY_STAT_KEYS,
            'npe_model_type': args.npe_model_type,
            'sbi_version': getattr(__import__('sbi'), '__version__', 'unknown'),
            'npe_train_sims': args.npe_train_sims,
            'template_trials_for_training': DEFAULT_TEMPLATE_TRIALS,
            'density_estimator_build_kwargs': density_estimator_build_kwargs,
            'maf_hidden_features': args.maf_hidden_features if args.npe_model_type == 'maf' else None,
            'maf_num_transforms': args.maf_num_transforms if args.npe_model_type == 'maf' else None,
            'training_seed': args.seed
        }
        torch.save(npe_checkpoint, npe_file_path)
        logger.info(f"Saved trained NPE to {npe_file_path}")

    else: # Load existing NPE
        logger.info(f"Loading NPE from {npe_file_path}")
        try:
            checkpoint = torch.load(npe_file_path, map_location=device)

            # --- SAFER LOADING: Create fresh prior, do NOT unpickle prior from checkpoint ---
            # Use current script's roberts_lows and roberts_highs, which are already on correct device
            fresh_prior = BoxUniform(low=roberts_lows, high=roberts_highs, device=device.type)

            # Get necessary metadata from checkpoint
            density_estimator_state_dict = checkpoint['model_state_dict']
            num_summary_stats_trained = checkpoint.get('num_summary_stats')
            if num_summary_stats_trained is None:
                logger.error("'num_summary_stats' missing from checkpoint."); sys.exit(1)

            # Check consistency with current stat definition
            current_num_stats = len(SUMMARY_STAT_KEYS)
            if current_num_stats != num_summary_stats_trained:
                logger.error(f"CRITICAL MISMATCH: Loaded NPE expects {num_summary_stats_trained} stats, current script defines {current_num_stats}. You MUST align get_roberts_summary_stat_keys() in this script to produce exactly {num_summary_stats_trained} stats in the original order.")
                sys.exit(1)

            loaded_build_kwargs = checkpoint.get('density_estimator_build_kwargs', density_estimator_build_kwargs)

            # Instantiate SNPE with the fresh prior
            npe_empirical_object = SNPE(
                prior=fresh_prior,
                density_estimator=checkpoint.get('npe_model_type', args.npe_model_type),
                device=device.type,
                logging_level=args.log_level.upper(),
                density_estimator_build_kwargs=loaded_build_kwargs
            )

            # Build dummy network and load state dict
            dummy_theta = fresh_prior.sample((2,))
            dummy_x = torch.randn(2, num_summary_stats_trained, device=device)
            _ = npe_empirical_object.append_simulations(dummy_theta, dummy_x).train(max_num_epochs=0)
            npe_empirical_object.neural_net.load_state_dict(density_estimator_state_dict, strict=False)
            npe_empirical_object.neural_net.eval()
            logger.info("Successfully loaded and reconstructed NPE using saved state_dict and fresh prior.")
            density_estimator_empirical_obj = npe_empirical_object

        except FileNotFoundError:
            logger.error(f"NPE file {npe_file_path} not found and --retrain_npe is False."); sys.exit(1)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load/reconstruct NPE from {npe_file_path}: {e}.", exc_info=True)
            if "size mismatch" in str(e): logger.error("Size mismatch: D_for_model_x or param dimensions differ.")
            sys.exit(1)

    # --- EMPIRICAL FITTING --- (This is the block you provided)
    df_roberts = None
    if args.fit_empirical:
        try:
            df_roberts = pd.read_csv(args.roberts_data_file)
            logger.info(f"Loaded Roberts empirical data from {args.roberts_data_file}. Shape: {df_roberts.shape}")
        except FileNotFoundError:
            logger.error(f"Roberts data file {args.roberts_data_file} not found. Cannot perform empirical fitting.")
            if not args.run_sbc: sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading Roberts data: {e}")
            if not args.run_sbc: sys.exit(1)
    
    subject_ids_to_fit_str = []
    if args.fit_empirical and df_roberts is not None:
        if args.subject_ids:
            subject_ids_to_fit_str = [s.strip() for s in args.subject_ids.split(',')]
        elif 'subject' in df_roberts.columns:
            subject_ids_to_fit_str = [str(s) for s in sorted(df_roberts['subject'].unique())]
        else:
            logger.warning("No 'subject' column in Roberts data and no --subject_ids provided.")
        logger.info(f"Subjects selected for fitting: {subject_ids_to_fit_str if subject_ids_to_fit_str else 'None'}")

    # --- Subject Fitting Loop (from user) ---
    if args.fit_empirical and density_estimator_empirical_obj is not None and df_roberts is not None and subject_ids_to_fit_str:
        logger.info("Starting empirical subject fitting loop...")
        if not param_names_empirical:
            logger.error("param_names_empirical is empty before fitting loop. Exiting."); sys.exit(1)
        logger.info(f"Using param_names_empirical for fitting: {param_names_empirical}")

        all_subjects_results = []
        for subject_idx, subject_id_str in enumerate(subject_ids_to_fit_str):
            subject_id_int = int(subject_id_str)
            logger.info(f"\n{'='*80}\nPROCESSING SUBJECT {subject_id_int} ({subject_idx + 1}/{len(subject_ids_to_fit_str)})\n{'='*80}")
            
            df_subject_data = df_roberts[df_roberts['subject'] == subject_id_int]
            if df_subject_data.empty:
                logger.warning(f"No data found for subject {subject_id_int}. Skipping.")
                all_subjects_results.append({'subject_id': subject_id_str, 'error': 'No data found', 'posterior_samples': np.array([])})
                continue

            obs_sumstats_df_subj, n_subj_trials = calculate_summary_stats_roberts_final(
                df_subject_data, param_names_empirical, summary_stat_keys=SUMMARY_STAT_KEYS
            )
            
            if obs_sumstats_df_subj.empty or obs_sumstats_df_subj.isnull().all().all():
                logger.warning(f"Empty or all-NaN summary stats for subject {subject_id_int} (n_trials={n_subj_trials}). Skipping.")
                if not obs_sumstats_df_subj.empty and obs_sumstats_df_subj.isnull().any().any():
                    logger.warning(f"NaNs found in summary stats for subject {subject_id_int}: \n{obs_sumstats_df_subj[obs_sumstats_df_subj.isnull().any(axis=1)]}")
                all_subjects_results.append({'subject_id': subject_id_str, 'error': 'Empty or NaN summary stats', 'posterior_samples': np.array([])})
                continue

            obs_sumstats_clean = obs_sumstats_df_subj.iloc[0].values
            if np.isnan(obs_sumstats_clean).any():
                logger.warning(f"NaNs found in summary statistics numpy array for subject {subject_id_int}. Stats: {obs_sumstats_clean}")
                all_subjects_results.append({'subject_id': subject_id_str, 'error': 'NaNs in calculated summary stats array', 'posterior_samples': np.array([])})
                continue

            if D_for_model_x is None: 
                logger.error("CRITICAL: D_for_model_x is None before subject fitting loop."); sys.exit(1)

            num_calculated_stats = len(obs_sumstats_clean)
            obs_sumstats_for_fitting = obs_sumstats_clean
            if num_calculated_stats != D_for_model_x:
                logger.warning(f"Subj {subject_id_int}: Calc stats dim ({num_calculated_stats}) != NPE expected ({D_for_model_x}).")
                if num_calculated_stats > D_for_model_x:
                    obs_sumstats_for_fitting = obs_sumstats_clean[:D_for_model_x]
                    logger.info(f"    Using first {D_for_model_x} calculated summary statistics.")
                else: 
                    logger.error(f"    Calculated only {num_calculated_stats} stats vs {D_for_model_x} needed. Skipping subject.")
                    all_subjects_results.append({'subject_id': subject_id_str, 'error': 'Insufficient summary stats calculated', 'posterior_samples': np.array([])})
                    continue 
            
            logger.info(f"Subject {subject_id_int}: Using summary stats of shape {obs_sumstats_for_fitting.shape} for fitting.")
            obs_sumstats_tensor_subj = torch.tensor(obs_sumstats_for_fitting, dtype=torch.float32).to(device)

            if torch.isnan(obs_sumstats_tensor_subj).any():
                logger.error(f"NaNs in obs_sumstats_tensor_subj for subject {subject_id_int} before .sample().")
                all_subjects_results.append({'subject_id': subject_id_str, 'error': 'NaNs in summary stats tensor for fitting', 'posterior_samples': np.array([])})
                continue

            try:
                subject_results = fit_single_subject(
                    subject_id_str,
                    density_estimator_empirical_obj, 
                    obs_sumstats_tensor_subj, 
                    num_posterior_samples=args.npe_posterior_samples,
                    device=device,
                    param_names_for_fitting=param_names_empirical # Important: use consistent param_names
                )
                all_subjects_results.append(subject_results)
            except Exception as e_fit_subject:
                logger.error(f"Error processing subject {subject_id_str} in fit_single_subject: {e_fit_subject}", exc_info=True)
                error_entry = {'subject_id': subject_id_str, 'error': str(e_fit_subject), 'posterior_samples': np.array([])}
                for p_name in param_names_empirical:
                    error_entry[f'mean_{p_name}'] = np.nan
                    error_entry[f'median_{p_name}'] = np.nan
                    error_entry[f'std_{p_name}'] = np.nan
                error_entry['framing_effect_ntc'] = np.nan 
                error_entry['framing_effect_tc'] = np.nan
                error_entry['framing_effect_avg'] = np.nan
                all_subjects_results.append(error_entry)
        
        if all_subjects_results:
            df_all_fitted_params = pd.DataFrame(all_subjects_results)
            output_data_dir = OUTPUT_DIR_EMPIRICAL / 'data'
            # Use stem of npe_file_path for a cleaner filename part if it's a path
            npe_name_part = Path(npe_file_path).stem if Path(npe_file_path).name != npe_file_path_str else npe_file_path_str.replace('.pt','')
            results_filename_final = output_data_dir / f"roberts_fitted_nes_params_{npe_name_part}_template{DEFAULT_TEMPLATE_TRIALS}.csv"
            df_all_fitted_params.to_csv(results_filename_final, index=False, float_format='%.4f')
            logger.info(f"Saved all subject fitting results to {results_filename_final}")

            if not df_all_fitted_params.empty:
                try:
                    plot_cols = [f'mean_{p}' for p in param_names_empirical if f'mean_{p}' in df_all_fitted_params.columns]
                    if plot_cols: # Ensure there are columns to plot
                        numeric_plot_cols_df = df_all_fitted_params[plot_cols].select_dtypes(include=np.number)
                        if not numeric_plot_cols_df.empty:
                            pairplot_fig = sns.pairplot(numeric_plot_cols_df)
                            plots_dir = OUTPUT_DIR_EMPIRICAL / 'plots'
                            pairplot_filename = plots_dir / f'parameter_means_pairplot_{npe_name_part}.png'
                            pairplot_fig.savefig(pairplot_filename)
                            logger.info(f"Saved parameter means pairplot to {pairplot_filename}")
                            plt.close(pairplot_fig)
                        else:
                            logger.warning("No numeric columns found for pairplot of parameter means after selecting mean columns.")
                    else:
                        logger.warning("No 'mean_<param>' columns found in the results for pairplot.")
                except Exception as e_plot:
                    logger.error(f"Plotting parameter means failed: {e_plot}", exc_info=True)
        else:
            logger.info("No subjects were successfully processed or no results to save for empirical fitting.")
    
    elif not args.fit_empirical:
        logger.info("Skipping empirical fitting as --fit_empirical is not set.")
    elif density_estimator_empirical_obj is None:
        logger.info("density_estimator_empirical_obj is None, skipping empirical subject fitting loop.")
    elif df_roberts is None:
        logger.info("Empirical data (df_roberts) not loaded, skipping empirical subject fitting loop.")
    elif not subject_ids_to_fit_str:
        logger.info("No subjects to fit, skipping empirical subject fitting loop.")

    # --- SBC Logic (if enabled) ---
    if args.run_sbc:
        logger.info(f"Starting Simulation-Based Calibration (SBC) with {args.sbc_datasets} datasets...")
        if density_estimator_empirical_obj is None or prior_sbc_dist is None or not param_names_sbc:
            logger.error("NPE, SBC prior, or SBC param_names not available for SBC. Cannot proceed.")
            sys.exit(1)

        theta_test_sbc = prior_sbc_dist.sample((args.sbc_datasets,))
        logger.info(f"Generated {args.sbc_datasets} ground truth parameter sets (theta_test_sbc) for SBC. Shape: {theta_test_sbc.shape}")

        def sbi_sbc_simulator_wrapper(parameter_sample_batch_tensor):
            return sbi_simulator_roberts(
                params_set_numpy=parameter_sample_batch_tensor.cpu().numpy(),
                template_trials_for_sim=DEFAULT_TEMPLATE_TRIALS, 
                current_summary_stat_keys=SUMMARY_STAT_KEYS,
                current_param_names=param_names_sbc
            )
        
        logger.info(f"Generating {args.sbc_datasets} observation sets (x_test_sbc) for SBC...")
        x_test_sbc = sbi_sbc_simulator_wrapper(theta_test_sbc.to(device))
        logger.info(f"Generated x_test_sbc. Shape: {x_test_sbc.shape}")

        if torch.isnan(x_test_sbc).any() or torch.isinf(x_test_sbc).any():
            logger.error("NaNs or Infs found in SBC observations (x_test_sbc). Check simulator."); sys.exit(1)

        sbc_posterior_estimator = density_estimator_empirical_obj

        logger.info(f"Calculating SBC ranks for {args.sbc_datasets} datasets...")
        ranks = sbc_rank_stats(
            theta_test_sbc.to(device), 
            x_test_sbc.to(device), 
            sbc_posterior_estimator, 
            num_posterior_samples=args.sbc_npe_posterior_samples,
            show_progress_bar=True
        )
        logger.info(f"SBC ranks calculated. Shape: {ranks.shape}")

        sbc_output_data_dir = OUTPUT_DIR_SBC / 'data'
        sbc_npe_name_part = Path(npe_file_path).stem if Path(npe_file_path).name != npe_file_path_str else npe_file_path_str.replace('.pt','')
        sbc_ranks_filename = sbc_output_data_dir / f"sbc_ranks_{sbc_npe_name_part}_template{DEFAULT_TEMPLATE_TRIALS}_datasets{args.sbc_datasets}.csv"
        df_ranks = pd.DataFrame(ranks.cpu().numpy(), columns=[f"rank_{p}" for p in param_names_sbc])
        df_ranks.to_csv(sbc_ranks_filename, index=False)
        logger.info(f"Saved SBC ranks to {sbc_ranks_filename}")
        
        df_theta_test_sbc = pd.DataFrame(theta_test_sbc.cpu().numpy(), columns=[f"true_{p}" for p in param_names_sbc])
        sbc_true_params_filename = sbc_output_data_dir / f"sbc_true_params_{sbc_npe_name_part}_template{DEFAULT_TEMPLATE_TRIALS}_datasets{args.sbc_datasets}.csv"
        df_theta_test_sbc.to_csv(sbc_true_params_filename, index=False)
        logger.info(f"Saved SBC true parameters to {sbc_true_params_filename}")

        try:
            fig, axes = plt.subplots(1, len(param_names_sbc), figsize=(5 * len(param_names_sbc), 4), squeeze=False)
            plot_sbc_rank_plot = get_sbc_rank_plot(ranks.T, param_names_sbc, fig=fig, axes=axes.flatten()) # ranks.T might be needed
            sbc_plot_filename = OUTPUT_DIR_SBC / 'plots' / f"sbc_ranks_plot_{sbc_npe_name_part}_template{DEFAULT_TEMPLATE_TRIALS}_datasets{args.sbc_datasets}.png"
            fig.savefig(sbc_plot_filename)
            plt.close(fig)
            logger.info(f"Saved SBC rank plot to {sbc_plot_filename}")
        except Exception as e_sbc_plot:
            logger.error(f"Error plotting SBC ranks: {e_sbc_plot}", exc_info=True)

    else:
        logger.info("Skipping SBC logic as --run_sbc is not specified.")

    if args.sbc_only:
        logger.info("SBC_only is True. Exiting after SBC completion (if run).")
        sys.exit(0)
        
    logger.info("Script execution finished.")

if __name__ == "__main__":
    main()