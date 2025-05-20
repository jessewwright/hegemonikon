#!/usr/bin/env python
# Filename: analyze_5param_correlations.py
# Purpose: Analyze parameters from the 5-parameter NES model fit and their correlations
#          with behavioral measures from the Roberts dataset.

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
fitted_params_file = "sbc_nes_roberts_5param_alpha_gain/empirical_fit_results/empirical_fitting_results.csv"
original_data_file = "Roberts_Framing_Data/ftp_osf_data.csv"
output_dir = "sbc_nes_roberts_5param_alpha_gain/correlation_analysis"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    logging.info(f"Loading fitted parameters from {fitted_params_file}")
    fitted_params_df = pd.read_csv(fitted_params_file)
    
    logging.info(f"Loading original Roberts data from {original_data_file}")
    original_data_df = pd.read_csv(original_data_file)
    
    # Print first few records from the original data to debug
    logging.info("Original data columns:")
    logging.info(f"  - {list(original_data_df.columns)[:10]}...")
    
    # Standardize column names if needed
    column_mapping = {}
    if 'subject' in original_data_df.columns and 'subject_id' not in original_data_df.columns:
        column_mapping['subject'] = 'subject_id'
    
    if column_mapping:
        original_data_df = original_data_df.rename(columns=column_mapping)
        logging.info(f"Renamed columns: {column_mapping}")
    
    # Calculate behavioral metrics from the original data
    logging.info("Calculating behavioral metrics from the original data")
    
    # Convert subject_id to the same type for merging
    if 'subject_id' in original_data_df.columns:
        original_data_df['subject_id'] = original_data_df['subject_id'].astype(str)
    
    fitted_params_df['subject_id'] = fitted_params_df['subject_id'].astype(str)
    
    # Get behavioral metrics per subject
    behavioral_metrics = calculate_behavioral_metrics(original_data_df)
    
    # Merge behavioral metrics with fitted parameters
    if not behavioral_metrics.empty:
        logging.info(f"Merging behavioral metrics with fitted parameters")
        fitted_params_df = pd.merge(fitted_params_df, behavioral_metrics, on='subject_id', how='left')
    
    return fitted_params_df, original_data_df

def calculate_behavioral_metrics(df):
    """Calculate behavioral metrics for each subject from the raw data."""
    if 'subject_id' not in df.columns or 'frame' not in df.columns or 'cond' not in df.columns:
        logging.warning("Original data missing required columns (subject_id, frame, cond)")
        return pd.DataFrame()
    
    # Determine which columns to use for trial type and choice
    trial_type_col = 'trialType' if 'trialType' in df.columns else None
    choice_col = 'choice' if 'choice' in df.columns else ('outcome' if 'outcome' in df.columns else None)
    
    if trial_type_col is None or choice_col is None:
        logging.warning(f"Missing required columns. Available: {list(df.columns)}")
        return pd.DataFrame()
    
    # Filter to only include target trials if possible
    if trial_type_col and 'target' in df[trial_type_col].unique():
        df = df[df[trial_type_col] == 'target'].copy()
        logging.info(f"Filtered to {len(df)} target trials")
    
    # Examine values in the choice/outcome column
    choice_values = df[choice_col].unique()
    logging.info(f"Choice values: {choice_values}")
    
    # Determine how to categorize gambles based on available data
    if choice_col == 'choice':
        # Assuming 1 = gamble, 0 = sure option (common coding)
        df['did_gamble'] = df[choice_col] == 1
    else:  # outcome column
        risky_values = [val for val in choice_values if 'risk' in str(val).lower()]
        df['did_gamble'] = df[choice_col].isin(risky_values)
    
    # Create condition flags
    df['is_tc'] = df['cond'] == 'tc'
    df['is_gain'] = df['frame'] == 'gain'
    
    # Calculate metrics per subject and condition
    metrics_list = []
    
    for subject_id, subject_data in df.groupby('subject_id'):
        subject_metrics = {'subject_id': subject_id}
        
        # Overall gambling proportion
        subject_metrics['prop_gamble_overall_obs'] = subject_data['did_gamble'].mean()
        
        # Gambling proportion by condition
        for frame in ['gain', 'loss']:
            for time_constraint in ['tc', 'ntc']:
                condition_mask = (subject_data['frame'] == frame) & (subject_data['cond'] == time_constraint)
                condition_data = subject_data[condition_mask]
                
                if len(condition_data) > 0:
                    cond_name = f"{frame.capitalize()}_{time_constraint.upper()}"
                    subject_metrics[f"prop_gamble_{cond_name}_obs"] = condition_data['did_gamble'].mean()
                    
                    # RT metrics if available
                    # Check various RT column naming conventions
                    rt_col = None
                    for possible_rt_col in ['rt', 'RT', 'response_time', 'responseTime']:
                        if possible_rt_col in condition_data.columns:
                            rt_col = possible_rt_col
                            break
                    
                    if rt_col:
                        subject_metrics[f"mean_rt_{cond_name}_obs"] = condition_data[rt_col].mean()
        
        # Calculate framing effects
        if all(f"prop_gamble_{c}_obs" in subject_metrics for c in ['Gain_TC', 'Loss_TC']):
            subject_metrics['framing_effect_tc_obs'] = subject_metrics['prop_gamble_Loss_TC_obs'] - subject_metrics['prop_gamble_Gain_TC_obs']
        
        if all(f"prop_gamble_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Loss_NTC']):
            subject_metrics['framing_effect_ntc_obs'] = subject_metrics['prop_gamble_Loss_NTC_obs'] - subject_metrics['prop_gamble_Gain_NTC_obs']
        
        # RT contrasts
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_TC', 'Loss_TC']):
            subject_metrics['rt_GvL_TC_obs'] = subject_metrics['mean_rt_Gain_TC_obs'] - subject_metrics['mean_rt_Loss_TC_obs']
        
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Loss_NTC']):
            subject_metrics['rt_GvL_NTC_obs'] = subject_metrics['mean_rt_Gain_NTC_obs'] - subject_metrics['mean_rt_Loss_NTC_obs']
        
        if all(f"mean_rt_{c}_obs" in subject_metrics for c in ['Gain_NTC', 'Gain_TC']):
            subject_metrics['gain_rt_speedup_obs'] = subject_metrics['mean_rt_Gain_NTC_obs'] - subject_metrics['mean_rt_Gain_TC_obs']
        
        # Average framing effect
        if 'framing_effect_tc_obs' in subject_metrics and 'framing_effect_ntc_obs' in subject_metrics:
            subject_metrics['framing_effect_avg_obs'] = (subject_metrics['framing_effect_tc_obs'] + subject_metrics['framing_effect_ntc_obs']) / 2
            subject_metrics['delta_framing_obs'] = subject_metrics['framing_effect_tc_obs'] - subject_metrics['framing_effect_ntc_obs']
        
        metrics_list.append(subject_metrics)
    
    if not metrics_list:
        logging.warning("No behavioral metrics could be calculated")
        return pd.DataFrame()
    
    return pd.DataFrame(metrics_list)

def calculate_derived_measures(df):
    """Calculate derived behavioral measures if they're not already in the DataFrame."""
    # Delta Framing (difference between framing effects in TC vs NTC)
    if 'framing_effect_tc_obs' in df.columns and 'framing_effect_ntc_obs' in df.columns:
        df['delta_framing_obs'] = df['framing_effect_tc_obs'] - df['framing_effect_ntc_obs']
        logging.info("Calculated delta_framing_obs")
    else:
        logging.warning("Couldn't calculate delta_framing_obs (missing required columns)")
    
    # Gain RT Speedup (difference in RT between Gain NTC and TC)
    if 'mean_rt_Gain_NTC_obs' in df.columns and 'mean_rt_Gain_TC_obs' in df.columns:
        df['gain_rt_speedup_obs'] = df['mean_rt_Gain_NTC_obs'] - df['mean_rt_Gain_TC_obs']
        logging.info("Calculated gain_rt_speedup_obs")
    else:
        logging.warning("Couldn't calculate gain_rt_speedup_obs (missing required columns)")
    
    # RT Gain vs Loss in TC condition
    if 'mean_rt_Gain_TC_obs' in df.columns and 'mean_rt_Loss_TC_obs' in df.columns:
        df['rt_GvL_TC_obs'] = df['mean_rt_Gain_TC_obs'] - df['mean_rt_Loss_TC_obs']
        logging.info("Calculated rt_GvL_TC_obs")
    else:
        logging.warning("Couldn't calculate rt_GvL_TC_obs (missing required columns)")
    
    # RT Gain vs Loss in NTC condition
    if 'mean_rt_Gain_NTC_obs' in df.columns and 'mean_rt_Loss_NTC_obs' in df.columns:
        df['rt_GvL_NTC_obs'] = df['mean_rt_Gain_NTC_obs'] - df['mean_rt_Loss_NTC_obs']
        logging.info("Calculated rt_GvL_NTC_obs")
    else:
        logging.warning("Couldn't calculate rt_GvL_NTC_obs (missing required columns)")
    
    return df

def calculate_and_print_correlation(df, param_col, metric_col):
    """Calculate and print correlation between parameter and behavioral measure."""
    # Drop rows with NaNs in the specific columns being correlated
    cleaned_df = df[[param_col, metric_col]].dropna()
    if len(cleaned_df) < 3:  # Need at least 3 data points for a meaningful correlation
        logging.warning(f"Correlation between {param_col} and {metric_col}: Not enough data points ({len(cleaned_df)})")
        return None, None, None
    
    r, p_value = pearsonr(cleaned_df[param_col], cleaned_df[metric_col])
    logging.info(f"Correlation between {param_col} and {metric_col}: r = {r:.3f}, p = {p_value:.4f} (N={len(cleaned_df)})")
    return r, p_value, len(cleaned_df)

def analyze_v_norm_correlations(df):
    """Analyze correlations between v_norm and behavioral measures."""
    logging.info("\n--- v_norm Correlations ---")
    v_norm_col = 'v_norm_mean' # Correct column name in the dataset
    
    if v_norm_col not in df.columns:
        logging.error(f"Column '{v_norm_col}' not found in DataFrame.")
        return None
    
    # Collect results in a dictionary
    results = {}
    
    # Analyze correlations with various measures
    behavioral_measures = [
        'framing_effect_avg_obs', 'delta_framing_obs',
        'prop_gamble_Gain_NTC_obs', 'prop_gamble_Gain_TC_obs',
        'prop_gamble_Loss_NTC_obs', 'prop_gamble_Loss_TC_obs'
    ]
    
    for measure in behavioral_measures:
        if measure in df.columns:
            r, p, n = calculate_and_print_correlation(df, v_norm_col, measure)
            if r is not None:
                results[measure] = {'r': r, 'p': p, 'n': n}
        else:
            logging.warning(f"Behavioral measure '{measure}' not found in DataFrame.")
    
    # Create plots for significant correlations
    for measure, stats in results.items():
        if stats['p'] < 0.05:  # Significant correlation
            plot_correlation(df, v_norm_col, measure, stats, output_dir)
    
    return results

def analyze_alpha_gain_correlations(df):
    """Analyze correlations between alpha_gain and behavioral measures."""
    logging.info("\n--- alpha_gain Correlations ---")
    alpha_gain_col = 'alpha_gain_mean' # Correct column name in the dataset
    
    if alpha_gain_col not in df.columns:
        logging.error(f"Column '{alpha_gain_col}' not found in DataFrame.")
        return None
    
    # Collect results in a dictionary
    results = {}
    
    # Analyze correlations with various measures
    behavioral_measures = [
        'gain_rt_speedup_obs', 'rt_GvL_TC_obs', 'rt_GvL_NTC_obs',
        'mean_rt_Gain_TC_obs', 'mean_rt_Gain_NTC_obs',
        'mean_rt_Loss_TC_obs', 'mean_rt_Loss_NTC_obs'
    ]
    
    for measure in behavioral_measures:
        if measure in df.columns:
            r, p, n = calculate_and_print_correlation(df, alpha_gain_col, measure)
            if r is not None:
                results[measure] = {'r': r, 'p': p, 'n': n}
        else:
            logging.warning(f"Behavioral measure '{measure}' not found in DataFrame.")
    
    # Create plots for significant correlations
    for measure, stats in results.items():
        if stats['p'] < 0.05:  # Significant correlation
            plot_correlation(df, alpha_gain_col, measure, stats, output_dir)
    
    return results

def plot_parameter_distributions(df, output_dir):
    """Plot distributions of fitted parameters."""
    logging.info("\n--- Plotting Parameter Distributions ---")
    
    # Plot v_norm distribution
    if 'v_norm_mean' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['v_norm_mean'].dropna(), kde=True, bins=15)
        plt.title("Distribution of v_norm Across Subjects")
        plt.xlabel("v_norm (mean of posterior)")
        plt.ylabel("Count")
        plt.axvline(df['v_norm_mean'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {df['v_norm_mean'].mean():.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/v_norm_distribution.png", dpi=300)
        plt.close()
        logging.info(f"Saved v_norm distribution plot to {output_dir}/v_norm_distribution.png")
    
    # Plot alpha_gain distribution
    if 'alpha_gain_mean' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['alpha_gain_mean'].dropna(), kde=True, bins=15)
        plt.title("Distribution of alpha_gain Across Subjects")
        plt.xlabel("alpha_gain (mean of posterior)")
        plt.ylabel("Count")
        plt.axvline(df['alpha_gain_mean'].mean(), color='red', linestyle='--', 
                   label=f"Mean: {df['alpha_gain_mean'].mean():.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/alpha_gain_distribution.png", dpi=300)
        plt.close()
        logging.info(f"Saved alpha_gain distribution plot to {output_dir}/alpha_gain_distribution.png")
    
    # Plot all parameters in a single figure
    param_cols = [col for col in df.columns if col.endswith('_mean') and 
                   any(param in col for param in ['v_norm', 'a_0', 'w_s_eff', 't_0', 'alpha_gain'])]
    
    if len(param_cols) > 0:
        plt.figure(figsize=(15, 10))
        for i, param in enumerate(param_cols):
            plt.subplot(3, 2, i+1) if len(param_cols) <= 6 else plt.subplot(3, 3, i+1)
            sns.histplot(df[param].dropna(), kde=True, bins=15)
            plt.title(f"Distribution of {param.replace('_mean', '')}")
            plt.axvline(df[param].mean(), color='red', linestyle='--', 
                       label=f"Mean: {df[param].mean():.3f}")
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_parameters_distribution.png", dpi=300)
        plt.close()
        logging.info(f"Saved all parameters distribution plot to {output_dir}/all_parameters_distribution.png")

def plot_correlation(df, param_col, metric_col, stats, output_dir):
    """Create a scatter plot for a correlation between parameter and behavioral measure."""
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot with regression line
    sns.regplot(data=df, x=param_col, y=metric_col, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
    
    # Add correlation information to title
    plt.title(f"Correlation: {param_col} vs {metric_col}\nr = {stats['r']:.3f}, p = {stats['p']:.4f}, N = {stats['n']}")
    
    # Clean up axes labels
    plt.xlabel(param_col.replace('mean_', ''))
    plt.ylabel(metric_col.replace('_obs', ''))
    
    # Add grid for readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    param_name = param_col.replace('mean_', '')
    metric_name = metric_col.replace('_obs', '')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/corr_{param_name}_vs_{metric_name}.png", dpi=300)
    plt.close()
    logging.info(f"Saved correlation plot to {output_dir}/corr_{param_name}_vs_{metric_name}.png")

def main():
    """Main function for the correlation analysis."""
    logging.info("Starting correlation analysis of 5-parameter model results")
    
    # Load and prepare data
    fitted_params_df, original_data_df = load_and_prepare_data()
    
    # Print the actual column names from the fitted parameters file
    logging.info("Column names in fitted parameters file:")
    for col in fitted_params_df.columns:
        logging.info(f"  - {col}")
    
    # Calculate derived measures
    fitted_params_df = calculate_derived_measures(fitted_params_df)
    
    # Plot parameter distributions
    plot_parameter_distributions(fitted_params_df, output_dir)
    
    # Analyze correlations
    v_norm_results = analyze_v_norm_correlations(fitted_params_df)
    alpha_gain_results = analyze_alpha_gain_correlations(fitted_params_df)
    
    # Save correlation results to CSV
    if v_norm_results:
        v_norm_df = pd.DataFrame(v_norm_results).T
        v_norm_df.index.name = 'behavioral_measure'
        v_norm_df.to_csv(f"{output_dir}/v_norm_correlations.csv")
        logging.info(f"Saved v_norm correlation results to {output_dir}/v_norm_correlations.csv")
    
    if alpha_gain_results:
        alpha_gain_df = pd.DataFrame(alpha_gain_results).T
        alpha_gain_df.index.name = 'behavioral_measure'
        alpha_gain_df.to_csv(f"{output_dir}/alpha_gain_correlations.csv")
        logging.info(f"Saved alpha_gain correlation results to {output_dir}/alpha_gain_correlations.csv")
    
    logging.info("Correlation analysis complete")

if __name__ == "__main__":
    main()
