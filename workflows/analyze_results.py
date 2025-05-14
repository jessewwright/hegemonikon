import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import datetime
from src.analysis import sbc_metrics
from src.utils.visualization import plot_recovery

def analyze_sbc_results(timestamp):
    """
    Analyze SBC results from a specific run
    
    Args:
        timestamp: The timestamp string to match against result files
    """
    # Find all result CSV files
    csv_files = glob.glob('wn_sbc_results/stroop_sbc_results_seed*.csv')
    
    if not csv_files:
        print("No result files found!")
        return
    
    # Filter files by timestamp
    matching_files = [f for f in csv_files if f.endswith(f'_{timestamp}.csv')]
    
    if not matching_files:
        print(f"No result files found for timestamp {timestamp}!")
        return
    
    # Read and concatenate results
    all_results = []
    for file in matching_files:
        try:
            df = pd.read_csv(file)
            all_results.append(df)
            print(f"Loaded results from {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if not all_results:
        print("No valid result files found!")
        return
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Plot rank histogram
    plt.figure(figsize=(10, 6))
    plt.hist(combined_df['sbc_rank'], bins=range(0, combined_df['n_posterior_samples'].max() + 2), 
            edgecolor='black', alpha=0.7)
    plt.title('SBC Rank Histogram')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save histogram with timestamp
    hist_filename = f'sbc_rank_histogram_{timestamp}.png'
    hist_filename = f'combined_sbc_rank_histogram_{timestamp}.png'
    plt.savefig(f'wn_sbc_results/{hist_filename}')
    plt.close()
    
    # Print summary statistics
    print("\nCombined Results Summary:")
    print(f"Total iterations: {len(combined_df)}")
    print(f"Mean SBC rank: {combined_df['sbc_rank'].mean():.2f}")
    print(f"Mean posterior mean: {combined_df['posterior_mean'].mean():.4f}")
    
    # Save combined results with timestamp
    results_filename = f'combined_sbc_results_{timestamp}.csv'
    combined_df.to_csv(f'wn_sbc_results/{results_filename}', index=False)
    print(f"\nCombined results saved to wn_sbc_results/{results_filename}")

if __name__ == "__main__":
    # Find all result files
    csv_files = glob.glob('wn_sbc_results/stroop_sbc_results_seed*.csv')
    if not csv_files:
        print("No result files found!")
        exit(1)
    
    # Extract timestamps from all files
    timestamps = [f.split('_')[-1].split('.')[0] for f in csv_files]
    
    # Get the latest timestamp
    latest_timestamp = max(timestamps)
    
    print(f"Analyzing results for latest timestamp: {latest_timestamp}")
    analyze_sbc_results(latest_timestamp)
