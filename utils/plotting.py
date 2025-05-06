import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_evidence_trace(results_df, trial_idx, save_path=None):
    """
    Plot the evidence trace for a specific trial.
    
    Args:
        results_df: DataFrame containing simulation results
        trial_idx: Index of the trial to plot
        save_path: Optional path to save the plot
    """
    trial_data = results_df.iloc[trial_idx]
    
    # Parse the trace string into a list of floats
    trace = eval(trial_data['trace'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(trace, label='Activation Trace')
    plt.axhline(y=trial_data['threshold_a'], color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (dt units)')
    plt.ylabel('Activation')
    plt.title(f"Trial {trial_idx + 1}: {trial_data['stimulus']} - {trial_data['outcome']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_trial_comparison(results_df, trial_type='go', n_trials=5, save_path=None):
    """
    Plot evidence traces for multiple trials of the same type.
    
    Args:
        results_df: DataFrame containing simulation results
        trial_type: 'go' or 'nogo'
        n_trials: Number of trials to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get n_trials of the specified type
    trials = results_df[results_df['stimulus'] == trial_type].head(n_trials)
    
    for idx, trial in trials.iterrows():
        trace = eval(trial['trace'])
        plt.plot(trace, alpha=0.7, label=f"Trial {idx + 1} ({trial['outcome']})")
    
    plt.axhline(y=trials.iloc[0]['threshold_a'], color='r', linestyle='--', label='Threshold')
    plt.xlabel('Time (dt units)')
    plt.ylabel('Activation')
    plt.title(f"{n_trials} {trial_type} Trials Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_performance_metrics(results_df, save_path=None):
    """
    Plot performance metrics (hit rate, false alarm rate) with reaction time distributions.
    
    Args:
        results_df: DataFrame containing simulation results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Split trials by type and outcome
    go_trials = results_df[results_df['stimulus'] == 'go']
    nogo_trials = results_df[results_df['stimulus'] == 'nogo']
    
    # Plot reaction time distributions
    plt.subplot(1, 2, 1)
    for outcome, color in [('Hit', 'green'), ('False Alarm', 'red')]:
        trials = results_df[results_df['outcome'] == outcome]
        if not trials.empty:
            sns.kdeplot(trials['rt'], label=f'{outcome} RT', color=color, shade=True)
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Density')
    plt.title('Reaction Time Distributions')
    plt.legend()
    
    # Plot performance metrics
    plt.subplot(1, 2, 2)
    metrics = {
        'Hit Rate': (go_trials['outcome'] == 'Hit').mean(),
        'Miss Rate': (go_trials['outcome'] == 'Miss').mean(),
        'False Alarm Rate': (nogo_trials['outcome'] == 'False Alarm').mean(),
        'Correct Rejection Rate': (nogo_trials['outcome'] == 'Correct Rejection').mean()
    }
    
    plt.bar(metrics.keys(), metrics.values(), color=['green', 'red', 'red', 'green'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
