import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os # Import os module

# --- Configuration ---
DATA_DIR = os.path.join('..', 'data')  # Go up one directory to find the data folder
RESULTS_FILE = 'mvnes_gng_results.csv'
PARAMS_FILE = 'mvnes_gng_params.json'
RESULTS_PATH = os.path.join(DATA_DIR, RESULTS_FILE)
PARAMS_PATH = os.path.join(DATA_DIR, PARAMS_FILE)

# --- 1. Load Data ---
print(f"Attempting to load data from: {RESULTS_PATH}")
try:
    # Check if the directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please ensure the script is run from the correct parent directory")
        print("or adjust the DATA_DIR variable in the script.")
        exit()
        
    # Check if the files exist
    if not os.path.isfile(RESULTS_PATH):
        print(f"Error: Results file '{RESULTS_FILE}' not found in '{DATA_DIR}'.")
        print("Please make sure you have run the simulator (`src/simulator.py`)")
        print("and the output CSV file exists.")
        exit()
    
    if not os.path.isfile(PARAMS_PATH):
        print(f"Error: Parameters file '{PARAMS_FILE}' not found in '{DATA_DIR}'.")
        print("Please make sure you have run the simulator (`src/simulator.py`)")
        print("and the parameters JSON file exists.")
        exit()
        
    # Load trial results
    df = pd.read_csv(RESULTS_PATH)
    print(f"\nSuccessfully loaded trial results from {RESULTS_PATH}")
    print("First 5 rows:\n", df.head())
    print("\nValue Counts for 'outcome':\n", df['outcome'].value_counts())
    print("\nValue Counts for 'stimulus':\n", df['stimulus'].value_counts())
    
    # Load parameters
    import json
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"\nSuccessfully loaded parameters from {PARAMS_PATH}")
    print("Parameters:", params)
    print("-" * 30)

except Exception as e:
    print(f"\nAn error occurred loading or processing the CSV: {e}")
    exit()

# --- 2. Calculate Metrics ---
n_trials = len(df)
if n_trials == 0:
    print("Error: The CSV file is empty.")
    exit()

n_go_trials = (df['stimulus'] == 'go').sum()
n_nogo_trials = (df['stimulus'] == 'nogo').sum()

print("\n--- Performance Metrics ---")
print(f"Total Trials Analyzed: {n_trials}")
print(f"Go Trials: {n_go_trials}")
print(f"NoGo Trials: {n_nogo_trials}")

if n_go_trials == 0 and n_nogo_trials == 0:
    print("Warning: No Go or NoGo trials found in the data.")
    # Exit or handle as appropriate if this state is unexpected
    exit()

# Calculate counts for each outcome
hits = (df['outcome'] == 'Hit').sum()
misses = (df['outcome'] == 'Miss').sum()
correct_rejections = (df['outcome'] == 'Correct Rejection').sum()
false_alarms = (df['outcome'] == 'False Alarm').sum()

# Verify counts add up
if (hits + misses) != n_go_trials:
    print(f"Warning: Hit ({hits}) + Miss ({misses}) count ({hits+misses}) does not match total Go trials ({n_go_trials}). Check data.")
if (correct_rejections + false_alarms) != n_nogo_trials:
    print(f"Warning: CR ({correct_rejections}) + FA ({false_alarms}) count ({correct_rejections+false_alarms}) does not match total NoGo trials ({n_nogo_trials}). Check data.")


# Calculate percentages (handle division by zero)
percent_hit = np.divide(hits * 100, n_go_trials) if n_go_trials > 0 else 0
percent_miss = np.divide(misses * 100, n_go_trials) if n_go_trials > 0 else 0
percent_cr = np.divide(correct_rejections * 100, n_nogo_trials) if n_nogo_trials > 0 else 0
percent_fa = np.divide(false_alarms * 100, n_nogo_trials) if n_nogo_trials > 0 else 0

# Calculate mean RTs (only for actual responses, not timeouts unless FA/Hit)
mean_rt_hit = df.loc[df['outcome'] == 'Hit', 'rt'].mean()
std_rt_hit = df.loc[df['outcome'] == 'Hit', 'rt'].std()
mean_rt_fa = df.loc[df['outcome'] == 'False Alarm', 'rt'].mean()
std_rt_fa = df.loc[df['outcome'] == 'False Alarm', 'rt'].std()

# Print Metrics
print("-" * 25)
print("Outcome Counts & Percentages:")
print(f"  Hits: {hits} ({percent_hit:.2f}% of Go trials)")
print(f"  Misses: {misses} ({percent_miss:.2f}% of Go trials)")
print(f"  Correct Rejections: {correct_rejections} ({percent_cr:.2f}% of NoGo trials)")
print(f"  False Alarms: {false_alarms} ({percent_fa:.2f}% of NoGo trials)")
print("-" * 25)
print("Mean Reaction Times (SD):")
# Handle cases where there are no Hits or FAs for RT calculation
print(f"  Mean RT (Hits): {mean_rt_hit:.4f} (SD: {std_rt_hit:.4f})" if hits > 0 else "  Mean RT (Hits): N/A (No Hits)")
print(f"  Mean RT (False Alarms): {mean_rt_fa:.4f} (SD: {std_rt_fa:.4f})" if false_alarms > 0 else "  Mean RT (False Alarms): N/A (No False Alarms)")
print("-" * 25)


# --- 3. Create Visualizations ---

# Prepare data for plotting proportions more easily
plot_df = pd.DataFrame({
    'Outcome': ['Hit', 'Miss', 'Correct Rejection', 'False Alarm'],
    'Count': [hits, misses, correct_rejections, false_alarms],
    'Proportion': [percent_hit, percent_miss, percent_cr, percent_fa], # Use percentages directly
    'Trial Type': ['Go', 'Go', 'NoGo', 'NoGo']
})

plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Use fig, axes for better control

# --- Subplot 1: Bar chart of outcome proportions ---
# Grouped bar chart showing % within each trial type
try:
    if not plot_df.empty:
        sns.barplot(ax=axes[0], x='Trial Type', y='Proportion', hue='Outcome', data=plot_df, palette='viridis')
        axes[0].set_title('Outcome Proportions by Trial Type')
        axes[0].set_ylabel('Percentage (%)')
        axes[0].set_xlabel('Stimulus Type')
        axes[0].set_ylim(0, 105) # Ensure y-axis goes slightly above 100%
        axes[0].legend(title='Outcome')
        # Add text labels on bars
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f%%', label_type='edge', fontsize=9, padding=2)
    else:
         axes[0].text(0.5, 0.5, 'No data to plot proportions', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
         axes[0].set_title('Outcome Proportions')

except Exception as e:
    print(f"An error occurred during plotting proportions: {e}")
    axes[0].text(0.5, 0.5, 'Error plotting proportions', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    axes[0].set_title('Outcome Proportions')


# --- Subplot 2: RT Histograms/Density Plots for Hits vs. False Alarms ---
try:
    plot_rt = False # Flag to check if we add labels/legend
    # Filter data for plotting RTs
    df_hits = df[df['outcome'] == 'Hit']
    df_fas = df[df['outcome'] == 'False Alarm']

    if not df_hits.empty:
        sns.histplot(ax=axes[1], data=df_hits, x='rt', kde=True, color='forestgreen', label=f'Hits (N={hits})', stat='density', bins=15, alpha=0.6)
        plot_rt = True
    if not df_fas.empty:
        sns.histplot(ax=axes[1], data=df_fas, x='rt', kde=True, color='crimson', label=f'False Alarms (N={false_alarms})', stat='density', bins=15, alpha=0.6)
        plot_rt = True

    if plot_rt:
        axes[1].set_title('RT Distribution for Hits and False Alarms')
        axes[1].set_xlabel('Reaction Time (s)')
        axes[1].set_ylabel('Density')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No Hits or False Alarms to plot RTs', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
        axes[1].set_title('RT Distribution')

except Exception as e:
    print(f"An error occurred during plotting RTs: {e}")
    axes[1].text(0.5, 0.5, 'Error plotting RTs', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    axes[1].set_title('RT Distribution')


# --- Display Plots ---
plt.tight_layout(pad=2.0) # Add padding between plots
plt.suptitle('MVNES Go/No-Go Simulation Analysis', fontsize=16, y=1.02) # Add overall title
plt.show()


# --- 4. Interpretation Guidance (Prompts for discussion) ---
print("\n--- Interpretation Prompts ---")
print("Review the metrics and plots. Let's discuss:")
print(f"1. Accuracy: How well did the MVNES perform? (Go Acc: {percent_hit:.2f}%, NoGo Acc: {percent_cr:.2f}%)")
print(f"2. Inhibition: Was the agent effective at withholding responses on NoGo trials (%CR: {percent_cr:.2f}%)?")
print(f"3. Errors: Were there many Misses ({percent_miss:.2f}%) or False Alarms ({percent_fa:.2f}%)?")
print(f"4. RT Comparison:")
if hits > 0 and false_alarms > 0:
    rt_comparison = "FASTER" if mean_rt_fa < mean_rt_hit else "SLOWER" if mean_rt_fa > mean_rt_hit else "SIMILAR"
    print(f"   - Were False Alarms ({mean_rt_fa:.4f}s) {rt_comparison} than Hits ({mean_rt_hit:.4f}s)?")
    print(f"   - How do the shapes of the RT distributions compare in the plot?")
elif hits > 0:
    print(f"   - Mean RT for Hits was {mean_rt_hit:.4f}s. (No FAs for comparison).")
elif false_alarms > 0:
    print(f"   - Mean RT for FAs was {mean_rt_fa:.4f}s. (No Hits for comparison).")
else:
     print("   - No Hits or False Alarms occurred, so RT comparison is not possible.")

print(f"\n5. Plausibility & Parameters:")
# Display parameters from the first row (assuming they are constant)
try:
    sim_params = df.iloc[0][['w_s', 'w_n', 'threshold_a', 't', 'noise_std_dev', 'dt', 'max_time']]
    print(f"   - Consider the parameters used:")
    print(sim_params.to_string())
    # TODO: We need to know the S and N input values for Go/NoGo trials to fully interpret drift rate.
    # Let's assume standard GNG: Go (S=1, N=0), NoGo (S=0, N=1) for now, unless simulator.py used different inputs.
    # If so, the drift for Go would be w_s*1 - w_n*0 = w_s
    # And the drift for NoGo would be w_s*0 - w_n*1 = -w_n
    print(f"   - Assumed Go trial drift: w_s = {sim_params['w_s']:.2f}")
    print(f"   - Assumed NoGo trial drift: -w_n = {-sim_params['w_n']:.2f}")
    print(f"   - Do the observed accuracy and RTs seem reasonable given these drift rates, threshold ({sim_params['threshold_a']:.2f}), noise ({sim_params['noise_std_dev']:.2f}), and non-decision time ({sim_params['t']:.2f})?")
    print(f"   - For example, does a negative drift on NoGo trials explain the high CR rate (if observed)? Does a positive drift on Go trials explain the Hit rate?")
except KeyError as e:
    print(f"   - Could not extract all expected parameter columns: {e}")
except IndexError:
    print(f"   - Could not extract parameters (empty dataframe?).")


print("\nWhat are your initial thoughts on whether the MVNES is behaving plausibly based on this first run?")