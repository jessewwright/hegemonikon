# Test script to verify error rate profiles for different w_n values
import numpy as np
import pandas as pd

# Fixed parameters for the DDM
W_S = 0.7
THRESHOLD = 0.6
NOISE_STD = 0.40
DT = 0.02
T0 = 0.2
MAX_TIME = 4.0

# Conflict levels and proportions
CONFLICT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONFLICT_PROPORTIONS = [0.2, 0.2, 0.2, 0.2, 0.2]

def simulate_ddm_trial(drift, threshold, noise_std, dt, t0, max_time):
    """Simulates a single trial of a two-choice DDM using Euler-Maruyama."""
    t = t0
    x = 0.0
    
    while t < max_time:
        # Decision process
        dx = drift * dt + noise_std * np.sqrt(dt) * np.random.randn()
        x += dx
        t += dt
        
        # Check boundaries
        if x >= threshold:
            return 1, t  # Correct choice
        elif x <= -threshold:
            return 0, t  # Error choice
    
    return np.nan, np.nan  # No decision within max_time

def generate_stroop_like_dataset(w_n_true, n_trials, conflict_levels, conflict_proportions):
    """Generates a dataset for the Stroop-like task for a given true w_n."""
    # Generate trial types according to proportions
    n_levels = len(conflict_levels)
    trial_types = np.random.choice(
        np.arange(n_levels),
        size=n_trials,
        p=conflict_proportions
    )
    
    # Initialize lists to store results
    choices = []
    rts = []
    conflict_levels_used = []
    
    for trial_type in trial_types:
        # Calculate drift based on conflict level
        conflict_level = conflict_levels[trial_type]
        drift = W_S * (1.0 - conflict_level) - w_n_true * conflict_level
        
        # Simulate trial
        choice, rt = simulate_ddm_trial(
            drift=drift,
            threshold=THRESHOLD,
            noise_std=NOISE_STD,
            dt=DT,
            t0=T0,
            max_time=MAX_TIME
        )
        
        if not np.isnan(choice):  # Only include valid trials
            choices.append(choice)
            rts.append(rt)
            conflict_levels_used.append(conflict_level)
    
    # Create DataFrame
    df = pd.DataFrame({
        'choice': choices,
        'rt': rts,
        'conflict_level': conflict_levels_used
    })
    
    return df

# Test different w_n values
w_n_values = [0.2, 0.9, 1.7]

print("\nTesting error rate profiles for different w_n values:")

for w_n in w_n_values:
    print(f"\n=== Testing w_n = {w_n:.1f} ===")
    
    # Generate dataset
    df = generate_stroop_like_dataset(
        w_n_true=w_n,
        n_trials=300,
        conflict_levels=CONFLICT_LEVELS,
        conflict_proportions=CONFLICT_PROPORTIONS
    )
    
    # Print error rates by conflict level
    error_rates = df.groupby('conflict_level')['choice'].mean()
    print("\nError rates by conflict level:")
    print(error_rates)
    
    # Print RT means by conflict level
    rt_means = df.groupby('conflict_level')['rt'].mean()
    print("\nRT means by conflict level:")
    print(rt_means)
    
    # Print summary statistics
    print(f"\nTotal trials: {len(df)}")
    print(f"Correct trials: {df['choice'].sum()}")
    print(f"Error trials: {(df['choice'] == 0).sum()}")
    print("-" * 50)

print("\nTest complete. Check for different error rate profiles across w_n values.")
