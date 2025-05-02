# --- Generate Synthetic Data for Parameter Recovery ---
import numpy as np
import pandas as pd
import random

# --- Generate True Parameters for Subjects ---
def generate_true_parameters(n_subjects):
    """
    Generate plausible true parameters for subjects.
    """
    true_params = []
    
    # Generate parameters with some variability
    for subj_id in range(n_subjects):
        # k values between 0.01 and 0.1 (typical discounting range)
        k = np.random.uniform(0.01, 0.1)
        
        # Threshold values between 0.4 and 1.0 (typical decision threshold range)
        threshold = np.random.uniform(0.4, 1.0)
        
        true_params.append({
            'subject': subj_id,
            'true_k': k,
            'true_threshold': threshold
        })
    
    return pd.DataFrame(true_params)

# --- Generate Synthetic Choices ---
def generate_synthetic_choices(true_params_df, n_trials_per_subject=50):
    """
    Generate synthetic choices based on true parameters.
    """
    all_choices = []
    
    # Fixed options for all subjects
    ss_option = {'amount': 5, 'delay': 0}
    ll_amount = 10
    delays = [1, 5, 10, 20, 50]  # Same delays as in the test script
    
    for _, row in true_params_df.iterrows():
        subj_id = row['subject']
        k = row['true_k']
        threshold = row['true_threshold']
        
        # Generate trials for this subject
        for trial in range(n_trials_per_subject):
            # Randomly select a delay
            delay = random.choice(delays)
            
            # Calculate discounted value
            v_ll = ll_amount / (1 + k * delay)
            v_ss = ss_option['amount']
            
            # Simulate choice based on values and threshold
            # This is a simplified version - in real simulation we would use NES components
            evidence_diff = v_ll - v_ss
            if evidence_diff > threshold:
                choice = 'Choose_LL'
            else:
                choice = 'Choose_SS'
            
            # Add some noise to make it more realistic
            if random.random() < 0.1:  # 10% chance of random choice
                choice = random.choice(['Choose_LL', 'Choose_SS'])
            
            all_choices.append({
                'subject': subj_id,
                'trial': trial + 1,
                'll_delay': delay,
                'choice': choice,
                'v_ll': v_ll,
                'v_ss': v_ss
            })
    
    return pd.DataFrame(all_choices)

# --- Main ---
if __name__ == "__main__":
    print("Generating synthetic data for parameter recovery...")
    
    # Generate data for 50 subjects
    n_subjects = 50
    true_params_df = generate_true_parameters(n_subjects)
    synthetic_data_df = generate_synthetic_choices(true_params_df)
    
    # Save the data
    true_params_df.to_csv('true_parameters.csv', index=False)
    synthetic_data_df.to_csv('synthetic_data.csv', index=False)
    
    print(f"\nGenerated data for {n_subjects} subjects:")
    print("True parameters saved to 'true_parameters.csv'")
    print("Synthetic choices saved to 'synthetic_data.csv'")
