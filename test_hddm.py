import hddm
import os
import pandas as pd
import numpy as np

print("HDDM imported successfully!")

# Load the example dataset
print("Loading dataset...")
data_path = os.path.join('hddm', 'hddm', 'examples', 'simple_difficulty.csv')
print(f"Looking for data at: {data_path}")

try:
    # Load data using pandas first to inspect
    data = pd.read_csv(data_path)
    print(f"Successfully loaded data. Columns: {data.columns.tolist()}")
    print(f"Unique difficulty values: {data['difficulty'].unique()}")
    print(f"Number of rows: {len(data)}")
    
    # Convert response to 0/1 if needed
    if data['response'].dtype == 'float64':
        data['response'] = data['response'].astype(int)
    
    # Create and fit model with more verbosity
    print("\nCreating model...")
    
    # Add a subject ID column (treating all data as from one subject for this test)
    data['subj_idx'] = 0
    
    # Create model with explicit parameters
    model = hddm.HDDM(data, 
                     depends_on={'v': 'difficulty'},
                     include=('v', 'a', 't'),
                     p_outlier=0.05,
                     is_group_model=True,
                     group_only_nodes=['v', 'a', 't'])
    
    print("Model created. Starting sampling (this may take a few minutes)...")
    # Run with fewer samples for testing
    model.sample(1000, burn=100, dbname='traces.db', db='txt')
    
    # Print model statistics
    print("\nModel statistics:")
    stats = model.gen_stats()
    print(stats)
    
    # Save results to a file
    results_file = 'hddm_results.csv'
    stats.to_csv(results_file)
    print(f"\nResults saved to {results_file}")
    
    # Print parameter estimates
    print("\nParameter estimates:")
    for param in ['a', 'v', 't']:
        if param in stats.index:
            print(f"\n{param}:")
            print(stats.loc[[p for p in stats.index if p.startswith(param)]])
        else:
            print(f"\nWarning: Parameter {param} not found in model results")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    import traceback
    print("\nError details:")
    print(traceback.format_exc())
    print("\nCurrent working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
