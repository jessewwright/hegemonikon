import numpy as np
import pandas as pd
from pathlib import Path

def load_training_stat_means(means_csv_path):
    """
    Load the training summary statistic means from a CSV file.
    Assumes a single row of means, optionally with header.
    Returns a numpy array of means.
    """
    df = pd.read_csv(means_csv_path, header=None)
    # If there's a header, try to detect and skip
    if df.shape[0] == 1:
        means = df.iloc[0].values
    else:
        means = df.values.flatten()
    return means.astype(float)

if __name__ == "__main__":
    # Example usage
    means_path = Path(r"C:/Users/jesse/Hegemonikon Project/hegemonikon/sbc_nes_roberts_template250_v2/data/training_stat_means.csv")
    means = load_training_stat_means(means_path)
    print(f"Loaded means: {means}")
