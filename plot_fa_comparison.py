import pandas as pd
import matplotlib.pyplot as plt
import os

# Load summary files
def load_summary_data():
    # Load both summary files
    df_no_stress = pd.read_csv(os.path.join("..", "data", "wn_sweep_summary.csv"))
    df_stress = pd.read_csv(os.path.join("..", "data", "wn_sweep_summary_stress_0.8.csv"))
    
    # Extract w_n and false_alarm_rate
    return df_no_stress[['w_n', 'false_alarm_rate']], df_stress[['w_n', 'false_alarm_rate']] 

# --- Load summaries ---
df_no_stress, df_stress = load_summary_data()

# --- Plot comparison ---
plt.figure(figsize=(8, 5))
plt.plot(df_no_stress["w_n"], df_no_stress["false_alarm_rate"], label="No Stress", marker='o')
plt.plot(df_stress["w_n"], df_stress["false_alarm_rate"], label="Stress Modulation", marker='s')
plt.xlabel("Norm Weight (wₙ)")
plt.ylabel("False Alarm Rate")
plt.title("NES False Alarm Rate vs. wₙ (Stress vs. No Stress)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
