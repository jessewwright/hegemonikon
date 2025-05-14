import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chisquare, kstest, uniform
from sklearn.metrics import mean_absolute_error, r2_score

# CONFIG
CSV_PATH = "sbc_results_npe_20250514_001830.csv"  # Update path if needed
RANK_COLS = ["rank_w_n_eff", "rank_a", "rank_t", "rank_w_s"]
TRUE_COLS = ["true_w_n_eff", "true_a", "true_t", "true_w_s"]
MEAN_COLS = ["posterior_mean_w_n_eff", "posterior_mean_a", "posterior_mean_t", "posterior_mean_w_s"]

df = pd.read_csv(CSV_PATH)

# === QUANTITATIVE UNIFORMITY TESTS ===
print("\n[Chi-Squared + KS Tests on SBC Ranks (0–1000)]")
for col in RANK_COLS:
    name = col.replace("rank_", "")
    ranks = df[col].dropna().astype(int)
    hist, _ = np.histogram(ranks, bins=10, range=(0, 1000))
    chi_p = chisquare(hist).pvalue
    ks_p = kstest(ranks / 1000, 'uniform').pvalue
    print(f"{name:<6} | Chi² p = {chi_p:.3f} | KS p = {ks_p:.3f}")

# === ECDF PLOTS ===
plt.figure(figsize=(10, 8))
for i, col in enumerate(RANK_COLS):
    plt.subplot(2, 2, i+1)
    data = df[col].dropna().sort_values() / 1000
    plt.plot(data, np.linspace(0, 1, len(data), endpoint=False), label="ECDF")
    plt.plot([0,1], [0,1], 'r--', label="Ideal")
    plt.title(col.replace("rank_", "ECDF: "))
    plt.xlabel("Normalized Rank")
    plt.ylabel("Empirical CDF")
    plt.legend()
plt.tight_layout()
plt.savefig("ecdf_comparison_all_params.png")
print("\nSaved ECDF plot: ecdf_comparison_all_params.png")

# === RECOVERY METRICS ===
print("\n[Parameter Recovery Metrics]")
for t_col, p_col, name in zip(TRUE_COLS, MEAN_COLS, [c.replace("true_", "") for c in TRUE_COLS]):
    true = df[t_col].dropna()
    pred = df[p_col].dropna()
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    bias = np.mean(pred - true)
    print(f"{name:<6} | R² = {r2:.3f} | MAE = {mae:.3f} | Bias = {bias:+.3f}")
