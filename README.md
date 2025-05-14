# Hegemonikon

[![CI](https://github.com/jessewwright/hegemonikon/actions/workflows/ci.yml/badge.svg)](https://github.com/jessewwright/hegemonikon/actions)

## Getting Started

```bash
git clone https://github.com/jessewwright/hegemonikon.git
cd hegemonikon

# Create a venv & install
python -m venv .venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Run a quick Stroop sim
nes-run stroop --params params/stroop_default.json --n-trials 100
```

## Examples

- **Stroop demo**: [`notebooks/Stroop_demo.ipynb`](notebooks/Stroop_demo.ipynb)

## HDDM Analysis

This project includes tools for running Simulation-Based Calibration (SBC) using HDDM (Hierarchical Drift Diffusion Model) to validate parameter recovery of the NES model's w_n parameter.

### Prerequisites

- Docker and Docker Compose installed on your system
- Sufficient system resources (HDDM can be memory-intensive)

### Running the Analysis

1. **Build and run the HDDM Docker container**:

   ```bash
   # From the project root directory
   docker-compose -f docker/compose.yml build hddm
   ```

2. **Run the SBC analysis**:

   ```bash
   # Basic run with default parameters (20 iterations, 200 trials each)
   docker-compose -f docker/compose.yml run --rm hddm python run_sbc_for_wn_with_hddm.py
   
   # Customize the number of iterations and trials
   docker-compose -f docker/compose.yml run --rm hddm python run_sbc_for_wn_with_hddm.py --sbc_iterations 20 --n_trials 200
   ```

### Output

The analysis will generate several output files in the `sbc_hddm_results` directory:

- `sbc_results_wn_hddm_*.csv`: CSV file containing the SBC results
- `recovery_diagnostics_*.png`: Diagnostic plots showing true vs. recovered w_n values
- `sbc_hist_w_n_*.png`: SBC rank histogram for w_n

### Analysis Parameters

Key parameters that can be adjusted in the script:

- `N_SBC_ITERATIONS`: Number of SBC iterations to run (default: 20)
- `N_TRIALS_PER_SUB`: Number of trials per subject (default: 200)
- `N_SUBJECTS`: Number of subjects per iteration (default: 20)
- `HDDM_SAMPLES`: Number of MCMC samples for HDDM (default: 2000)
- `HDDM_BURN`: Number of burn-in samples (default: 1000)

### Interpreting Results

- **Recovery Plot**: Should show points clustered around the y=x line for good recovery
- **SBC Rank Histogram**: Should be approximately uniform for well-calibrated inference
- **Recovery Metrics**:
  - Slope close to 1.0 indicates good linear recovery
  - Intercept close to 0.0 indicates minimal bias
  - Low standard deviation of recovery errors indicates precise estimation

### Troubleshooting

- If you encounter memory issues, try reducing the number of trials or subjects
- For faster debugging, reduce the number of iterations or trials
- Check the console output for any error messages or warnings
