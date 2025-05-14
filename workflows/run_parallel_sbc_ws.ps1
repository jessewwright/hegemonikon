# PowerShell script to run multiple SBC instances in parallel for w_s

# Number of parallel runs (reduced to avoid overloading)
$parallelRuns = 2
# Number of SBC iterations per run (reduced for quicker feedback)
$simsPerRun = 10
$baseSeed = 9000

# Create new directory for results
$resultsDir = "../outputs/sbc_results/ws_sbc_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $resultsDir -Force

# Create a new window for each run
for ($i = 0; $i -lt $parallelRuns; $i++) {
    # Assign a base seed that increments for each parallel run
    $seed = $baseSeed + $i
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '../hegemonikon'; python validate_ws_recovery_stroop.py $seed $simsPerRun"
}

Write-Host "Launched $parallelRuns parallel instances..."
Write-Host "Results will be saved in $resultsDir directory"
Write-Host "Please wait for all windows to complete before analyzing results"