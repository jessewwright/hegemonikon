# PowerShell script to run multiple SBC instances in parallel

# Number of parallel runs
$parallelRuns = 4
# Number of SBC iterations per run
$simsPerRun = 25

# Create a new window for each run
$processes = @()
for ($i = 1; $i -le $parallelRuns; $i++) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\jesse\Hegemonikon Project\hegemonikon'; python validate_wn_recovery_stroop_fixed.py $i $simsPerRun"
}

# Wait for all processes to complete by checking for result files
Write-Host "All processes have been started. Waiting for completion..."
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$expectedFiles = $parallelRuns
$foundFiles = 0

# Wait for all result files to appear
while ($foundFiles -lt $expectedFiles) {
    Start-Sleep -Seconds 2
    $foundFiles = (Get-ChildItem -Path "wn_sbc_results" -Filter "stroop_sbc_results_seed*_$timestamp.csv" | Measure-Object).Count
    Write-Host "Found $foundFiles of $expectedFiles files..."
}

# After completion, analyze results
Write-Host "All processes have completed. Starting analysis..."
python workflows/analyze_results.py
