# Create output directory if it doesn't exist
$outputDir = "sbc_wn_hddm_results"
if (-not (Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Get timestamp for log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$outputDir\sbc_wn_hddm_${timestamp}.log"

# Get container ID
$containerId = $(docker ps -q --filter name=hegemonikon)
if (-not $containerId) {
    Write-Host "Error: Hegemonikon container is not running. Please start it with 'docker-compose up -d'"
    exit 1
}

# Build the command with parameters
$pythonCmd = "python /home/jovyan/work/run_sbc_for_wn_with_hddm.py"

# Add any additional parameters here if needed
# Example: $pythonCmd += " --sbc_iterations 100 --n_subj 30"

Write-Host "Starting SBC analysis for w_n with HDDM..."
Write-Host "This may take a while. Progress will be saved to: $logFile"
Write-Host "Running command: $pythonCmd"

# Run the Python script in the container and capture output
docker exec -it $containerId $pythonCmd 2>&1 | Tee-Object -FilePath $logFile

# Check for results
Write-Host "`nAnalysis completed. Checking for results..."

# Look for any result files in the output directory
$resultFiles = Get-ChildItem -Path $outputDir -File | Where-Object { $_.Name -match "sbc_wn_hddm_.*(\.csv|\.png|\.pkl)$" }

if ($resultFiles.Count -gt 0) {
    Write-Host "`nFound result files:"
    $resultFiles | ForEach-Object { Write-Host "- $($_.Name)" }
    
    # Try to display the summary if available
    $summaryFile = $resultFiles | Where-Object { $_.Name -match "summary.*\.csv" } | Select-Object -First 1
    if ($summaryFile) {
        Write-Host "`nSBC Analysis Summary:"
        Get-Content $summaryFile.FullName | Select-Object -First 10 | ForEach-Object { Write-Host $_ }
    }
    
    Write-Host "`nCheck the full results in the $outputDir directory."
} else {
    Write-Host "No result files found. Check $logFile for details."
}

Write-Host "`nDone."
