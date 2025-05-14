# Create output directory if it doesn't exist
$outputDir = "hddm_recovery_results"
if (-not (Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Run the test script and capture output
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$outputDir\hddm_recovery_${timestamp}.log"

Write-Host "Running HDDM recovery test. This may take 20-30 minutes..."
Write-Host "Progress will be saved to: $logFile"

# Run the Python script and capture output
docker exec -it $(docker ps -q --filter name=hegemonikon) python /home/jovyan/work/hddm_recovery_test.py 2>&1 | Tee-Object -FilePath $logFile

# Check for results
Write-Host "`nTest completed. Checking for results..."

# Look for any result files
$resultFiles = Get-ChildItem -Path $outputDir -Filter "recovery_*" -File

if ($resultFiles.Count -gt 0) {
    Write-Host "`nFound result files:"
    $resultFiles | ForEach-Object { Write-Host "- $($_.Name)" }
    
    # Try to display the summary if available
    $summaryFile = $resultFiles | Where-Object { $_.Name -like "*summary*.csv" } | Select-Object -First 1
    if ($summaryFile) {
        Write-Host "`nRecovery Summary:"
        Get-Content $summaryFile.FullName | Select-Object -First 10 | ForEach-Object { Write-Host $_ }
    }
    
    Write-Host "`nCheck the full results in the $outputDir directory."
} else {
    Write-Host "No result files found. Check $logFile for details."
}
