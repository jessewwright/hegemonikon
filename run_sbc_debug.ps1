# PowerShell script to run SBC debug inside Docker, teeing output to a log file

# Name of your Docker container
$DOCKER_CONTAINER_NAME = "hegemonikon"
$SCRIPT_IN_CONTAINER = "/home/jovyan/work/run_sbc_for_wn_with_hddm.py"

# Set up output directory and log file
$outputDir = "sbc_debug_results"
if (-not (Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$outputDir\sbc_debug_${timestamp}.log"

Write-Host "Running SBC debug in Docker container. Output will be saved to: $logFile"

# Start the container if needed
docker start $DOCKER_CONTAINER_NAME | Out-Null

# Run the script in Docker and tee output to log
# Note: Remove '-it' if you want to run non-interactively (for logging only)
docker exec -it $DOCKER_CONTAINER_NAME python $SCRIPT_IN_CONTAINER --sbc_iterations 1 2>&1 | Tee-Object -FilePath $logFile

Write-Host "`nSBC debug run complete. Check $logFile for details."
