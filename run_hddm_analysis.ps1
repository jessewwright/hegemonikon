# Run HDDM analysis in Docker and copy results to local machine

# Configuration
$DOCKER_CONTAINER_NAME = "hegemonikon"
$SCRIPT_IN_CONTAINER = "/home/jovyan/work/test_hddm_on_nes_stroop_data.py"
$CONTAINER_OUTPUT_DIR = "/home/jovyan/work/hddm_results"
$LOCAL_OUTPUT_DIR = "$PSScriptRoot\hddm_results"

# Create local output directory if it doesn't exist
if (-not (Test-Path -Path $LOCAL_OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $LOCAL_OUTPUT_DIR | Out-Null
}

# Run the Python script in the container
Write-Host "Running HDDM analysis in Docker container..."
docker exec -it $DOCKER_CONTAINER_NAME python $SCRIPT_IN_CONTAINER

# Get the most recent output directory in the container
$latest_dir = docker exec $DOCKER_CONTAINER_NAME bash -c "ls -td $CONTAINER_OUTPUT_DIR/*/ | head -1"
$latest_dir = $latest_dir.Trim()

if ($latest_dir) {
    # Create the same directory structure locally
    $relative_path = $latest_dir.Replace($CONTAINER_OUTPUT_DIR, "")
    $local_dir = Join-Path $LOCAL_OUTPUT_DIR $relative_path
    
    if (-not (Test-Path -Path $local_dir)) {
        New-Item -ItemType Directory -Path $local_dir | Out-Null
    }
    
    # Copy all files from container to local
    Write-Host "Copying files from container to $local_dir"
    docker cp "${DOCKER_CONTAINER_NAME}:${latest_dir}" "$local_dir\..\"
    
    Write-Host "Files successfully copied to: $local_dir"
    
    # Open the output directory in File Explorer
    explorer $local_dir
} else {
    Write-Host "No output directory found in container. Please check the script output for errors."
}
