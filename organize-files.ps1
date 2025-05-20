### 
# Windows PowerShell Organizing Script
# Below is a PowerShell script you can run from your project root. It won’t touch .py, .ps1, .ipynb, README.md, or config files, but moves other files to the
# folders described above.
# You may review/tweak the folders to your preference before running!

    # PowerShell file organizing script for your project folder

    # Set your project root
    $Root = "C:\Users\jesse\Hegemonikon Project\hegemonikon"
    Set-Location $Root

    # Create directories if they don't exist
    $folders = @("results_csv", "databases_and_bins", "logs", "reports", "misc")
    $folders | ForEach-Object { if (!(Test-Path $_)) { New-Item -ItemType Directory $_ } }

    # Move CSVs (but keep code/notebook files in root)
    Get-ChildItem -Path $Root -File -Filter *.csv |
        Where-Object { $_.Name -notlike "*test*" -and $_.Name -notlike "*dataset*" } |
        Move-Item -Destination "$Root\results_csv"

    # Move DB and EXE files
    Get-ChildItem -Path $Root -File -Include *.db, *.exe |
        Move-Item -Destination "$Root\databases_and_bins"

    # Move TXT logs (exclude README and requirements)
    Get-ChildItem -Path $Root -File -Filter *.txt |
        Where-Object { $_.Name -ne "README.md" -and $_.Name -ne "requirements.txt" } |
        Move-Item -Destination "$Root\logs"

    # Move Markdown reports (exclude README.md)
    Get-ChildItem -Path $Root -File -Filter *.md |
        Where-Object { $_.Name -ne "README.md" } |
        Move-Item -Destination "$Root\reports"

    # Move anything not handled above, and not scripts/config/readme
    Get-ChildItem -Path $Root -File | Where-Object {
        $_.Extension -notin @('.py','.ps1','.ipynb','.csv','.db','.exe','.txt','.md') `
        -and $_.Name -ne "README.md" -and $_.Name -ne "requirements.txt" `
        -and $_.Name -ne ".gitignore" -and $_.Name -ne ".pre-commit-config.yaml" `
        -and $_.Name -ne "docker-compose.yml" -and $_.Name -notlike "Dockerfile*"
    } | Move-Item -Destination "$Root\misc"

    Write-Host "File organization complete!"