param(
  [ValidateSet("core", "full")]
  [string]$Profile = "full",
  [string]$Dataset = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts/check_offline_environment.py
if ($Dataset -ne "") { python scripts/install_austin_dataset.py --source $Dataset }
else { python scripts/install_austin_dataset.py }
python scripts/verify_merge.py
python scripts/run_austin_experiments.py --profile $Profile
python scripts/package_results.py
