<#
    SecureAgentNet - one-command Windows bootstrap.

    Takes a fresh `git clone` to a running web app. setup_windows.ps1 only
    creates the venv and installs requirements; this adds the three things
    a clone cannot carry:

      1. A CUDA build of torch. pip installs the CPU wheel by default on
         Windows, so torch.cuda.is_available() returns False even with a
         working driver. Blackwell (RTX 50-series, sm_120) additionally
         needs cu128 or newer - cu124 produces no working kernels.
      2. The trained checkpoints. Five of the project's checkpoints are
         253 MB each against GitHub's 100 MB per-file hard limit, so they
         are published to the Hugging Face Hub instead (see
         scripts/publish_models.py).
      3. The `combined_max_v7` config, which just references two downloaded
         members and is written locally.

    NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1
    reads .ps1 as ANSI unless the file carries a UTF-8 BOM, which turns
    any non-ASCII character into mojibake and breaks parsing.

    Usage (from the repo root):
        powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows.ps1

    Options:
        -ModelsRepo <id>    HF repo holding the checkpoints
        -CudaIndex  <url>   torch wheel index (cu128 default; see table below)
        -SkipModels         set up the environment only
        -SkipCuda           keep the CPU build of torch

    GPU -> wheel index:
        Blackwell  (RTX 50xx, sm_120)  cu128 or newer   <- default
        Ada/Ampere (RTX 40xx/30xx)     cu124
        no NVIDIA GPU                  use -SkipCuda
#>
[CmdletBinding()]
param(
    # Published by scripts/publish_models.py. Public, 571 MB: the current
    # recommended checkpoints plus the superseded ones, kept so earlier
    # results stay reproducible. Override for your own copy.
    [string]$ModelsRepo = "mpgowtham/secureagentnet-models",
    [string]$CudaIndex  = "https://download.pytorch.org/whl/cu128",
    [string]$PythonVersion = "3.13",
    [switch]$SkipModels,
    [switch]$SkipCuda
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

# --- 1. environment ---------------------------------------------------
Step 1 "Virtual environment and dependencies"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The 'py' launcher was not found. Install Python 3.11+ from python.org and re-run."
}
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & py "-$PythonVersion" -m venv (Join-Path $RepoRoot ".venv")
} else {
    Write-Host "  reusing existing .venv"
}
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-windows.txt") --quiet
Write-Host "  dependencies installed"

# --- 2. CUDA torch ----------------------------------------------------
if (-not $SkipCuda) {
    Step 2 "CUDA build of PyTorch ($CudaIndex)"
    $cudaOk = & $VenvPython -c 'import torch;print(torch.cuda.is_available())' 2>$null
    if ($cudaOk -eq "True") {
        Write-Host "  CUDA already available - skipping (~3 GB download avoided)"
    } else {
        Write-Host "  installing (~3 GB, several minutes) ..."
        & $VenvPython -m pip install --upgrade --force-reinstall torch --index-url $CudaIndex --quiet
        # The CUDA index pulls a newer fsspec than `datasets` allows.
        & $VenvPython -m pip install "fsspec[http]<=2026.6.0" --quiet
    }
    & $VenvPython -c 'import torch;print("  torch", torch.__version__, "| cuda", torch.cuda.is_available())'
} else {
    Step 2 "Skipping CUDA (CPU build retained - training will be 30-40x slower)"
}

# --- 3. checkpoints ---------------------------------------------------
$ModelsDir = Join-Path $RepoRoot "secureagentnet\data\models"
if (-not $SkipModels) {
    Step 3 "Trained checkpoints"
    if ([string]::IsNullOrWhiteSpace($ModelsRepo)) {
        Write-Host "  No -ModelsRepo given and none configured." -ForegroundColor Yellow
        Write-Host "  Publish them once with:  python scripts\publish_models.py --repo-id <user>/secureagentnet-models"
        Write-Host "  then re-run with:        -ModelsRepo <user>/secureagentnet-models"
        Write-Host "  Or train locally (~45 min on GPU) - see README 'Training from scratch'."
    } else {
        New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
        # Built by concatenation on ONE line: PowerShell here-strings are
        # fragile about the closing delimiter's column and silently
        # reparse the embedded Python as PowerShell when they break.
        $dl = "from huggingface_hub import snapshot_download; print('  downloaded to', snapshot_download(repo_id='$ModelsRepo', local_dir=r'$ModelsDir'))"
        & $VenvPython -c $dl
    }
}

# --- 4. combined_max config -------------------------------------------
Step 4 "combined_max configuration"
$members = @("ensemble_v6_smooth3", "v3")
$haveAll = $true
foreach ($m in $members) {
    if (-not (Test-Path (Join-Path $ModelsDir "$m\config.json"))) { $haveAll = $false }
}
if ($haveAll) {
    $CombinedDir = Join-Path $ModelsDir "combined_max_v7"
    $mk = "from secureagentnet.detector.combined import CombinedRiskModel, CombinedRiskModelConfig; CombinedRiskModel(CombinedRiskModelConfig(members=['ensemble_v6_smooth3','v3'], mode='max')).save(r'$CombinedDir'); print('  combined_max written (references both members, no weight duplication)')"
    & $VenvPython -c $mk
} else {
    Write-Host "  members not present yet - skipped" -ForegroundColor Yellow
}

# --- 5. verify --------------------------------------------------------
Step 5 "Verifying"
& $VenvPython -m pytest (Join-Path $RepoRoot "secureagentnet\tests") -q
Write-Host ""
if (Test-Path (Join-Path $ModelsDir "combined_max_v7\config.json")) {
    Write-Host "Ready. Start the app with:" -ForegroundColor Green
    Write-Host ('  $env:SECUREAGENTNET_MODEL_DIR="' + (Join-Path $ModelsDir "combined_max_v7") + '"')
    Write-Host "  .\.venv\Scripts\python.exe -m secureagentnet.webapp.app"
    Write-Host "  then open http://127.0.0.1:5050"
} else {
    Write-Host "Environment ready, but no checkpoint is installed - the web app needs one." -ForegroundColor Yellow
    Write-Host "Either re-run with -ModelsRepo, or train locally (README: 'Training from scratch')."
}

