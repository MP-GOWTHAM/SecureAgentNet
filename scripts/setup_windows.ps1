<#
    SecureAgentNet - Windows setup (PowerShell).

    Same steps as the repo's macOS setup block in README.md
    (`python3 -m venv .venv && source .venv/bin/activate; pip install -r
    requirements.txt; pytest secureagentnet/tests`), expressed for Windows.

    Usage (from the repo root):
        powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
    Optional:
        -PythonVersion 3.13     # any interpreter the `py` launcher knows
        -SkipTests
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.13",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host "Repo root: $RepoRoot"

# pyproject.toml declares requires-python >= 3.11.
$py = (Get-Command py -ErrorAction SilentlyContinue)
if ($null -eq $py) {
    throw "The 'py' launcher was not found. Install Python 3.11+ from python.org and re-run."
}

$VenvDir = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment (.venv) with Python $PythonVersion ..."
    & py "-$PythonVersion" -m venv $VenvDir
} else {
    Write-Host "Reusing existing virtual environment at .venv"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { throw "venv creation failed: $VenvPython not found" }

$reported = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "venv Python: $reported"
$parts = $reported.Split('.')
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 11)) {
    throw "SecureAgentNet requires Python >= 3.11 (pyproject.toml); venv has $reported"
}

Write-Host "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements-windows.txt ..."
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-windows.txt")

if (-not $SkipTests) {
    Write-Host "Running the test suite ..."
    & $VenvPython -m pytest secureagentnet/tests
}

Write-Host ""
Write-Host "Done. Activate the environment with:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "Then, e.g.:"
Write-Host "    python -m secureagentnet.webapp.app"
