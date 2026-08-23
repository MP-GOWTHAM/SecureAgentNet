@echo off
REM SecureAgentNet -- Windows setup (Command Prompt).
REM Same steps as the repo's macOS setup block in README.md.
REM Usage from the repo root:  scripts\setup_windows.bat  [python-version]

setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

set "PYVER=%~1"
if "%PYVER%"=="" set "PYVER=3.13"

where py >nul 2>&1
if errorlevel 1 (
    echo ERROR: the 'py' launcher was not found. Install Python 3.11+ from python.org.
    exit /b 1
)

if not exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    echo Creating virtual environment ^(.venv^) with Python %PYVER% ...
    py -%PYVER% -m venv "%REPO_ROOT%\.venv"
    if errorlevel 1 exit /b 1
) else (
    echo Reusing existing virtual environment at .venv
)

set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing dependencies from requirements-windows.txt ...
"%VENV_PY%" -m pip install -r "%REPO_ROOT%\requirements-windows.txt"
if errorlevel 1 exit /b 1

echo Running the test suite ...
"%VENV_PY%" -m pytest secureagentnet/tests

echo.
echo Done. Activate the environment with:
echo     .venv\Scripts\activate.bat
echo Then, e.g.:
echo     python -m secureagentnet.webapp.app
endlocal
