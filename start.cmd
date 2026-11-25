@echo off
setlocal
cd /d "%~dp0"
if defined LEDGER_PYTHON (
    "%LEDGER_PYTHON%" app.py --open
    exit /b
)
python -c "import numpy, pypdf" >nul 2>&1
if not errorlevel 1 (
    python app.py --open
    exit /b
)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "(Get-AppxPackage -Name 'OpenAI.CodexPrimaryRuntime*' | Sort-Object Version -Descending | Select-Object -First 1).InstallLocation"`) do (
    if exist "%%I\dependencies\python\python.exe" (
        "%%I\dependencies\python\python.exe" -c "import numpy, pypdf" >nul 2>&1
        if not errorlevel 1 (
            "%%I\dependencies\python\python.exe" app.py --open
            exit /b
        )
    )
)
echo Install Python 3.12 or later and run the following command:
echo python -m pip install --no-cache-dir -r requirements.txt
pause
exit /b 1
