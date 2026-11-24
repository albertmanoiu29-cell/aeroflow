@echo off
rem AeroFlow Studio - start the app with the Python environment made by setup_windows.bat.
rem Usage: double-click, or "run_windows.bat path\to\model.stl" to open a mesh at start-up.
setlocal
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo The .venv environment was not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
"%VPY%" "%~dp0main.py" %*
if errorlevel 1 (
    echo.
    echo AeroFlow Studio exited with an error - see the messages above.
    pause
)
