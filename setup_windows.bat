@echo off
rem AeroFlow Studio - one-step Windows setup.
rem Double-click this file (or run it from Command Prompt / PowerShell) in the aeroflow_studio folder.
rem It creates a private Python environment in .venv and installs every library into it, then checks
rem the libraries import. It never relies on "activate", so PowerShell script policies don't matter.
setlocal
cd /d "%~dp0"

echo.
echo === AeroFlow Studio setup ===
echo.

rem ---- 1. Find a suitable 64-bit Python (3.10 - 3.14) -------------------------------------------
set "PYEXE="
for %%V in (3.12 3.13 3.11 3.14 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys, struct; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys, struct; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) and struct.calcsize('P') == 8 else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo ERROR: No suitable Python found.
    echo Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/
    echo and tick "Add python.exe to PATH" in the installer, then run this file again.
    goto :fail
)
echo Using Python:
%PYEXE% -c "import sys; print('   ', sys.version.split()[0], sys.executable)"

rem ---- 2. Create (or re-create) the .venv environment -------------------------------------------
set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" (
    "%VPY%" -c "import sys, struct; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Existing .venv uses an unsupported Python - recreating it...
        rmdir /s /q "%~dp0.venv"
    )
)
if not exist "%VPY%" (
    echo Creating the .venv environment...
    %PYEXE% -m venv "%~dp0.venv"
    if errorlevel 1 goto :fail
)

rem ---- 3. Install the libraries into .venv ------------------------------------------------------
echo.
echo Installing libraries (first time takes a few minutes)...
"%VPY%" -m pip install --upgrade pip wheel
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail

rem ---- 4. Check that everything imports ---------------------------------------------------------
echo.
echo Checking the installation...
"%VPY%" -c "import PySide6, vtk, pyvista, pyvistaqt, pyqtgraph, numpy, scipy, matplotlib, PIL, trimesh; print('    All libraries import correctly.')"
if errorlevel 1 goto :fail

echo.
echo === Setup complete. Start the app with run_windows.bat ===
echo.
pause
exit /b 0

:fail
echo.
echo === Setup did NOT finish. Scroll up for the first red/ERROR line and send it to support. ===
echo.
pause
exit /b 1
