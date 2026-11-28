@echo off
rem AeroFlow Studio - one-step Windows setup.
rem Double-click this file (or run it from Command Prompt / PowerShell) in the aeroflow_studio folder.
rem It creates a private Python environment in .venv and installs every library into it, then checks
rem the libraries import. It never relies on "activate", so PowerShell script policies don't matter.
rem Used by the installer with:  AEROFLOW_PYTHON=<full path of python.exe>  (which Python to use) and
rem AEROFLOW_UNATTENDED=1  (no "press any key" pauses).
setlocal
cd /d "%~dp0"

echo.
echo === AeroFlow Studio setup ===
echo.

rem ---- 1. Find a suitable 64-bit Python (3.10 - 3.14) -------------------------------------------
set "PYEXE="
if defined AEROFLOW_PYTHON if exist "%AEROFLOW_PYTHON%" set "PYEXE="%AEROFLOW_PYTHON%""
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
"%VPY%" -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=15)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Cannot reach the Python package index ^(https://pypi.org^).
    echo The libraries are downloaded from there: check the internet connection, proxy or firewall,
    echo then run this setup again.
    goto :fail
)
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
"%VPY%" -c "import numba; print('    numba', numba.__version__, '- compiled CPU kernels enabled.')"
if errorlevel 1 echo    WARNING: numba is missing - the CPU solvers will run 10-20x slower.

rem ---- 5. Optional: NVIDIA GPU acceleration for the Live Flow Lab and Supersonic Lab -------------
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo No NVIDIA GPU driver found - the Labs use an AMD or Intel GPU through OpenCL if one is present.
    "%VPY%" -c "from core import compute; i = compute.gpu_info(); print('   ', i.summary())"
) else (
    echo.
    echo NVIDIA GPU detected - installing CUDA acceleration (CuPy, no CUDA Toolkit needed^)...
    "%VPY%" -m pip install -r "%~dp0requirements-gpu.txt"
    if errorlevel 1 (
        echo    WARNING: GPU libraries could not be installed - the Labs will run on the CPU.
    ) else (
        "%VPY%" -c "from core import compute; i = compute.gpu_info(); print('   ', i.summary())"
    )
)

rem ---- 6. Compile the solver kernels now, so the first start of the app is instant ----------------
echo.
echo Compiling the solver kernels (one time, about a minute)...
pushd "%~dp0"
"%VPY%" -m core.precompile
if errorlevel 1 echo    WARNING: kernel pre-compilation failed - they will compile on first use instead.
popd

echo.
echo === Setup complete. Start the app with run_windows.bat ===
echo.
if not defined AEROFLOW_UNATTENDED pause
exit /b 0

:fail
echo.
echo === Setup did NOT finish. Scroll up for the first red/ERROR line and send it to support. ===
echo.
if not defined AEROFLOW_UNATTENDED pause
exit /b 1
