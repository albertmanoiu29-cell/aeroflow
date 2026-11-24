# AeroFlow Studio - build the Windows release:
#   dist\AeroFlow Studio.exe                  the launcher (native, no console window)
#   dist\AeroFlowStudio-Setup-<version>.exe    the installer (bundles Python 3.12 for PCs without Python)
#
# Requirements: Windows 10/11 and Inno Setup 6 (winget install JRSoftware.InnoSetup).
# The C# compiler of the .NET Framework that ships with Windows builds the launcher.
# Usage (PowerShell):  powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1

$ErrorActionPreference = "Stop"
$pkg = $PSScriptRoot
$root = Split-Path $pkg -Parent
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force $dist | Out-Null

# -- version from config/settings.py ---------------------------------------------------------------
$m = Select-String -Path (Join-Path $root "config\settings.py") -Pattern 'APP_VERSION: Final\[str\] = "([0-9.]+)"'
if (-not $m) { throw "APP_VERSION not found in config\settings.py" }
$version = $m.Matches[0].Groups[1].Value
Write-Host "AeroFlow Studio $version"

# -- 1. launcher ------------------------------------------------------------------------------------
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path $csc)) { throw "C# compiler not found: $csc" }
$obj = Join-Path $dist "obj"
New-Item -ItemType Directory -Force $obj | Out-Null
$versionCs = Join-Path $obj "AssemblyVersion.cs"
"[assembly: System.Reflection.AssemblyVersion(""$version.0"")]`n[assembly: System.Reflection.AssemblyFileVersion(""$version.0"")]" |
    Set-Content -Encoding UTF8 $versionCs
$exe = Join-Path $dist "AeroFlow Studio.exe"
& $csc /nologo /target:winexe /optimize+ "/win32icon:$pkg\aeroflow.ico" /r:System.Windows.Forms.dll "/out:$exe" `
    "$pkg\launcher\AeroFlowLauncher.cs" $versionCs
if ($LASTEXITCODE -ne 0) { throw "Launcher build failed" }
Write-Host "Launcher:  $exe"

# -- 2. bundled Python installer (verified signature) ----------------------------------------------
$redist = Join-Path $pkg "redist"
New-Item -ItemType Directory -Force $redist | Out-Null
$py = Join-Path $redist "python-3.12.10-amd64.exe"
if (-not (Test-Path $py)) {
    Write-Host "Downloading the Python 3.12.10 installer..."
    Invoke-WebRequest -UseBasicParsing "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $py
}
$sig = Get-AuthenticodeSignature $py
if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "Python Software Foundation") {
    Remove-Item $py
    throw "The Python installer's signature is not valid - deleted it; run the build again."
}

# -- 3. installer -----------------------------------------------------------------------------------
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Inno Setup 6 not found (winget install JRSoftware.InnoSetup)" }
& $iscc /Q "/DAppVersion=$version" (Join-Path $pkg "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "Installer build failed" }
Write-Host "Installer: $(Join-Path $dist "AeroFlowStudio-Setup-$version.exe")"
