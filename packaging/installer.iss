; AeroFlow Studio - Windows installer (Inno Setup 6).
;
; Build with packaging\build_release.ps1 (it compiles the launcher, fetches the Python installer and
; passes the version). What the installer does:
;   1. copies the app into %LOCALAPPDATA%\Programs\AeroFlow Studio (per-user: no administrator rights);
;   2. installs Python 3.12 (bundled, per-user) only if no 64-bit Python 3.10-3.14 is installed;
;   3. runs setup_windows.bat, which builds the private .venv and installs every library
;      (plus the NVIDIA GPU libraries when an NVIDIA driver is present) - needs internet access;
;   4. creates Start-menu / desktop shortcuts to "AeroFlow Studio.exe" and an uninstaller.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "AeroFlow Studio"
#define AppExe "AeroFlow Studio.exe"
#define PyInstaller "python-3.12.10-amd64.exe"

[Setup]
AppId={{6F1B3D52-8A4E-4C1B-9E7F-2D5A0C3B8E41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=AeroFlow
AppPublisherURL=https://github.com/albertmanoiu29-cell/aeroflow
AppSupportURL=https://github.com/albertmanoiu29-cell/aeroflow/issues
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile=aeroflow.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
OutputDir=..\dist
OutputBaseFilename=AeroFlowStudio-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nThe installer also sets up the Python libraries the program needs (and GPU acceleration: CUDA on NVIDIA graphics cards, OpenCL on AMD and Intel ones). This needs an internet connection and takes about 5-10 minutes; a console window shows the progress.%n%nIt is recommended that you close all other applications before continuing.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "openwith"; Description: "Add AeroFlow Studio to ""Open with"" for 3-D model files (.stl, .obj, .ply, .off)"; GroupDescription: "File types:"; Flags: unchecked

[Registry]
; AeroFlow's own project files (.aeroflow) open with the app by default.
Root: HKCU; Subkey: "Software\Classes\.aeroflow"; ValueType: string; ValueData: "AeroFlowStudio.Project"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Project"; ValueType: string; ValueData: "AeroFlow Studio project"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Project\DefaultIcon"; ValueType: string; ValueData: "{app}\aeroflow.ico"
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Project\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExe}"" ""%1"""
; "Open with" entries only (per user): the user's default program for these files is left untouched.
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Model"; ValueType: string; ValueData: "3-D model (AeroFlow Studio)"; Flags: uninsdeletekey; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Model\DefaultIcon"; ValueType: string; ValueData: "{app}\aeroflow.ico"; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\AeroFlowStudio.Model\shell\open\command"; ValueType: string; ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\.stl\OpenWithProgids"; ValueType: string; ValueName: "AeroFlowStudio.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\.obj\OpenWithProgids"; ValueType: string; ValueName: "AeroFlowStudio.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\.ply\OpenWithProgids"; ValueType: string; ValueName: "AeroFlowStudio.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith
Root: HKCU; Subkey: "Software\Classes\.off\OpenWithProgids"; ValueType: string; ValueName: "AeroFlowStudio.Model"; ValueData: ""; Flags: uninsdeletevalue; Tasks: openwith

[Files]
; The application (sources run by the private Python environment)
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements-gpu.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\setup_windows.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\run_windows.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\config\*.py"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "..\core\*.py"; DestDir: "{app}\core"; Flags: ignoreversion
Source: "..\ui\*.py"; DestDir: "{app}\ui"; Flags: ignoreversion
Source: "..\utils\*.py"; DestDir: "{app}\utils"; Flags: ignoreversion
Source: "..\visualization\*.py"; DestDir: "{app}\visualization"; Flags: ignoreversion
; The launcher and its icon
Source: "..\dist\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
Source: "aeroflow.ico"; DestDir: "{app}"; Flags: ignoreversion
; Python installer: extracted only when no suitable Python is found
Source: "redist\{#PyInstaller}"; Flags: dontcopy

[InstallDelete]
; Stale byte-code from an older version
Type: filesandordirs; Name: "{app}\config\__pycache__"
Type: filesandordirs; Name: "{app}\core\__pycache__"
Type: filesandordirs; Name: "{app}\ui\__pycache__"
Type: filesandordirs; Name: "{app}\utils\__pycache__"
Type: filesandordirs; Name: "{app}\visualization\__pycache__"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Comment: "Aerodynamics simulation studio"
Name: "{autoprograms}\{#AppName} - repair libraries"; Filename: "{app}\setup_windows.bat"; WorkingDir: "{app}"; IconFilename: "{app}\aeroflow.ico"; Comment: "Re-run the library setup (fixes a damaged installation)"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent; Check: SetupSucceeded
; One-click updates from inside the app run this installer with /SILENT: start the new version afterwards.
Filename: "{app}\{#AppExe}"; Flags: nowait; Check: WizardSilent and SetupSucceeded

[UninstallDelete]
; Everything the setup created inside the program folder (user data lives in %APPDATA%\aeroflow_studio)
Type: filesandordirs; Name: "{app}\.venv"
Type: files; Name: "{app}\setup.log"
Type: filesandordirs; Name: "{app}\config\__pycache__"
Type: filesandordirs; Name: "{app}\core\__pycache__"
Type: filesandordirs; Name: "{app}\ui\__pycache__"
Type: filesandordirs; Name: "{app}\utils\__pycache__"
Type: filesandordirs; Name: "{app}\visualization\__pycache__"
Type: dirifempty; Name: "{app}"

[Code]
var
  LibrariesOk: Boolean;

{ Silent / scripted installs: exit code 10 tells the caller that the library setup failed. }
function GetCustomSetupExitCode(): Integer;
begin
  if LibrariesOk then
    Result := 0
  else
    Result := 10;
end;

function SetupSucceeded(): Boolean;
begin
  Result := LibrariesOk;
end;

{ A registered 64-bit CPython 3.10-3.14 (python.org installers register under PythonCore). }
function FindPythonIn(RootKey: Integer; Ver: String): String;
var
  P: String;
begin
  Result := '';
  if RegQueryStringValue(RootKey, 'Software\Python\PythonCore\' + Ver + '\InstallPath', 'ExecutablePath', P) and FileExists(P) then
    Result := P
  else if RegQueryStringValue(RootKey, 'Software\Python\PythonCore\' + Ver + '\InstallPath', '', P) and FileExists(AddBackslash(P) + 'python.exe') then
    Result := AddBackslash(P) + 'python.exe';
end;

function FindPython(): String;
var
  Versions: TArrayOfString;
  I: Integer;
begin
  Result := '';
  SetArrayLength(Versions, 5);
  Versions[0] := '3.12'; Versions[1] := '3.13'; Versions[2] := '3.11'; Versions[3] := '3.14'; Versions[4] := '3.10';
  for I := 0 to GetArrayLength(Versions) - 1 do
  begin
    Result := FindPythonIn(HKCU, Versions[I]);
    if Result = '' then
      Result := FindPythonIn(HKLM64, Versions[I]);
    if Result <> '' then
      Exit;
  end;
end;

function InstallPython(): String;
var
  Code: Integer;
begin
  Result := '';
  WizardForm.StatusLabel.Caption := 'Installing Python 3.12 (per-user)...';
  ExtractTemporaryFile('{#PyInstaller}');
  if not Exec(ExpandConstant('{tmp}\{#PyInstaller}'),
              '/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_doc=0 SimpleInstall=1',
              '', SW_SHOW, ewWaitUntilTerminated, Code) or (Code <> 0) then
  begin
    SuppressibleMsgBox('Python 3.12 could not be installed (code ' + IntToStr(Code) + ').' + #13#10#13#10 +
           'Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ and then run ' +
           '"AeroFlow Studio - repair libraries" from the Start menu.', mbError, MB_OK, IDOK);
    Exit;
  end;
  Result := FindPython();
end;

{ Text inside a PowerShell single-quoted string: ' becomes ''. }
function PsQuote(S: String): String;
begin
  Result := S;
  StringChangeEx(Result, '''', '''''', True);
end;

procedure RunLibrarySetup();
var
  Python, Cmd: String;
  Code: Integer;
begin
  LibrariesOk := False;
  Python := FindPython();
  if Python = '' then
    Python := InstallPython();
  if Python = '' then
    Exit;
  WizardForm.StatusLabel.Caption := 'Installing the Python libraries (about 5-10 minutes, see the console window)...';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  { The console shows the progress; PowerShell's Tee-Object also keeps it in setup.log for support. }
  { stderr is merged inside cmd (PowerShell 5.1 would print native stderr as red error records). }
  Cmd := '-NoProfile -ExecutionPolicy Bypass -Command "$env:AEROFLOW_UNATTENDED=''1''; ' +
         '$env:AEROFLOW_PYTHON=''' + PsQuote(Python) + '''; ' +
         '& $env:ComSpec /c ''"' + PsQuote(ExpandConstant('{app}\setup_windows.bat')) + '" 2>&1'' | Tee-Object -FilePath ''' +
         PsQuote(ExpandConstant('{app}\setup.log')) + '''; exit $LASTEXITCODE"';
  if Exec('powershell.exe', Cmd, ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, Code) and (Code = 0) then
    LibrariesOk := True
  else
    SuppressibleMsgBox('The Python libraries could not be installed completely (code ' + IntToStr(Code) + ').' + #13#10#13#10 +
           'Check your internet connection, then run "AeroFlow Studio - repair libraries" from the Start menu ' +
           '(or setup_windows.bat in the program folder) to finish the setup.' + #13#10#13#10 +
           'The full output is in ' + ExpandConstant('{app}\setup.log'), mbError, MB_OK, IDOK);
  WizardForm.ProgressGauge.Style := npbstNormal;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    RunLibrarySetup();
end;

{ Some library files sit deeper than Windows' classic 260-character path limit, which the regular
  uninstall delete cannot remove; rmdir with the \\?\ prefix can. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Code: Integer;
begin
  if (CurUninstallStep = usPostUninstall) and DirExists(ExpandConstant('{app}\.venv')) then
  begin
    Exec(ExpandConstant('{cmd}'), '/c rmdir /s /q "\\?\' + ExpandConstant('{app}\.venv') + '"', '',
         SW_HIDE, ewWaitUntilTerminated, Code);
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
