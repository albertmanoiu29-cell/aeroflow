// AeroFlow Studio launcher: "AeroFlow Studio.exe".
//
// A tiny native Windows program that sits next to main.py and starts the app with the private Python
// environment (.venv) made by setup_windows.bat - without a console window. If the environment does not
// exist yet it offers to run the setup; if the app stops right after starting it points to the log.
//
// Build (no SDK needed, the C# compiler ships with Windows' .NET Framework):
//   csc /target:winexe /win32icon:aeroflow.ico /r:System.Windows.Forms.dll /out:"AeroFlow Studio.exe" AeroFlowLauncher.cs
// packaging\build_release.ps1 does this for you.

using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("AeroFlow Studio")]
[assembly: AssemblyProduct("AeroFlow Studio")]
[assembly: AssemblyDescription("AeroFlow Studio - aerodynamics simulation")]
// AssemblyVersion / AssemblyFileVersion come from a generated file (build_release.ps1, from APP_VERSION).

static class Launcher
{
    const string Title = "AeroFlow Studio";

    [STAThread]
    static int Main(string[] args)
    {
        string dir = AppDomain.CurrentDomain.BaseDirectory;
        string main = Path.Combine(dir, "main.py");
        string pythonw = Path.Combine(dir, ".venv", "Scripts", "pythonw.exe");
        string setup = Path.Combine(dir, "setup_windows.bat");

        if (!File.Exists(main))
        {
            Error("main.py was not found next to the launcher:\n" + dir +
                  "\n\nKeep \"AeroFlow Studio.exe\" in the AeroFlow Studio folder.");
            return 1;
        }

        if (!File.Exists(pythonw))
        {
            if (!File.Exists(setup))
            {
                Error("The Python environment (.venv) is missing and setup_windows.bat was not found.");
                return 1;
            }
            DialogResult answer = MessageBox.Show(
                "AeroFlow Studio is not set up on this computer yet.\n\n" +
                "Run the setup now? It installs the required libraries into a private folder " +
                "(needs an internet connection and takes a few minutes).",
                Title, MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (answer != DialogResult.Yes)
                return 1;
            var psi = new ProcessStartInfo("cmd.exe", "/c \"\"" + setup + "\"\"")
            {
                WorkingDirectory = dir,
                UseShellExecute = false,
            };
            using (Process p = Process.Start(psi))
                p.WaitForExit();
            if (!File.Exists(pythonw))
            {
                Error("The setup did not finish, so AeroFlow Studio cannot start.\n\n" +
                      "Run setup_windows.bat in the AeroFlow Studio folder to see what went wrong.");
                return 1;
            }
        }

        var sb = new StringBuilder(Quote(main));
        foreach (string a in args)
            sb.Append(' ').Append(Quote(a));              // e.g. a mesh file dropped onto the exe
        var start = new ProcessStartInfo(pythonw, sb.ToString())
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
        };
        try
        {
            using (Process app = Process.Start(start))
            {
                // Watch the start-up only: a crash in the first seconds is reported, then we step aside.
                if (app.WaitForExit(20000) && app.ExitCode != 0)
                {
                    string logs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                                               "aeroflow_studio", "logs");
                    Error("AeroFlow Studio stopped while starting (exit code " + app.ExitCode + ").\n\n" +
                          "The log is in:\n" + logs + "\n\n" +
                          "Running run_windows.bat in the AeroFlow Studio folder shows the error messages; " +
                          "running setup_windows.bat again repairs a damaged installation.");
                    return app.ExitCode;
                }
            }
        }
        catch (Exception ex)
        {
            Error("Could not start Python:\n" + pythonw + "\n\n" + ex.Message +
                  "\n\nRun setup_windows.bat again to repair the installation.");
            return 1;
        }
        return 0;
    }

    static string Quote(string s)
    {
        // Windows command-line quoting (backslashes before a quote are doubled).
        var sb = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in s)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { sb.Append('\\', slashes * 2 + 1).Append('"'); slashes = 0; continue; }
            sb.Append('\\', slashes).Append(c);
            slashes = 0;
        }
        sb.Append('\\', slashes * 2).Append('"');
        return sb.ToString();
    }

    static void Error(string text)
    {
        MessageBox.Show(text, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }
}
