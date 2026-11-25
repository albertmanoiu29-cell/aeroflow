# Changelog

All notable changes to AeroFlow Studio. Versions follow [semantic versioning](https://semver.org/);
download releases from the [Releases page](https://github.com/albertmanoiu29-cell/aeroflow/releases).

## [Unreleased]

### Added
- **Update check:** *Help ▸ Check for updates…*. An automatic check runs at most once a day and can be turned
  off in the Help menu. When a newer release exists, a status-bar button opens its release notes and
  **Download and install** fetches and starts the new installer. The only data sent is a generic
  `AeroFlow-Studio/<version>` user agent.
- **About and system report:** *Help ▸ About* and a new *System* tab with versions, GPU and compute back-ends,
  plus *Copy to clipboard*. Your home folder is shown as `~`.
- **Report a problem:** opens a pre-filled GitHub issue in your browser. Nothing is sent until you submit it.
- **File ▸ Open recent** (the last 10 models).
- **Installer:**
  - An optional "Open with" entry for `.stl`, `.obj`, `.ply` and `.off` files. It doesn't change your default app.
  - The library setup output is saved to `setup.log`.
  - Setup checks for an internet connection first and explains the problem when there is none.
  - Silent installs return exit code 10 when the library setup fails.
- Continuous integration: tests and lint on every push, and the installer is built and attached to a GitHub
  release whenever a `v*` tag is pushed.

### Fixed
- The installer no longer depends on `cmd` searching the current folder (fails on PCs with
  `NoDefaultCurrentDirectoryInExePath`).
- Silent installs could hang on an error message box.
- OS name in reports: Windows 11 was shown as Windows 10.

## [1.4.0] - 2026-11-24

### Added
- **GPU (CUDA) acceleration** for the Live Flow Lab and the Supersonic Lab, with an Auto / GPU / CPU compute
  device setting (*Simulation ▸ Compute device* or the Lab panels).
  - Supersonic Lab: about 60 FPS on the Fast to Detailed grids, up from about 13 FPS.
  - Live Flow Lab: 60 FPS on every grid, with 10–25× more steps per second.
- Pipelined frames: the next batch of solver steps runs while the current frame is drawn.
- Windows installer (`AeroFlowStudio-Setup-<version>.exe`) and native launcher (`AeroFlow Studio.exe`).

### Fixed
- OpenFOAM results now show the surface pressure (Cp) on the body.
- The Supersonic Lab's sonic line is drawn by a vectorised contour (it previously cost up to ~95 ms per frame).

### Changed
- OpenFOAM's default number of CPU cores is the physical cores minus one.

## [1.3.0]

- CFD wind tunnel, Live Flow Lab (lattice Boltzmann) and Supersonic Lab (compressible Euler).
