# Changelog

All notable changes to AeroFlow Studio. Versions follow [semantic versioning](https://semver.org/);
download releases from the [Releases page](https://github.com/albertmanoiu29-cell/aeroflow/releases).

## [1.5.0] - 2026-11-25

### Added: 3-D visualization
- **Mouse tools**:
  - **Point probe:** click to read the local speed, Cp, total pressure and vorticity.
  - **Line probe:** two clicks plot the flow along the line, with CSV export.
  - **Seed:** click to release a bundle of streamlines from that spot (for example at a wing tip, to see the tip vortex).
- **Surface oil-flow streaks:** the direction of the air right at the skin, showing attachment and separation lines and tip roll-up.
- **Three orthogonal slices** and an animated **slice sweep**.
- **λ2 vortex criterion** for vortex-core isosurfaces, next to Q.
- **Lock colour ranges**, so two results are coloured on the same scale.
- **Spanwise Cp sections:** chordwise pressure at chosen span stations, with the cuts drawn on the model.
- **Compare results:** pin a result, then view it side by side with the current one. The two views share a camera and colour scales, with a table of coefficient differences and a ΔCp surface.
- **Exports:**
  - transparent-background 4K screenshots;
  - MP4 video (in addition to GIF);
  - the flow field, surface and streamlines as VTK for **ParaView**, plus *Open in ParaView*.
- **Camera bookmarks:** Ctrl+Alt+1…3 saves a view, Alt+1…3 returns to it.

### Added: analysis
- **Parametric study:** sweep sideslip, speed (Reynolds and Mach), altitude or angle of attack, with charts, a table and CSV export.
- **Stability:** lift-curve and pitching-moment slopes, neutral point, static margin and trim angle from every polar.
- **Lifting-line theory overlay** on the polar charts (Helmbold lift slope and ideal drag polar), for validation at a glance.
- **Grid convergence check:** solves on three grids and reports a Richardson-extrapolated value and the GCI (Celik et al. 2008).
- **Compressibility:** Kármán-Tsien correction of the surface pressures, and a warning once the flow turns locally supersonic.
- **Ground effect** (image method) with a ground-height h/b control.
- **Flaps and ailerons:** deflect plain control surfaces on any wing (hinge position, span range, angle), with undo.

### Added: OpenFOAM
- **Polar sweeps with OpenFOAM:** the mesh is built once, then one solver run per angle.
- **Skin friction Cf and y⁺** on the body (*Surface shows*), with a warning when y⁺ is outside the wall-function range.
- **checkMesh** summary after meshing, a **boundary-layer count** control, and a live **CL / CD convergence** chart.

### Added: Labs
- **Airfoil coordinate files** (`.dat`, Selig and Lednicer formats) in the Live Flow Lab and the Supersonic Lab.
- **Time-averaged and RMS-fluctuation fields** in the Live Flow Lab.
- **Supersonic inlet** (ramp + cowl): the cowl is placed for shock-on-lip at the design Mach number.
- **GPU-only Extreme grids**, and **MP4 recording** of both Labs.

### Added: application
- **Project files (`.aeroflow`):** save and reopen a whole session (geometry, conditions, settings, camera, polar). The installer associates `.aeroflow` files with the app.
- **PDF reports** (A4, print-friendly) alongside HTML.
- **Units:** SI or Imperial (kt, ft, °F, lbf).

### Fixed
- OpenFOAM 12+ iteration counting (`Time = 1s`): the residual and coefficient histories were collapsed onto one point.
- A test now imports every UI module, so a broken UI file can't pass the suite.

## [1.4.1] - 2026-11-25

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
