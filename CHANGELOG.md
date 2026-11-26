# Changelog

All notable changes to AeroFlow Studio. Versions follow [semantic versioning](https://semver.org/);
download releases from the [Releases page](https://github.com/albertmanoiu29-cell/aeroflow/releases).

## [1.7.0] - 2026-11-26

### Added: 3-D view in the Rocket Nozzle Lab
- **A 2-D section / 3-D view switch** above the canvas. The 3-D view revolves the live solution about the nozzle axis (a planar nozzle is extruded across its width instead) and follows the selected field while the engine runs.
- **Volume mode:** GPU volume rendering of the exhaust plume, with the surrounding air transparent. The shock diamonds and Mach disks appear as a glowing 3-D body, and the nozzle is drawn as a 270° cut-away shell so the white-hot chamber and the throat stay visible. The voxels are concentrated on the plume and re-fitted as it grows.
- **Cut-away mode:** a 90° notch out of the engine and plume. The field is painted on both cut faces (still air is see-through for the glow), and the plume boundary (50 % exhaust) is a translucent revolved surface.
- **Camera presets** (side, three-quarter, from behind, nozzle close-up) with mouse orbit and zoom. *Snapshot* saves the 3-D view when it is shown.
- The 2-D view skips drawing while it is hidden, so the solver keeps the time.

## [1.6.1] - 2026-11-26

### Improved: Rocket Nozzle Lab speed (about 9x faster start-ups)
- **Single-precision GPU kernels**, the new default. They are 6.4x faster per step on consumer GPUs, which have few double-precision units: 0.25 ms instead of 1.6 ms per step on the default grid. Results agree with double precision to about 0.1 %. *GPU precision: Double (exact)* in the Simulation card keeps the old kernels, which match the CPU exactly. Time and time step stay in double precision, so long runs do not drift.
- **Larger stable time step** (CFL 0.7 instead of 0.5). All eight presets stay stable, and their thrust, mass flow and exit Mach number are unchanged.
- **Faster measurement:** vectorised shock-cell detection (6.5x) and wall-pressure binning.
- **Stricter steady-state indicator:** it also requires the throat and exit mass flows, and the wall thrust and exit-plane thrust, to agree. A start-up plateau with the starting shock still in the nozzle no longer counts as steady.

## [1.6.0] - 2026-11-26

### Added: Rocket Nozzle Lab (Ctrl+5)
- **Rocket engine nozzle and exhaust plume, live from ignition.** It solves the compressible Euler equations, axisymmetric or planar, on the GPU or the CPU. Ignition and shut-down ramps, live throttling, and live changes of altitude and flight Mach number.
- **Everything is editable:**
  - propellant (9 presets or custom γ, molar mass and chamber temperature);
  - chamber pressure or mass flow;
  - throat, exit and chamber diameters, chamber length, convergent angle;
  - a Rao bell, conical or sonic contour with its angles, throat radii, wall thickness and a vehicle body;
  - ISA altitude or ambient pressure and temperature;
  - grid and plume region.

  Eight engine presets are included.
- **Views:** exhaust glow (flame colour by propellant), Mach number, pressure relative to ambient, temperature, velocity, density, schlieren and exhaust fraction.
- **Overlays:** the sonic line, detected Mach disks and shock cells, the theoretical shock-cell spacing, the separation point and streamlines. Hover for local values.
- **Results against 1-D theory:**
  - thrust as the force on the chamber and nozzle, with the exit-plane and base-pressure contributions;
  - Isp, CF, c*, mass flow and the exit state;
  - the expansion regime, the Summerfield separation point, the 1-D normal-shock position;
  - shock-cell length (Prandtl–Pack), the Mach-disk position (Ashkenas–Sherman) and the supersonic core length.
- **Charts:** thrust, chamber pressure and mass flow from ignition; centre-line, exit-plane and wall-pressure profiles; thrust and Isp against altitude; the nozzle contour.
- **Exports:** CSV (table, histories, profiles), PNG snapshots and MP4 video.
- **Numerics:**
  - cut-cell (embedded-boundary) walls: exactly conservative, with the exact piston wall pressure;
  - an axisymmetric finite-volume update that keeps a gas at rest exactly at rest;
  - a non-reflecting characteristic injector inlet and an entrainment far field;
  - CUDA kernels whose boundary parameters live in device memory, so throttling never rebuilds the CUDA graphs.

## [1.5.1] - 2026-11-25

### Fixed
- **OpenFOAM polar sweeps** crashed as soon as you confirmed the sweep. They now run again, including compressible and rotating-propeller sweeps.

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
- **Compressible air** (`fluid` / rhoSimpleFoam, perfect gas, freestream far field). It switches on automatically from Mach 0.3 and adds a local **Mach number** field.
- **Transient runs** (PIMPLE with a Courant-adaptive time step and saved frames) with a **player**: frame slider and Play, CL and CD against t*, and the **Strouhal number** of vortex shedding.
- **Rotating propellers and rotors** (MRF zone built with createZones or topoSet): thrust, torque, power, advance ratio J, CT, CP and efficiency. Includes a built-in **propeller** geometry preset.

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
