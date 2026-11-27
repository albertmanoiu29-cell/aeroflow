# Changelog

All notable changes to AeroFlow Studio. Versions follow [semantic versioning](https://semver.org/);
download releases from the [Releases page](https://github.com/albertmanoiu29-cell/aeroflow/releases).

## [1.8.0] - 2026-11-27

### Added: Rocket Nozzle Lab design tools
- **Propellant mixture ratio (O/F):** each propellant's chamber temperature, γ and molar mass follow the O/F you set; a sweep marks the best-Isp ratio (kerolox about 2.3).
- **Heat transfer and cooling:** Bartz wall heat flux along the nozzle with regenerative, radiation-cooled, mixed (regenerative chamber plus radiation-cooled extension) or adiabatic walls. The 3-D hot-wall glow uses this wall temperature.
- **Flow separation:** the Schmucker criterion is drawn on the wall pressure profile, and the predicted separation point is reported.
- **Expansion-ratio optimiser:** finds the expansion ratio with the best Isp at one altitude, or averaged over an altitude range, optionally rejecting ratios that would separate at the lowest altitude.
- **Noise estimate** (NASA SP-8072): overall sound level against distance and direction, and the spectrum peak.
- **Ascent profile:** flies the engine from sea level to a chosen altitude while the simulation runs, and plots simulated thrust against theory.
- **Engine comparison** of saved designs side by side, and a **PDF report** with the charts and results.
- **Dual-bell nozzles** that switch between a low-altitude and a high-altitude mode, and **imported contours** from CSV, TXT or DXF files.
- **Auto grid resolution:** picks the smallest grid that puts at least 12 cells across the throat.
- **Undo / redo** (Ctrl+Z / Ctrl+Y) and engine design files (`.aeronozzle`, Ctrl+S / Ctrl+O).

### Added: visuals
- Afterburning in the mixing layer for fuel-rich exhausts, bloom on the bright core, smoke for solid motors.
- 3-D tracer particles that follow the exhaust and the entrained air.

### Added: new tools
- **Propeller analysis** (*Simulation ▸ Propeller analysis*): blade-element momentum theory with Prandtl tip and hub losses. It gives thrust, power and efficiency curves against advance ratio and the loading along the blade, and can build the propeller in 3-D.
- **Flight dynamics** (*Simulation ▸ Flight dynamics*): from the computed polar, trims the aircraft (angle of attack and elevator) and finds the short-period and phugoid modes. It simulates the response to an elevator pulse or a gust and adds plain-language handling notes.
- **CAD import:** STEP, IGES and BREP files are meshed with gmsh and converted from millimetres to metres.
- **OpenFOAM options:** a choice of turbulence model (k-ω SST, Spalart-Allmaras, realizable or standard k-ε, laminar), extra surface refinement and wake refinement.

### Improved
- **Speed:** GPU kernels are compiled during setup, so the first run starts at once. A single-precision GPU option makes the Supersonic Lab about 5.8x faster, with the same drag coefficient to five digits.
- **Usability:**
  - a light theme and an interface-size setting (*View ▸ Appearance*);
  - an in-app guide (F1 in the Nozzle Lab, Shift+F1 elsewhere);
  - problem reports now bundle the log files.

### Not included
- **Two-gas (mixture) exhausts, engine clusters and local grid refinement.** These need a multi-species solver, a 3-D solver and a non-uniform grid respectively; the Nozzle Lab's solver is 2-D, single-gas and uniform. The Auto grid above is the part of local refinement that fits the current solver.
- **Aerospike nozzles** need an annular inlet and an inner plug wall, which the solver does not support yet.

## [1.7.2] - 2026-11-27

### Fixed: Rocket Nozzle Lab layout
- **The control panel was too narrow and could hardly be widened**, which clipped some inputs.
  - **Cause:** long, non-wrapping chart titles forced the analysis panel to at least ~590 px.
  - **Fix:** the titles are now short and wrap, and their explanations moved into the notes under the charts, so the analysis panel can shrink to 260 px. The control panel can no longer be squeezed below the width its inputs need, and it can be dragged much wider.
- **Panel buttons:** *Hide controls* and *Hide analysis* buttons beside the 2-D / 3-D switch give the flow view, or the controls, the whole width.
- **Screen fit:** the window and the starting panel widths fit the screen (a 1920-pixel display at 125 % scaling included).
- **Remembered layout:** panel widths and visibility carry over between sessions.

## [1.7.1] - 2026-11-26

### Improved: smoother, more realistic 3-D nozzle view
- **No more rings:** the solution is revolved with linear interpolation in radius instead of nearest-row lookup, removing the concentric "wood-grain" rings. A compiled, parallel kernel fills the voxels (21 million in about 9 ms) straight into VTK's memory.
- **Smoother updates:** the view updates about 10 times a second instead of 4. Voxels are up to 2.5x finer, ray steps are finer, interpolation is linear, the nozzle shell is tessellated twice as finely with crease-aware normals, and the cut faces use filtered, mip-mapped textures.
- **More realistic:**
  - a night-time background for the exhaust glow, so the flame is the light source;
  - a more translucent flame with lifted mid-tones, so the bright core and shock diamonds shine through the cooler outer layer as in photographs;
  - a warm key light on brushed-steel walls;
  - **hot nozzle glow:** the wall takes a blackbody colour from its estimated temperature (half the local gas temperature; the chamber is treated as regeneratively cooled), as on radiation-cooled nozzle extensions. It can be switched off in the 3-D toolbar.

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
