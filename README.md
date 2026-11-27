# AeroFlow Studio

**Drag-and-drop 3-D CFD, streamline visualisation and polar analysis, in one desktop app.**

[Download the Windows installer](https://github.com/albertmanoiu29-cell/aeroflow/releases/latest) ·
[Changelog](CHANGELOG.md) · [Report a problem](https://github.com/albertmanoiu29-cell/aeroflow/issues)
(or *Help ▸ Report a problem* in the app)

You can import an STL, OBJ or STEP file, or start from one of the six built-in shapes, and press **Run** to get:

- 3-D streamlines.
- The surface pressure map.
- CL, CD, CM and L/D.

A typical solve takes about a second, so the flow updates live as you move the sliders.

| Mode | What you get |
|---|---|
| **Wind Tunnel CFD** | Stream tubes (from a line, plane or sphere seed rake), animated smoke tracers, a surface Cp heatmap, a movable slice plane with velocity arrows, a floor grid, KPI cards on the view, 4K screenshots and GIF recording |
| **Polar Analysis** | An angle-of-attack sweep giving the lift curve (CL,max and stall marked), drag polar, L/D (best cruise angle marked) and pitching moment, with CSV and HTML report export |
| **Live Flow Lab** | A real-time 2-D wind tunnel (Lattice-Boltzmann): vortex streets, wakes and stall form live around an airfoil, a cylinder, other shapes, a cross-section of your 3-D model, or walls you paint with the mouse. Shows speed, vorticity, pressure or smoke, with live CL / CD and the shedding Strouhal number |

Colour conventions are the same in every view:

- **Speed:** the colour scale is centred on the freestream speed, so undisturbed air is the middle colour. Accelerated air is warm and slowed air (stagnation, wake) is cool.
- **Pressure (Cp):** a diverging blue–white–red scale that is symmetric about Cp = 0. Blue is suction and red is high pressure.

## What's new since 1.7 (next release)

- **Rocket Nozzle Lab design tools:**
  - O/F mixture ratio;
  - Bartz heat transfer and cooling modes;
  - Schmucker flow separation;
  - an expansion-ratio optimiser;
  - a noise estimate;
  - ascent profile;
  - engine comparison;
  - PDF report;
  - dual-bell nozzles;
  - CSV / DXF contour import;
  - Auto grid resolution;
  - undo / redo and `.aeronozzle` design files.
- **Propeller analysis** (blade-element momentum theory) and **flight dynamics** from the polar (trim, short-period and phugoid modes, gust and elevator responses), both in the *Simulation* menu.
- **CAD import:** STEP, IGES and BREP files.
- **OpenFOAM:** selectable turbulence models, plus surface and wake refinement.
- **Interface:** a light theme, an interface-size setting and an in-app guide.
- **Supersonic Lab:** a faster single-precision GPU mode.

See the [Changelog](CHANGELOG.md) for details.

## What's new in 1.6: Rocket Nozzle Lab

Open it with **Nozzle Lab** in the top bar or **View ▸ Rocket Nozzle Lab** (Ctrl+5). It simulates a rocket engine's nozzle and exhaust plume live, from ignition:

- **Everything is an input.** You can set:
  - the propellant: 9 presets, or your own γ, molar mass and chamber temperature;
  - the chamber pressure, or the mass flow rate instead;
  - the nozzle: throat, exit and chamber diameters, chamber length, convergent angle, a bell (Rao), conical or sonic divergent section with its angles, throat radii, wall thickness and a vehicle body;
  - the environment: ISA altitude or ambient pressure and temperature, and the flight Mach number.

  Eight engine presets range from a Merlin-class sea-level booster and an RS-25-class hydrolox stage to a vacuum upper stage, a solid motor, a cold-gas thruster and two laboratory jets.
- **Ignition, throttle, shut-down.** *Ignite* ramps the chamber pressure up, and you can watch the starting shock, the starting vortex ring and the plume forming. Changing the chamber pressure or the altitude while the engine fires throttles it or "climbs" live. *Shut down* ramps the chamber pressure down.
- **Shock diamonds and Mach disks** form from the physics. Views:
  - a synthetic **exhaust glow**, coloured by propellant; hot gas glows brightest behind shocks, as real plumes do;
  - Mach number, pressure relative to ambient, temperature, velocity and density;
  - schlieren and exhaust-gas fraction.

  Overlays: the sonic line, the detected Mach disks and shock cells, the theoretical shock-cell spacing and separation point, and streamlines. Hover for local values; zoom and pan with the mouse.
- **3-D view** (the *2-D section / 3-D view* switch): the solution revolved about the axis, live. *Volume* renders the plume as a glowing 3-D body inside a cut-away nozzle; *Cut-away* paints the field on the faces of a 90° notch, with a translucent plume-boundary surface. Camera presets and mouse orbit are included.
- **Detailed results**, simulation against 1-D rocket-nozzle theory:
  - **Performance:** thrust (the force on the chamber and nozzle, so it is correct during start-up too), the exit-plane and base-pressure contributions, Isp, CF, c*, mass flow at the throat and the exit, the exit Mach number, pressure, temperature and velocity, the expansion regime, Summerfield separation, shock-cell length, the Mach-disk position and the supersonic core length.
  - **Live charts:** thrust, chamber pressure and mass flow from ignition.
  - **Profiles:** along the centre line, across the exit plane and along the nozzle wall.
  - **Altitude:** thrust and Isp from sea level to 60 km, with the design altitude marked.
  - **Nozzle:** the contour to scale.

  Snapshots, MP4 video and CSV export are included.
- **Solver:** compressible Euler equations, axisymmetric (round nozzle) or planar (slot nozzle). The HLLC/HLL fluxes and MUSCL reconstruction are those of the Supersonic Lab. The nozzle walls are **cut cells**, sub-cell accurate and exactly conservative: a gas at rest stays exactly at rest. The injector is a non-reflecting characteristic inlet, and the far field draws entrained air from still air. It runs on the GPU or the CPU. On the GPU it uses single precision by default, about 6x faster on consumer cards and within about 0.1 % of double precision; *Double (exact)* matches the CPU to 1e-15.
- **Validation:**
  - **Theory:** the thrust coefficient matches the closed-form equation to 1e-9.
  - **Engine presets:** thrust is 95–98.6 % of theory and mass flow 96–100 %. The lower figures are on grids with only 6–8 cells across the throat; the Lab warns about that.
  - **Sonic jet:** the Mach disk sits at 2.89 exit diameters, against 3.0 from Ashkenas & Sherman.
  - **Kerolox plume:** the Mach-disk spacing is 3.7 De, against 3.4 De from the Prandtl–Pack shock-cell theory.
- **Limitations:**
  - One frozen perfect gas: the surrounding air uses the exhaust's γ and molar mass.
  - The flow is inviscid: no boundary layers, no viscous separation and no afterburning. The wall-pressure chart shows where a real nozzle would separate.
  - Real engines take 0.1–1 s to start; the ignition ramp is shortened so the start-up stays watchable.

## What's new in 1.5

The full list is in the [CHANGELOG](CHANGELOG.md).

- **Probe the flow with the mouse:** point readouts, line plots, and click-to-seed streamlines. **Oil-flow** streaks show the flow at the surface.
- **More slices and comparisons:** three-plane slices, an animated slice sweep, λ2 vortex cores, spanwise Cp sections, and a **side-by-side comparison** of two results with a ΔCp surface.
- **Engineering analysis:**
  - parametric studies (sideslip, speed/Reynolds/Mach, altitude);
  - stability (neutral point, static margin);
  - lifting-line theory overlays;
  - a grid convergence check (Richardson / GCI);
  - Kármán-Tsien compressibility, ground effect, flaps and ailerons.
- **OpenFOAM:** polar sweeps on one shared mesh, compressible air (automatic from Mach 0.3), transient runs with a frame player and Strouhal number, rotating propellers (MRF) with thrust, torque and efficiency, skin friction and y⁺ on the body, a checkMesh summary, a boundary-layer control and a live CL/CD chart.
- **Labs:** load airfoil `.dat` files, time-averaged and RMS fields, a supersonic inlet, GPU-only Extreme grids and MP4 recording.
- **Workflow:** `.aeroflow` project files, PDF reports, MP4/transparent/VTK exports with *Open in ParaView*, camera bookmarks and Imperial units.

## GPU acceleration (Live Flow Lab and Supersonic Lab)

- **Compute device: Auto / GPU / CPU.** Pick it in the *Simulation* card of either Lab or under **Simulation ▸ Compute device**. The choice is remembered, and the flow keeps running when you switch. *Auto* uses an NVIDIA GPU when one works and falls back to the CPU otherwise (also if a GPU kernel ever fails).
- **CUDA kernels via CuPy.** The lattice-Boltzmann and compressible-Euler solvers have CUDA ports. The Euler port runs in double precision and matches the CPU to round-off (about 1e-14); the LBM port runs in float32 like the CPU kernel. Batches of steps replay as CUDA graphs, and the GPU computes the next frame while the CPU draws the current one.
- **Measured on an RTX 2070 SUPER / 8-core CPU, in the real windows:** Live Flow Lab at 62 FPS on every grid with 9,400–82,000 steps/s on the GPU (CPU: 400–3,300). Supersonic Lab at 55–61 FPS on Fast/Balanced/Detailed with 800–2,500 steps/s (CPU: 60–330; without numba: about 15).
- **Setup:** `setup_windows.bat` installs the GPU libraries when `nvidia-smi` works. Manual install: `pip install -r requirements-gpu.txt`. No CUDA Toolkit is needed.
- **Overrides:** `AEROFLOW_DEVICE=cpu|gpu|auto` forces the device, and `AEROFLOW_NO_GPU=1` disables CUDA.
- OpenFOAM and the Instant Aero Engine stay on the CPU. OpenFOAM has no practical GPU build, and the Instant engine's double-precision n-body sums are no faster on consumer GPUs (fp64 runs at 1/32 rate). The OpenFOAM *CPU cores* setting now defaults to the physical cores minus one.

## What's new in 1.3

- **Supersonic Lab (Ctrl+4, separate window).** It solves the compressible Euler equations live:
  - HLLC/HLL fluxes, second-order MUSCL reconstruction and SSP Runge–Kutta time stepping, compiled with numba.
  - Immersed-boundary walls that follow the true body outline.
  - Mach 0.5–6, from subsonic and transonic through supersonic to hypersonic.
  - **Bodies:** wedge, diamond airfoil, flat plate, NACA airfoil, cylinder, blunt capsule, or a section of your 3-D model.
  - **Views:** Mach number, pressure, density, temperature, total-pressure loss, numerical schlieren and shadowgraph.
  - **Overlays:** the sonic line (M = 1), detected shocks, and the exact theoretical shock lines.
  - **Analysis tabs:**
    - *Theory vs simulation:* normal and oblique shocks, Prandtl–Meyer, isentropic and pitot values, linear airfoil theory.
    - *Line probe:* pressure and Mach number through the shocks.
    - *Surface pressure.*
    - *θ–β–M design chart:* with the current case marked.
  - **Validation:** every theory formula matches NACA Report 1135 tables. The solver reproduces:
    - oblique-shock pressure and angle within 1–2 %;
    - blunt-body stagnation pressure within ~3 %;
    - diamond-airfoil wave drag within 10 %.

    These checks run as automated tests.
- **Smoother 3-D view:**
  - **Faster result updates:** a new result recolours the existing surface in place instead of rebuilding it, so applying a result takes about 950 → 300 ms.
  - **Live preview:** dragging a flow slider updates CL/CD and the surface pressure live; the full flow field updates on release.
  - **Lighter visuals:** stream tubes skip a redundant smoothing pass, and the arrow glyphs use simpler shapes.
- **New visualisations:**
  - A GPU-rendered **vortex cloud** (vorticity volume rendering) in 3-D.
  - A flowing **line-integral-convolution "flow texture"** in the Live Flow Lab.
- **Run history** (Runs tab): every solve is logged in a sortable table. Double-click a run to restore it, or export the table as CSV.
- **Polar comparison:** pin up to four polars and overlay them as dashed reference curves.
- **Polish:** a start-up splash screen and non-modal toast notifications.

## What's new in 1.2

- **Live Flow Lab (Ctrl+3).** A transient, viscous 2-D solver runs in real time: a numba-compiled D2Q9 Lattice-Boltzmann scheme with a Smagorinsky sub-grid model, absorbing sponge layers and momentum-exchange forces.
  - **What you can change while the air flows:** the body (any NACA airfoil, cylinder, square, plate, ellipse, wedge, or the mid-span section of the loaded 3-D model), the angle of attack, the Reynolds number (40 to 100,000) and the grid resolution. You can also paint and erase walls with the mouse.
  - **What you see:** speed, vorticity, pressure or smoke streaks, with tracer particles, live CL/CD history and the vortex-shedding Strouhal number.
  - **Validation:** a cylinder at Re 100 gives St 0.18 and CD 1.7 (12.5 % tunnel blockage), consistent with published data.
- **New 3-D visualisations:**
  - **Isosurfaces** show vortex cores (Q-criterion: the trailing and wing-tip vortices), a speed shell, or the viscous wake (total-pressure loss).
  - **New colour fields:** slices and streamlines can now also be coloured by vorticity or total pressure.
  - **Off the GUI thread:** the derived fields are computed in the worker thread, so they never stall the UI.
- **New analysis charts:**
  - **Spanwise lift distribution** compared with the ideal elliptic loading.
  - **Pitching-moment curve (CM vs α)** in the polar views, which are now laid out as a 2×2 grid.

## What's new in 1.1

- **About 5–30× faster solves.** The volume field is evaluated by JIT-compiled, multi-threaded kernels (numba), with an automatic NumPy fallback. A MEDIUM solve now takes about 1 s instead of 6–25 s, on a grid with 3.4× more points.
- **Instant polar sweeps.** The source strengths are now precomputed for each geometry, so each angle of attack takes a few milliseconds.
- **The UI no longer freezes during solves.** The kernels release Python's GIL, so the GUI keeps running while the solver works in the background.
- **The flight simulator has been removed,** so the app is focused on CFD and starts faster, with one fewer OpenGL view.
- **Professional UI:** a menu bar (File, Edit, View, Simulation, Help), window and panel layout remembered between sessions, and closed panels can be re-opened from the View menu.
- **Better visuals:**
  - Smoother stream tubes.
  - Correctly ordered transparency.
  - One colour bar per layer.
  - Centred speed and Cp scales.
  - Flicker-free GIFs.
  - Animation at roughly 60 FPS: heavy still-image effects are paused while particles move.
- **Physics fixes:**
  - CM now shows the post-stall nose-down break.
  - Points ahead of the wing, such as a fuselage nose, no longer get leading-edge suction.
  - Induced drag can no longer grow post-stall.
  - The geometry cache now respects the quality setting.
- **About 40 bug fixes**, including:
  - Blank polar charts.
  - The clipped control panel.
  - Stacked error dialogs during live solves.
  - Esc breaking OpenFOAM detection.
  - Wrong drag-polar axis labels in reports.
- **A headless test suite** (`python -m pytest tests`).

---

## 1. Installation

AeroFlow Studio needs **64-bit Python 3.10–3.14** and an OpenGL 3.2+ capable GPU driver.

### Windows

**Easiest: the installer.** Download **`AeroFlowStudio-Setup-<version>.exe`** from the
[Releases page](https://github.com/albertmanoiu29-cell/aeroflow/releases) and run it. No administrator rights
are needed. It installs Python 3.12 if your PC has no suitable Python, runs `setup_windows.bat` to install
every library (plus NVIDIA GPU acceleration when an NVIDIA driver is present), and adds Start-menu and
desktop shortcuts to **`AeroFlow Studio.exe`**. Setup needs an internet connection and takes about 5–10
minutes. To repair a damaged install, use **AeroFlow Studio – repair libraries** from the Start menu.
Uninstall it from *Settings ▸ Apps*. Your cases and exports in `%APPDATA%\aeroflow_studio` are kept.

To build the installer yourself, install Inno Setup 6 (`winget install JRSoftware.InnoSetup`) and run
`powershell -ExecutionPolicy Bypass -File packaging\build_release.ps1`; the output goes to `dist\`.

**From the source folder** (requires 64-bit Python 3.10–3.14 from [python.org](https://www.python.org/downloads/windows/)):
double-click **`setup_windows.bat`** once. It finds a suitable Python, creates a private
`.venv` environment, installs every library into it and checks they import. Then start the app with
**`run_windows.bat`**.

**Manually (PowerShell)**, using the venv's own `python.exe` so the libraries and the app always use the
same Python (no `Activate.ps1`, so script-execution policies don't matter):

```powershell
cd aeroflow_studio
py -3.12 -m venv .venv          # or any installed 3.10–3.14: see `py --list`
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

### macOS / Linux

```bash
cd aeroflow_studio
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
python main.py
```

On Linux you may need the system OpenGL and XCB libraries:

```bash
sudo apt-get install -y libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 \
     libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0
```

To open a model directly at start-up: `python main.py path/to/model.stl`

---

## 2. Quick start

1. The app opens with a **NACA 2412 wing** that has already been solved. Rotate the view with the left mouse button, zoom with the wheel, and pan with Shift and drag.
2. Pick another preset, then press **Generate**. Alternatively, **drag an STL/OBJ/PLY/OFF/VTK file onto the 3-D view**.
3. Make sure the nose faces the wind. The wind blows in the +X direction, so the nose should point to −X. If it doesn't, use **Align nose → wind** or the 90° rotation buttons. CAD files in millimetres are detected automatically, and the app offers **mm → m**.
4. Move the **Flow conditions** sliders. With **Live update** on, the flow is solved again when you release a slider, which takes about a second.
5. Open **Visualization** to switch on animated smoke, the slice plane, arrows or the seed rake.
6. In **Polar Analysis**, press **Run polar sweep** to get the lift curve, drag polar and best L/D.
7. Export a report (Ctrl+E), the polar as CSV, a 4K screenshot or a GIF from the **File** menu.

### Keyboard

| Shortcut | Action | Shortcut | Action |
|---|---|---|---|
| Ctrl+O | Import geometry | Ctrl+R | Run simulation |
| Ctrl+Z | Undo geometry change | Esc | Abort |
| Ctrl+S | 4K screenshot | Ctrl+E | Export report |
| Ctrl+1 / 2 / 3 | Wind tunnel / Polar analysis / Live Flow Lab | F9 / F10 | Show or hide the left / right panel |
| F1 | Help | Ctrl+Q | Quit |

---

## 3. The two solver engines

### Engine A: Instant Aero Engine (built in, default)

NumPy/SciPy plus optional numba, with no external binaries. A warm solve takes about 0.7 s (Coarse), 1.2 s (Medium) or 4–9 s (Fine) on an 8-thread laptop CPU. Without numba it falls back to vectorised NumPy and is roughly 3–5× slower. Polar sweeps take a few milliseconds per angle.

1. **Surface discretisation.** Curvature-weighted samples of the mesh are clustered into about 700–1700 surface elements. This works on any triangle soup, including meshes that are not watertight.
2. **3-D source-panel potential flow.** A Neumann no-penetration system is LU-factorised once per geometry and then reused for every angle of attack. On a sphere it reproduces Cp_min = −1.19 against the exact −1.25.
3. **Lift.** Spanwise sections (chord, camber line, thickness) are extracted from the mesh and solved with a Weissinger vortex lattice. Zero-lift angle and section moment come from thin-airfoil theory applied to the extracted camber. Beyond the stall angle, lift blends smoothly to a flat-plate model.
   - An AR 6 NACA 0012 wing gives CL = 0.377 at 5°.
   - Induced drag follows CL²/(π·e·AR) with e ≈ 1.0.
4. **Drag.** Prandtl–Schlichting laminar/turbulent skin friction with a Hoerner form factor, plus induced drag, a leeward separation model for bluff bodies, and post-stall drag.
5. **Volume field.** The freestream, source and vortex-lattice velocity fields are superposed on a structured `pv.ImageData` grid by the compiled kernels in `core/accel.py`. The grid has about 75k, 175k or 580k points (Coarse, Medium, Fine) and extends 2.5 L upstream, 5 L downstream and 2.5 L to the sides. The body is located with a KD-tree signed-distance test, flow inside it is zero, flow tangency is enforced near the surface, and a momentum-conserving wake deficit is added.

> The Instant engine gives engineering estimates. It is accurate for attached subsonic flow (Mach < 0.6), approximate for massively separated or transonic flow, and it does not model Reynolds-dependent stall. Use OpenFOAM for final numbers.

### Engine B: OpenFOAM (optional, high fidelity)

At start-up AeroFlow looks for OpenFOAM **natively** (on `PATH`, `/opt/openfoam*`, `/usr/lib/openfoam/*` or `~/OpenFOAM/*`), then in **WSL2** on Windows, then as a **Docker** image (`openfoam/openfoam11-paraview510`, `…10…` or `…2312…`). If nothing is found, the top bar shows *"Instant Aero Engine · OpenFOAM not installed"*. Clicking it opens the Setup Guide, and the app carries on working normally.

When you press Run with OpenFOAM selected, AeroFlow:

1. Writes a complete case to `~/.aeroflow_studio/cases/<model>_<timestamp>/`:
   - the STL in `constant/triSurface` and `constant/geometry`;
   - `blockMeshDict` sized from the model;
   - `snappyHexMeshDict` with feature-edge refinement, a wake refinement box and boundary layers;
   - `controlDict` with a `forceCoeffs` function object (lift and drag directions follow α and β);
   - `fvSchemes` and `fvSolution` (SIMPLE plus PIMPLE dictionaries for Foundation v11+);
   - k-ω SST in both `momentumTransport` and `turbulenceProperties`;
   - `0/U p k omega nut` with inlet, outlet, freestream/slip tunnel walls and a no-slip body.
2. Runs `blockMesh` → `surfaceFeatures` / `surfaceFeatureExtract` → `snappyHexMesh -overwrite` → `simpleFoam` (or `foamRun -solver incompressibleFluid` on Foundation v11+) → `foamToVTK`, optionally in parallel with MPI.
3. Streams the residuals (Ux, Uy, Uz, p, k, ω) into the live convergence chart, and CL/CD into the status bar.
4. Loads the result into the same 3-D viewport, so streamlines, slices and the Cp heatmap all work as with the Instant engine.

**Abort** (or Esc) kills the whole process tree, including WSL and Docker children, and leaves no orphaned solver. Every stage writes `logs/log.<stage>` inside the case folder.

#### Connecting OpenFOAM

```bash
# Ubuntu / Debian (OpenFOAM Foundation 11)
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update && sudo apt-get install -y openfoam11
echo 'source /opt/openfoam11/etc/bashrc' >> ~/.bashrc

# or ESI / openfoam.com
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install -y openfoam2312-default
```

```powershell
# Windows: install WSL2 + Ubuntu (PowerShell as Administrator), then run the Ubuntu commands above inside WSL
wsl --install -d Ubuntu-22.04
```

```bash
# Any OS with Docker Desktop / Docker Engine running
docker pull openfoam/openfoam11-paraview510
```

Then restart AeroFlow Studio, or press **Re-detect** in the Setup Guide.

---

## 4. Architecture

```
aeroflow_studio/
├── main.py                    entry point (sets QT_API=pyside6 first, high-DPI, theme, logging)
├── requirements.txt
├── config/
│   ├── settings.py            enums, limits for every slider, ISA atmosphere, paths, OpenFOAM config
│   └── theme.py               palette, QSS stylesheet, vector icons, logo, generated assets
├── core/                      pure Python/NumPy — no Qt, testable headlessly
│   ├── results.py             shared result containers (both engines → same SolverResult)
│   ├── geometry_manager.py    import, repair, orient, scale, presets (NACA, delta, glider, sphere)
│   ├── instant_solver.py      Engine A
│   ├── accel.py               numba-compiled field kernels (source + horseshoe vortex) with NumPy fallback
│   ├── lbm2d.py               Live Flow Lab: real-time D2Q9 Lattice-Boltzmann solver, tracers, forces, LIC
│   ├── supersonic.py          Supersonic Lab: 2-D compressible Euler (HLLC, MUSCL, RK2, immersed boundary)
│   ├── nozzle.py              Rocket Nozzle Lab: 1-D nozzle theory, contours, cut-cell axisymmetric Euler solver
│   ├── nozzle_gpu.py          CUDA kernels of the nozzle solver (boundaries, cut-cell update)
│   ├── gasdynamics.py         exact shock / expansion / isentropic relations (validated against NACA 1135)
│   ├── shapes2d.py            2-D obstacles: NACA / bluff outlines, 3-D mesh sections, rasteriser
│   ├── openfoam_bridge.py     Engine B: detection, case generator, runner, VTK loader
│   └── polar_generator.py     AoA sweeps, CSV + HTML report
├── visualization/             PyVista/VTK and pyqtgraph (GUI thread only)
│   ├── cfd_viewport.py        wind-tunnel scene: streamlines, smoke, slices, isosurfaces, HUD, camera bar
│   ├── flowlab_view.py        Live Flow Lab canvas (field colouring, smoke, particles, wall painting)
│   ├── nozzle_view.py         Rocket Nozzle Lab canvas: mirrored side view, exhaust glow, overlays, probe
│   ├── nozzle_3d.py           Rocket Nozzle Lab 3-D view: revolved volume rendering and cut-away
│   └── telemetry_charts.py    residuals, chordwise Cp, span loading, polars, live forces
├── ui/
│   ├── main_window.py         layout, menus, persistence and orchestration
│   ├── cfd_panel.py           CFD / polar controls + shared widgets (accordion, sliders, toggles)
│   ├── flowlab_panel.py       Live Flow Lab controls
│   ├── supersonic_window.py   Supersonic Lab window: controls, theory table, probes, θ-β-M chart
│   ├── nozzle_window.py       Rocket Nozzle Lab window: inputs, presets, performance table, charts, export
│   ├── run_history.py         Runs tab (every solve, CSV export, re-apply)
│   ├── toast.py               non-modal notifications
│   ├── dialogs.py             OpenFOAM setup guide, report export, help/about
│   └── workers.py             QRunnable workers + TaskManager (cancellable, queued signals)
├── utils/
│   ├── logger.py              rotating log, Qt bridge, colour-coded console, exception hooks
│   └── validators.py          input clamping, safe maths, NACA parsing, mesh checks
└── tests/
    ├── test_core.py           headless regression tests (kernels, physics, polars, reports, span load)
    └── test_flowlab.py        Flow Lab tests (shapes, numba/NumPy agreement, vortex shedding, lift sign)
```

**Threading rules:**
- Loading, repair, solves, sweeps and OpenFOAM detection and runs all happen in `QThreadPool` workers.
- The compiled kernels release the GIL, and parallel kernel calls are serialised by a lock (numba's default threading layer must not be entered from two threads at once).
- Results reach the GUI only through queued Qt signals, via a GUI-thread relay object.
- Every VTK and widget call happens on the GUI thread.
- Actors have fixed names and are replaced in place, so repeated runs never leak VTK objects.
- `closeEvent` saves the window layout, stops all timers, cancels the workers, kills any OpenFOAM process tree and closes the `QtInteractor`.

Data lives in `~/.aeroflow_studio/`, with a temp-folder fallback. It holds `logs/`, `cases/`, `exports/` and `cache/`.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Black or blank 3-D view, or `wglMakeCurrent` / GLX errors | Update the GPU driver. In a VM or over remote desktop, use Mesa software OpenGL: `export LIBGL_ALWAYS_SOFTWARE=1` on Linux, or put Mesa's `opengl32.dll` next to `python.exe` on Windows |
| Linux Wayland: window opens but the 3-D view stays empty | AeroFlow already forces `QT_QPA_PLATFORM=xcb`; make sure XWayland is installed |
| KPI cards do not appear over the 3-D view | Some drivers draw native OpenGL windows above Qt overlays. Toggle full screen or resize the window once, and update the driver |
| Flow or forces look wrong (lift sideways, huge drag) | The nose is not at −X: use **Align nose → wind** |
| A side panel disappeared | **View → Controls panel / Analysis panel** (F9 / F10), or **View → Reset panel layout** |
| Solves are slower than the times above | Check the Log tab for "Field kernels ready: numba …". If it says NumPy, install numba (`pip install numba`). The first start compiles the kernels once (a few seconds) and caches them |
| The Labs say "GPU unavailable" | Hover over the *Compute* note for the reason. Usually CuPy or NVRTC is missing (`pip install -r requirements-gpu.txt`) or the NVIDIA driver is too old (CUDA 12 needs R525+). The Labs keep running on the CPU meanwhile |
| `snappyHexMesh` fails | Run **Repair mesh** (it needs a watertight surface), then try Coarse resolution |
| OpenFOAM solver diverges | Lower the speed or α, use Coarse, or reduce the iterations and inspect `logs/log.solver` |
| Import is slow or fails for huge CAD meshes | Decimate below about 500k triangles in your CAD tool, and export as binary STL |

Logs: **Help → Open log folder**. The **Log** tab can be saved with **Save…**.

---

## 6. Verification status

- **Live Flow Lab tests:** 9 further tests in `tests/test_flowlab.py` (included in the totals below) check:
  - shape rasterisation and the angle-of-attack sign;
  - mesh sections (NACA area within 8 %);
  - that the numba and NumPy steps agree;
  - that an empty tunnel stays uniform;
  - cylinder vortex shedding (St and CD in range);
  - that lift follows the angle of attack.

  A GUI run of the real app exercised every body, field, resolution and the wall painting, and logged no errors.
- **Supersonic tests:** 13 tests in `tests/test_supersonic.py` (included in the totals below) check:
  - the gas-dynamics tables;
  - exact preservation of uniform flow;
  - the wedge oblique shock;
  - cylinder stagnation pressure and stand-off;
  - diamond wave drag;
  - a stable, positive M = 6 blunt body;
  - no shocks in subsonic flow;
  - that the NumPy fallback runs.
- **Automated tests:** `python -m pytest -q tests` runs 44 headless tests. They pass with numba and with the NumPy fallback (`AEROFLOW_NO_NUMBA=1`). They cover:
  - compiled and NumPy kernels agreeing;
  - sphere Cp against the analytic potential-flow solution;
  - finite-wing lift slope, zero-lift angle and Oswald efficiency;
  - the post-stall CM break;
  - field consistency;
  - geometry cache behaviour;
  - the linear-basis shortcut against a direct solve;
  - polar sweeps and CSV/HTML export;
  - every built-in preset.
- **Regression against v1.0:** coefficients (CL, CD, CM) and velocity fields are bit-identical to the original solver at the same grid resolution.
- **GUI:** an automated run of the real application covered:
  - start-up solve;
  - live slider re-solve;
  - Esc abort;
  - streamlines, smoke, slice, arrows and Cp colouring;
  - polar sweep and polar screenshot;
  - help dialog and shutdown.

  It logged no errors.
- **Not yet run:** a real OpenFOAM installation. Report any issue with the log file attached.

OpenFOAM® is a registered trademark of OpenCFD Ltd. AeroFlow Studio is not affiliated with or endorsed by OpenCFD Ltd or the OpenFOAM Foundation.
