"""In-app guide: what each part of the program does, how to read what you see, and the physics behind it."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QListWidget, QTextBrowser, QWidget

from config.settings import APP_NAME

TOPICS: dict[str, str] = {
    "Getting started": """
<h2>Getting started</h2>
<p>AeroFlow Studio has a <b>wind tunnel</b> for 3-D models (load an STL/OBJ or build a preset wing, set the speed and
angle, press <b>Run</b>), a <b>polar analysis</b> that sweeps the angle of attack, and four live labs:</p>
<ul>
<li><b>Live Flow Lab</b> (Ctrl+3): a real-time 2-D wind tunnel - vortex streets, wakes and stall.</li>
<li><b>Supersonic Lab</b> (Ctrl+4): shock waves, expansion fans and bow shocks around bodies above Mach 1.</li>
<li><b>Rocket Nozzle Lab</b> (Ctrl+5): a rocket engine firing - shock diamonds, thrust, Isp, heat, noise.</li>
</ul>
<p>Two solver engines: the <b>Instant Aero Engine</b> (built in, seconds) and <b>OpenFOAM</b> (optional, minutes,
full Navier-Stokes). Results always come with the theory to compare against.</p>""",
    "Rocket Nozzle Lab: first run": """
<h2>Rocket Nozzle Lab: first run</h2>
<ol>
<li>Pick a <b>preset</b> engine (or keep the kerolox booster) and press <b>Ignite</b>.</li>
<li>Watch the start-up: the starting shock leaves the nozzle, a vortex ring rolls up, then the
<b>shock diamonds</b> settle in the plume.</li>
<li>When the state reads <b>FIRING · steady</b>, compare the simulated thrust and Isp with 1-D theory on the
<b>Performance</b> tab.</li>
<li>Change anything - chamber pressure (throttles live), altitude (the plume widens as the air thins), the nozzle,
the propellant - and watch the response.</li>
</ol>
<p><b>3-D view</b> revolves the solution: a glowing plume, tracer particles, or a cut-away.
<b>Fly an ascent</b> climbs through the atmosphere while the engine fires.</p>""",
    "Shock diamonds and Mach disks": """
<h2>Shock diamonds and Mach disks</h2>
<p>A nozzle exit pressure rarely equals the ambient pressure. <b>Overexpanded</b> (exit below ambient): oblique shocks
from the lip squeeze the jet; they cross on the axis and reflect from the jet boundary as expansion fans, and the
pattern repeats - a chain of <b>shock cells</b>. <b>Underexpanded</b> (exit above ambient): expansion fans at the lip
let the jet bulge out first, then reflect as shocks.</p>
<p>Behind each shock the gas is compressed and re-heated, so it glows brighter - the <b>diamonds</b> seen in rocket
plumes. When a shock is too strong to reflect regularly on the axis, a flat normal shock forms: the <b>Mach disk</b>
(white bars in the 2-D view).</p>
<p>The cell length follows Prandtl's vortex-sheet model, L ≈ 1.3 √(Mⱼ² − 1) Dⱼ (amber circles); for a sonic
orifice the first Mach disk sits 0.67 √(p₀/pₐ) diameters downstream (Ashkenas &amp; Sherman).</p>""",
    "Thrust, Isp and expansion": """
<h2>Thrust, Isp and expansion</h2>
<p><b>Thrust</b> F = ṁ vₑ + (pₑ − pₐ) Aₑ: momentum plus a pressure term. The simulation measures it as the force of
the gas on the chamber and nozzle walls (correct during start-up too); the exit-plane value is shown as a check.</p>
<p><b>Specific impulse</b> Isp = F / (ṁ g₀): seconds of thrust per weight of propellant - the engine's efficiency.
<b>c*</b> = pc Aₜ / ṁ measures the combustion; <b>CF</b> = F / (pc Aₜ) the nozzle.</p>
<p>A nozzle gives most thrust when pₑ = pₐ (<b>ideally expanded</b>): the design altitude. Below it the nozzle is too
long (overexpanded, and at sea level it may <b>separate</b> from the wall); above it too short. The <b>Design</b> tab
finds the expansion ratio that is best over a whole ascent.</p>""",
    "Heat, cooling and noise": """
<h2>Heat, cooling and noise</h2>
<p>The <b>Heat</b> tab uses the Bartz correlation: heating peaks at the throat (tens of MW/m² in big engines).
A <b>regeneratively cooled</b> wall is held cool by propellant flowing behind it; a <b>radiation-cooled</b> extension
runs hot and glows, shedding heat as εσT⁴ - the 3-D view colours the wall by this temperature.</p>
<p>The <b>Noise</b> tab estimates the sound power as ~0.5 % of the jet power (NASA SP-8072): about 140 dB at 100 m for
a booster engine, loudest ~45° off the exhaust axis, with most energy at low frequencies.</p>""",
    "Reading the colours": """
<h2>Reading the colours</h2>
<ul>
<li><b>Exhaust glow</b>: what a camera sees - hot gas, brightest behind shocks; afterburning where fuel-rich exhaust
mixes with air.</li>
<li><b>Mach number</b>: the white line is Mach 1 (the sonic line).</li>
<li><b>Pressure / ambient</b> (log): blue below ambient, red above - the alternating pattern is the shock-cell
structure.</li>
<li><b>Schlieren</b>: density gradients, as in an optical schlieren photo - shocks are dark lines.</li>
</ul>
<p>Hover anywhere in the 2-D view for local Mach number, pressure, temperature, speed and density.</p>""",
    "What the models can and cannot do": """
<h2>What the models can and cannot do</h2>
<ul>
<li>The labs solve the <b>inviscid</b> (Euler) or lattice-Boltzmann equations: shocks, expansions and vortices are
real physics; boundary layers, viscous separation and turbulence models are not included.</li>
<li>The nozzle uses one frozen perfect gas (the surrounding air takes the exhaust's γ and molar mass) and approximate
equilibrium chemistry tables.</li>
<li>Heat transfer (Bartz) and noise are engineering correlations: good to tens of per cent / a few dB.</li>
<li>For final numbers use OpenFOAM (wind tunnel) and dedicated rocket tools (CEA, RPA, CFD with chemistry).</li>
</ul>""",
    "Supersonic Lab": """
<h2>Supersonic Lab</h2>
<p>Above Mach 1 disturbances cannot travel upstream, so the air meets a body abruptly: a sharp nose carries an
attached <b>oblique shock</b> whose angle follows the θ-β-M relation; a blunt body pushes a detached <b>bow shock</b>
ahead of it; convex corners turn the flow through <b>Prandtl-Meyer expansion fans</b>.</p>
<p>The right panel compares the simulated shock angles, surface pressures and stand-off distance with exact
gas-dynamics theory.</p>""",
    "Live Flow Lab": """
<h2>Live Flow Lab</h2>
<p>A real-time 2-D wind tunnel (lattice-Boltzmann). Behind a cylinder above Re ≈ 47 the wake sheds a <b>von Kármán
vortex street</b> at a Strouhal number near 0.2; an airfoil at high angle of attack <b>stalls</b> as the flow
separates. Paint walls with the mouse to build your own obstacles.</p>""",
    "Keyboard shortcuts": """
<h2>Keyboard shortcuts</h2>
<table cellpadding='4'>
<tr><td>Ctrl+1 / 2 / 3</td><td>Wind tunnel / Polar analysis / Live Flow Lab</td></tr>
<tr><td>Ctrl+4 / Ctrl+5</td><td>Supersonic Lab / Rocket Nozzle Lab</td></tr>
<tr><td>F1</td><td>Help · in the Nozzle Lab: this guide</td></tr>
<tr><td>Ctrl+Z / Ctrl+Y</td><td>Nozzle Lab: undo / redo a change</td></tr>
<tr><td>Ctrl+S / Ctrl+O</td><td>Nozzle Lab: save / open an engine design</td></tr>
</table>""",
}


class GuideDialog(QDialog):
    """Topic list on the left, the explanation on the right."""

    def __init__(self, start: str = "Getting started", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Guide — {APP_NAME}")
        self.resize(900, 600)
        self.list = QListWidget()
        self.list.setMaximumWidth(260)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        for name in TOPICS:
            self.list.addItem(name)
        self.list.currentTextChanged.connect(lambda t: self.text.setHtml(TOPICS.get(t, "")))
        lay = QHBoxLayout(self)
        lay.addWidget(self.list)
        lay.addWidget(self.text, 1)
        items = self.list.findItems(start, Qt.MatchFlag.MatchExactly)
        self.list.setCurrentItem(items[0] if items else self.list.item(0))


def show_guide(parent: Optional[QWidget] = None, topic: str = "Getting started") -> None:
    dlg = GuideDialog(topic, parent)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.show()
