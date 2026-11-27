"""Point-mass ascent of a single-stage rocket powered by the Nozzle Lab engine (or a solid motor's thrust curve).

Two-dimensional flight over a flat Earth with gravity falling off as (R / (R + h))², the ISA atmosphere,
drag with a Mach-dependent drag coefficient, and thrust that follows the altitude (the nozzle's thrust at
each ambient pressure). The rocket holds its launch angle (rail, then guidance) up to a pitch-over speed, then flies a
gravity turn (thrust along the velocity). Returns the time histories and the usual flight summary (burnout, max-Q, apogee,
Δv budget with its gravity and drag losses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config.settings import ISA_MAX_ALTITUDE, isa_atmosphere

G0 = 9.80665
R_EARTH = 6.371e6


def atmosphere(h: float) -> tuple[float, float, float]:
    """(density kg/m³, pressure Pa, speed of sound m/s) at altitude ``h``; above the ISA table the density and
    pressure fall off with a 6.5 km scale height."""
    if h <= ISA_MAX_ALTITUDE:
        st = isa_atmosphere(max(h, 0.0))
        return st.density, st.pressure_pa, st.speed_of_sound
    st = isa_atmosphere(ISA_MAX_ALTITUDE)
    f = math.exp(-(h - ISA_MAX_ALTITUDE) / 6500.0)
    return st.density * f, st.pressure_pa * f, st.speed_of_sound


@dataclass(frozen=True)
class Vehicle:
    dry_mass: float                   # kg (structure, engine, payload)
    propellant_mass: float            # kg
    diameter: float                   # m (reference area π d² / 4)
    cd0: float = 0.35                 # subsonic drag coefficient
    engines: int = 1
    launch_angle: float = 89.0        # degrees above the horizon (held until the gravity turn starts)
    rail_length: float = 10.0         # m flown at the launch angle before the gravity turn
    turn_speed: float = 120.0         # m/s: the launch attitude is held (guided) until this speed, then a gravity
                                      # turn follows (thrust along the velocity). Very sensitive with a low T/W.


@dataclass
class Flight:
    t: np.ndarray
    altitude: np.ndarray              # m
    downrange: np.ndarray             # m
    speed: np.ndarray                 # m/s
    mach: np.ndarray
    q: np.ndarray                     # dynamic pressure, Pa
    accel_g: np.ndarray               # axial acceleration (thrust - drag) / (m g0)
    mass: np.ndarray
    thrust: np.ndarray
    burnout_time: float
    burnout_altitude: float
    burnout_speed: float
    max_q: float
    max_q_altitude: float
    max_accel_g: float
    apogee: float
    apogee_time: float
    downrange_at_apogee: float
    dv_ideal: float                   # ∫ F/m dt
    gravity_loss: float
    drag_loss: float
    liftoff_twr: float
    message: str


def drag_coefficient(cd0: float, mach: float) -> float:
    """A typical slender-rocket Cd(M): flat subsonic, a transonic rise to about 1.6 × at M 1.1, then a slow fall."""
    if mach < 0.8:
        f = 1.0
    elif mach < 1.1:
        f = 1.0 + 0.6 * ((mach - 0.8) / 0.3) ** 2
    else:
        f = max(1.6 - 0.25 * (mach - 1.1), 0.75)
    return cd0 * f


def fly(vehicle: Vehicle, thrust_at: Callable[[float, float], tuple[float, float]], burn_time_hint: Optional[float] = None,
        dt: float = 0.05, t_max: float = 3600.0) -> Flight:
    """Fly ``vehicle``. ``thrust_at(t, altitude)`` returns (thrust per engine N, mass flow per engine kg/s);
    the burn ends when the propellant is gone (or the thrust drops to zero)."""
    n = max(vehicle.engines, 1)
    area = 0.25 * math.pi * vehicle.diameter ** 2
    m0 = vehicle.dry_mass + vehicle.propellant_mass
    f0, _ = thrust_at(0.0, 0.0)
    twr = n * f0 / (m0 * G0)
    th = math.radians(min(max(vehicle.launch_angle, 1.0), 90.0))
    # state: x, h, vx, vh, propellant left
    x, h, vx, vh, mp = 0.0, 0.0, 0.0, 0.0, vehicle.propellant_mass
    t = 0.0
    out = []
    dv_ideal = g_loss = d_loss = 0.0
    burnout = None
    msg = ""

    def accel(t_, x_, h_, vx_, vh_, mp_):
        m = vehicle.dry_mass + max(mp_, 0.0)
        rho, _p, a_sound = atmosphere(h_)
        v = math.hypot(vx_, vh_)
        mach = v / a_sound
        f, md = thrust_at(t_, h_) if mp_ > 0.0 else (0.0, 0.0)
        f, md = n * f, n * md
        g = G0 * (R_EARTH / (R_EARTH + max(h_, 0.0))) ** 2
        if v > vehicle.turn_speed and math.hypot(x_, h_) > vehicle.rail_length:   # gravity turn once fast enough
            ux, uh = vx_ / v, vh_ / v
        else:
            ux, uh = math.cos(th), math.sin(th)
        qd = 0.5 * rho * v * v
        drag = qd * area * drag_coefficient(vehicle.cd0, mach)
        ax = (f - drag) * ux / m
        ah = (f - drag) * uh / m - g
        if h_ <= 0.0 and ah < 0.0 and v < 1e-6:            # on the pad
            ax = ah = 0.0
        return ax, ah, -md, f, drag, m, g, qd, mach, uh

    while t < t_max:
        ax, ah, dm, f, drag, m, g, qd, mach, uh = accel(t, x, h, vx, vh, mp)
        v = math.hypot(vx, vh)
        out.append((t, h, x, v, mach, qd, (f - drag) / (m * G0), m, f))
        if f > 0.0:
            dv_ideal += f / m * dt
            g_loss += g * uh * dt
        d_loss += drag / m * dt
        # Heun (RK2) step
        k1 = (vx, vh, ax, ah, dm)
        a2 = accel(t + dt, x + dt * k1[0], h + dt * k1[1], vx + dt * k1[2], vh + dt * k1[3], mp + dt * k1[4])
        k2 = (vx + dt * ax, vh + dt * ah, a2[0], a2[1], a2[2])
        x += 0.5 * dt * (k1[0] + k2[0])
        h += 0.5 * dt * (k1[1] + k2[1])
        vx += 0.5 * dt * (k1[2] + k2[2])
        vh += 0.5 * dt * (k1[3] + k2[3])
        mp_new = mp + 0.5 * dt * (k1[4] + k2[4])
        t += dt
        if burnout is not None and h > 100e3 and dt < 0.5:     # coasting in near-vacuum: larger steps
            dt = 0.5
        if mp > 0.0 and (mp_new <= 0.0 or f <= 0.0) and burnout is None:
            burnout = (t, h, math.hypot(vx, vh))
        mp = 0.0 if burnout is not None else max(mp_new, 0.0)
        if h < 0.0 and t > 1.0:
            if burnout is None:
                msg = ("The rocket hit the ground under power: the gravity turn tipped it over. Launch closer to "
                       "vertical (real boosters pitch over by only 1-2°) or raise the thrust-to-weight ratio.")
            break
        if burnout is not None and vh < 0.0 and h < out[-1][1] and len(out) > 2 and t > burnout[0] + 1.0:
            # past apogee: keep a short descent for the plot, then stop
            if h < 0.5 * max(o[1] for o in out):
                break
    a = np.asarray(out)
    k_ap = int(np.argmax(a[:, 1]))
    k_q = int(np.argmax(a[:, 5]))
    bo = burnout or (float(a[-1, 0]), float(a[-1, 1]), float(a[-1, 3]))
    lifted = np.nonzero(a[:, 1] > 0.1)[0]
    if twr <= 1.0 and lifted.size and not msg.startswith("The rocket hit"):
        msg = (f"Lift-off thrust-to-weight {twr:.2f} (< 1): the rocket sat on the pad for {a[lifted[0], 0]:.0f} s "
               "until it had burned enough propellant to lift off.")
    elif twr <= 1.0 and not lifted.size:
        msg = f"Lift-off thrust-to-weight {twr:.2f}: the rocket cannot leave the pad (needs > 1)."
    if not msg:
        msg = "Reached apogee." if k_ap < len(a) - 1 else "Still climbing at the end of the simulation."
    return Flight(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7], a[:, 8], bo[0], bo[1], bo[2],
                  float(a[k_q, 5]), float(a[k_q, 1]), float(a[:, 6].max()), float(a[k_ap, 1]), float(a[k_ap, 0]),
                  float(a[k_ap, 2]), dv_ideal, g_loss, d_loss, twr, msg)


def engine_thrust_table(theory_at: Callable[[float], tuple[float, float]], top_km: float = 100.0,
                        n: int = 101) -> Callable[[float, float], tuple[float, float]]:
    """A fast ``thrust_at(t, h)`` from a function of ambient pressure: tabulated over altitude once."""
    hs = np.linspace(0.0, top_km * 1e3, n)
    vals = np.array([theory_at(atmosphere(float(hh))[1]) for hh in hs])

    def thrust_at(_t: float, h: float) -> tuple[float, float]:
        return float(np.interp(h, hs, vals[:, 0])), float(np.interp(h, hs, vals[:, 1]))

    return thrust_at


def ideal_delta_v(isp: float, m0: float, mf: float) -> float:
    """Tsiolkovsky: Δv = Isp g0 ln(m0 / mf)."""
    return isp * G0 * math.log(m0 / mf) if mf > 0 and m0 > mf else 0.0


__all__ = ["Vehicle", "Flight", "drag_coefficient", "fly", "engine_thrust_table", "ideal_delta_v"]
