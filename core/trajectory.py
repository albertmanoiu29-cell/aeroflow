"""Point-mass ascent of a single-stage rocket powered by the Nozzle Lab engine (or a solid motor's thrust curve).

Two-dimensional flight over a round, non-rotating Earth (inverse-square gravity), the ISA atmosphere,
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


@dataclass(frozen=True)
class UpperStage:
    """A second stage: it separates when the first stage's propellant is gone and ignites ``coast`` s later."""

    dry_mass: float
    propellant_mass: float
    thrust_at: Callable[[float, float], tuple[float, float]]     # (time since ignition, altitude) per engine
    engines: int = 1
    coast: float = 3.0
    horizontal: bool = True      # steer the thrust along the local horizontal (build orbital speed), not the velocity


MU_EARTH = 3.986004418e14


def orbit_from_state(h: float, v_horizontal: float, v_vertical: float) -> Optional[tuple[float, float]]:
    """(perigee, apogee) altitudes in m of the Keplerian orbit through this state, or None if it is not
    closed or the perigee is inside the atmosphere (below 100 km)."""
    r = R_EARTH + h
    energy = 0.5 * (v_horizontal ** 2 + v_vertical ** 2) - MU_EARTH / r
    if energy >= 0.0:
        return None
    hm = r * v_horizontal
    a = -MU_EARTH / (2.0 * energy)
    e = math.sqrt(max(1.0 + 2.0 * energy * hm * hm / MU_EARTH ** 2, 0.0))
    rp, ra = a * (1.0 - e) - R_EARTH, a * (1.0 + e) - R_EARTH
    return (rp, ra) if rp > 100e3 else None


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
    staging: Optional[tuple[float, float, float]] = None     # (t, altitude, speed) at stage separation
    orbit: Optional[tuple[float, float]] = None               # (perigee, apogee) altitude, m


def drag_coefficient(cd0: float, mach: float) -> float:
    """A typical slender-rocket Cd(M): flat subsonic, a transonic rise to about 1.6 × at M 1.1, then a slow fall."""
    if mach < 0.8:
        f = 1.0
    elif mach < 1.1:
        f = 1.0 + 0.6 * ((mach - 0.8) / 0.3) ** 2
    else:
        f = max(1.6 - 0.25 * (mach - 1.1), 0.75)
    return cd0 * f


def fly(vehicle: Vehicle, thrust_at: Callable[[float, float], tuple[float, float]],
        upper: Optional[UpperStage] = None, dt: float = 0.05, t_max: float = 3600.0) -> Flight:
    """Fly ``vehicle`` over a round, non-rotating Earth. ``thrust_at(t, altitude)`` returns (thrust per engine N,
    mass flow per engine kg/s); the burn ends when the propellant is gone (or the thrust drops to zero). With an
    ``upper`` stage the first stage's dry mass is dropped at burnout and the upper stage fires after a short
    coast. If the final burnout state is a closed orbit with a perigee above 100 km, the flight stops there and
    reports the orbit."""
    n = max(vehicle.engines, 1)
    area = 0.25 * math.pi * vehicle.diameter ** 2
    up_mass = upper.dry_mass + upper.propellant_mass if upper is not None else 0.0
    m0 = vehicle.dry_mass + vehicle.propellant_mass + up_mass
    f0, _ = thrust_at(0.0, 0.0)
    twr = n * f0 / (m0 * G0)
    th = math.radians(min(max(vehicle.launch_angle, 1.0), 90.0))
    # state in an Earth-centred plane: position (X, Y) with the launch site at (0, R), velocity, propellant left
    X, Y, VX, VY, mp = 0.0, R_EARTH, 0.0, 0.0, vehicle.propellant_mass
    t = 0.0
    out = []
    dv_ideal = g_loss = d_loss = 0.0
    burnout = None
    staging = None
    stage = 0
    t_ign = 0.0
    msg = ""

    def local(x_, y_):
        r = math.hypot(x_, y_)
        ux, uy = x_ / r, y_ / r                    # up
        return r, ux, uy, uy, -ux                  # r, up (x, y), east / downrange (x, y)

    def accel(t_, x_, y_, vx_, vy_, mp_):
        if stage == 0:
            m = vehicle.dry_mass + max(mp_, 0.0) + up_mass
            f, md = thrust_at(t_, h_of(x_, y_)) if mp_ > 0.0 else (0.0, 0.0)
            f, md = n * f, n * md
        else:
            m = upper.dry_mass + max(mp_, 0.0)
            f, md = upper.thrust_at(t_ - t_ign, h_of(x_, y_)) if mp_ > 0.0 and t_ >= t_ign else (0.0, 0.0)
            f, md = upper.engines * f, upper.engines * md
        r, upx, upy, ex, ey = local(x_, y_)
        h_ = r - R_EARTH
        rho, _p, a_sound = atmosphere(h_)
        v = math.hypot(vx_, vy_)
        mach = v / a_sound
        g = MU_EARTH / (r * r)
        dist = R_EARTH * math.atan2(x_, y_)
        if stage == 1 and upper.horizontal and f > 0.0:
            # upper-stage guidance: build horizontal speed, pitching up just enough to hold the altitude
            # (lift gravity minus the centrifugal relief, and damp any sink or climb)
            vr = vx_ * upx + vy_ * upy
            vt = vx_ * ex + vy_ * ey
            need = g - vt * vt / r - 0.04 * vr
            s_up = min(max(need * m / f, -0.3), 0.95)
            c_h = math.sqrt(1.0 - s_up * s_up)
            dx, dy = c_h * ex + s_up * upx, c_h * ey + s_up * upy
        elif v > vehicle.turn_speed and math.hypot(dist, h_) > vehicle.rail_length:  # gravity turn once fast enough
            dx, dy = vx_ / v, vy_ / v
        else:                                                                        # hold the launch attitude
            dx, dy = math.cos(th) * ex + math.sin(th) * upx, math.cos(th) * ey + math.sin(th) * upy
        qd = 0.5 * rho * v * v
        drag = qd * area * drag_coefficient(vehicle.cd0, mach)
        if v > 1e-9:                                            # drag opposes the velocity
            dgx, dgy = -drag * vx_ / v, -drag * vy_ / v
        else:
            dgx = dgy = 0.0
        ax = (f * dx + dgx) / m - g * upx
        ay = (f * dy + dgy) / m - g * upy
        if h_ <= 0.0 and v < 1e-6 and (ax * upx + ay * upy) < 0.0:      # still on the pad
            ax = ay = 0.0
        up_frac = dx * upx + dy * upy
        return ax, ay, -md, f, drag, m, g, qd, mach, up_frac

    def h_of(x_, y_):
        return math.hypot(x_, y_) - R_EARTH

    def record(f, drag, m, qd, mach):
        r, upx, upy, ex, ey = local(X, Y)
        out.append((t, r - R_EARTH, R_EARTH * math.atan2(X, Y), math.hypot(VX, VY), mach, qd,
                    (f - drag) / (m * G0), m, f))

    def flight_from(a, bo, k_ap, msg_, orbit=None):
        k_q = int(np.argmax(a[:, 5]))
        return Flight(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7], a[:, 8], bo[0], bo[1],
                      bo[2], float(a[k_q, 5]), float(a[k_q, 1]), float(a[:, 6].max()),
                      orbit[1] if orbit else float(a[k_ap, 1]), float(a[k_ap, 0]), float(a[k_ap, 2]), dv_ideal,
                      g_loss, d_loss, twr, msg_, staging, orbit)

    while t < t_max:
        ax, ay, dm, f, drag, m, g, qd, mach, upf = accel(t, X, Y, VX, VY, mp)
        record(f, drag, m, qd, mach)
        if f > 0.0:
            dv_ideal += f / m * dt
            g_loss += g * upf * dt
        d_loss += drag / m * dt
        # Heun (RK2) step
        a2 = accel(t + dt, X + dt * VX, Y + dt * VY, VX + dt * ax, VY + dt * ay, mp + dt * dm)
        X += 0.5 * dt * (VX + VX + dt * ax)
        Y += 0.5 * dt * (VY + VY + dt * ay)
        VX += 0.5 * dt * (ax + a2[0])
        VY += 0.5 * dt * (ay + a2[1])
        mp_new = mp + 0.5 * dt * (dm + a2[2])
        t += dt
        h = h_of(X, Y)
        if burnout is not None and h > 100e3 and dt < 0.5:     # coasting in near-vacuum: larger steps
            dt = 0.5
        burning = stage == 0 or t > t_ign + dt
        if mp > 0.0 and (mp_new <= 0.0 or (f <= 0.0 and burning)) and burnout is None:
            if stage == 0 and upper is not None:           # separation: drop the first stage, coast, ignite
                staging = (t, h, math.hypot(VX, VY))
                stage, t_ign = 1, t + upper.coast
                mp_new = upper.propellant_mass
            else:
                burnout = (t, h, math.hypot(VX, VY))
                _r, upx, upy, ex, ey = local(X, Y)
                orbit = orbit_from_state(h, VX * ex + VY * ey, VX * upx + VY * upy)
                if orbit is not None:
                    record(0.0, 0.0, upper.dry_mass if upper else vehicle.dry_mass, 0.0, 0.0)
                    a = np.asarray(out)
                    return flight_from(a, burnout, int(np.argmax(a[:, 1])),
                                       f"In orbit: perigee {orbit[0] / 1e3:,.0f} km, apogee {orbit[1] / 1e3:,.0f} km "
                                       "(Keplerian orbit from the burnout state).", orbit)
        mp = 0.0 if burnout is not None else max(mp_new, 0.0)
        if h < 0.0 and t > 1.0:
            if burnout is None:
                msg = ("The rocket hit the ground under power: the gravity turn tipped it over. Launch closer to "
                       "vertical (real boosters pitch over by only 1-2°) or raise the thrust-to-weight ratio.")
            break
        if burnout is not None and t > burnout[0] + 1.0 and len(out) > 2 and h < out[-1][1]:
            # past apogee: keep a short descent for the plot, then stop
            if h < 0.5 * max(o[1] for o in out):
                break
    a = np.asarray(out)
    k_ap = int(np.argmax(a[:, 1]))
    bo = burnout or (float(a[-1, 0]), float(a[-1, 1]), float(a[-1, 3]))
    lifted = np.nonzero(a[:, 1] > 0.1)[0]
    if twr <= 1.0 and lifted.size and not msg.startswith("The rocket hit"):
        msg = (f"Lift-off thrust-to-weight {twr:.2f} (< 1): the rocket sat on the pad for {a[lifted[0], 0]:.0f} s "
               "until it had burned enough propellant to lift off.")
    elif twr <= 1.0 and not lifted.size:
        msg = f"Lift-off thrust-to-weight {twr:.2f}: the rocket cannot leave the pad (needs > 1)."
    if not msg:
        msg = ("Reached apogee." if k_ap < len(a) - 1 else
               f"Still climbing after {t_max / 60:.0f} minutes: a very high ballistic arc (or an escape trajectory).")
    return flight_from(a, bo, k_ap, msg)


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


__all__ = ["Vehicle", "UpperStage", "orbit_from_state", "Flight", "drag_coefficient", "fly", "engine_thrust_table", "ideal_delta_v"]
