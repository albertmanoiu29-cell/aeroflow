"""Longitudinal flight dynamics from a computed polar: trim, the short-period and phugoid modes, and a time
simulation of the response to an elevator pulse or a gust.

Aerodynamics come from the polar (CL, CD, Cm against α, interpolated; Cm is moved to the chosen CG), plus
two inputs the polar does not contain: the pitch-control power Cm_δe (elevator effectiveness) and the pitch
damping Cm_q (mostly from the tail). Thrust balances the trim drag and is then held constant. The equations
are the classical nonlinear point-mass + pitch equations in flight-path form:

    m dV/dt     = T cos α − D − W sin γ
    m V dγ/dt   = T sin α + L − W cos γ
    I_yy dq/dt  = q̄ S c [Cm(α) + Cm_δe δe + Cm_q q c / (2V)]
    dα/dt       = q − dγ/dt,   dh/dt = V sin γ

Mode frequencies and damping come from the eigenvalues of the numerically linearised system at trim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


import numpy as np

from config.settings import isa_atmosphere
from core.results import PolarData

G = 9.80665


@dataclass(frozen=True)
class Aircraft:
    mass: float                      # kg
    radius_of_gyration: float        # m (pitch): I_yy = m k²
    cg_shift: float = 0.0            # CG behind the polar's moment reference, fraction of the reference chord
    cm_delta_e: float = -0.02        # per degree of elevator (negative: trailing edge down pitches nose down)
    cm_q: float = -15.0              # pitch damping, per radian of q c / (2V)

    @property
    def iyy(self) -> float:
        return self.mass * self.radius_of_gyration ** 2


@dataclass
class Trim:
    ok: bool
    message: str
    speed: float = 0.0               # m/s
    alpha_deg: float = 0.0
    elevator_deg: float = 0.0
    cl: float = 0.0
    cd: float = 0.0
    thrust: float = 0.0              # N (= drag)
    rho: float = 1.225


@dataclass
class Mode:
    name: str
    eigenvalue: complex
    omega_n: float                   # rad/s
    damping: float
    period: float                    # s (inf if not oscillatory)
    half_or_double_time: float       # s: time to halve (stable) or double (unstable) the amplitude


class FlightModel:
    """Polar-based longitudinal dynamics of one aircraft at one altitude."""

    def __init__(self, polar: PolarData, aircraft: Aircraft, altitude_m: float = 0.0) -> None:
        self.p = polar
        self.ac = aircraft
        self.rho = isa_atmosphere(altitude_m).density
        self.S = max(polar.ref_area_m2, 1e-9)
        self.c = max(polar.ref_length_m, 1e-9)
        a = np.asarray(polar.alpha_deg, float)
        self._a = np.radians(a)
        self._cl = np.asarray(polar.cl, float)
        self._cd = np.asarray(polar.cd, float)
        self._cm = np.asarray(polar.cm, float) + self._cl * aircraft.cg_shift      # moments about the CG

    def coeffs(self, alpha: float) -> tuple[float, float, float]:
        """CL, CD, Cm at α (rad); linear extrapolation a little beyond the polar's range."""
        a = self._a
        if alpha <= a[0] or alpha >= a[-1]:
            k = (0, 1) if alpha <= a[0] else (-2, -1)
            t = (alpha - a[k[0]]) / (a[k[1]] - a[k[0]])
            return tuple(float(v[k[0]] + t * (v[k[1]] - v[k[0]])) for v in (self._cl, self._cd, self._cm))
        return (float(np.interp(alpha, a, self._cl)), float(np.interp(alpha, a, self._cd)),
                float(np.interp(alpha, a, self._cm)))

    # -- trim --------------------------------------------------------------------------------
    def trim(self, speed: float) -> Trim:
        """Level flight at ``speed``: the α giving lift = weight, the elevator that zeroes the pitching moment."""
        W = self.ac.mass * G
        q = 0.5 * self.rho * speed * speed
        cl_req = W / (q * self.S)
        k = int(np.argmax(self._cl))
        if cl_req > self._cl[k]:
            v_stall = math.sqrt(2.0 * W / (self.rho * self.S * self._cl[k]))
            return Trim(False, f"Too slow: level flight needs CL {cl_req:.2f}, above CL,max {self._cl[k]:.2f} "
                               f"(stall speed {v_stall:.1f} m/s).", rho=self.rho)
        a_lin, cl_lin = self._a[:k + 1], self._cl[:k + 1]
        if cl_req < cl_lin.min():
            return Trim(False, "Too fast for this polar's angle range (CL below its lowest value).", rho=self.rho)
        alpha = float(np.interp(cl_req, cl_lin, a_lin))
        cl, cd, cm = self.coeffs(alpha)
        de = -cm / self.ac.cm_delta_e if abs(self.ac.cm_delta_e) > 1e-12 else 0.0
        msg = "Trimmed for level flight."
        if abs(de) > 25.0:
            msg = f"Needs {de:+.0f}° of elevator: beyond a real elevator's range - move the CG or change the speed."
        return Trim(abs(de) <= 25.0, msg, speed, math.degrees(alpha), de, cl, cd, cd * q * self.S, self.rho)

    # -- dynamics -----------------------------------------------------------------------------
    def derivatives(self, s: np.ndarray, thrust: float, elevator_deg: float) -> np.ndarray:
        """d/dt of (V, γ, α, q, h, x)."""
        v, gam, alpha, qr = s[0], s[1], s[2], s[3]
        v = max(v, 1.0)
        cl, cd, cm = self.coeffs(alpha)
        qd = 0.5 * self.rho * v * v
        L, D = qd * self.S * cl, qd * self.S * cd
        W = self.ac.mass * G
        m = self.ac.mass
        dv = (thrust * math.cos(alpha) - D - W * math.sin(gam)) / m
        dg = (thrust * math.sin(alpha) + L - W * math.cos(gam)) / (m * v)
        cm_t = cm + self.ac.cm_delta_e * elevator_deg + self.ac.cm_q * qr * self.c / (2.0 * v)
        dq = qd * self.S * self.c * cm_t / self.ac.iyy
        return np.array([dv, dg, qr - dg, dq, v * math.sin(gam), v * math.cos(gam)])

    def modes(self, tr: Trim) -> list[Mode]:
        """Short-period and phugoid from the eigenvalues of the linearised (V, γ, α, q) system at trim."""
        x0 = np.array([tr.speed, 0.0, math.radians(tr.alpha_deg), 0.0, 0.0, 0.0])
        n = 4
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 * max(abs(x0[j]), 1.0)
            xp, xm = x0.copy(), x0.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (self.derivatives(xp, tr.thrust, tr.elevator_deg)[:n]
                       - self.derivatives(xm, tr.thrust, tr.elevator_deg)[:n]) / (2.0 * h)
        lam = np.linalg.eigvals(J)
        osc = sorted([z for z in lam if z.imag > 1e-9], key=lambda z: abs(z))
        out: list[Mode] = []
        names = ["Phugoid", "Short period"] if len(osc) >= 2 else ["Oscillation"]
        for name, z in zip(names, osc):
            wn = abs(z)
            zeta = -z.real / wn if wn > 0 else 0.0
            t_half = math.log(2.0) / abs(z.real) if abs(z.real) > 1e-12 else math.inf
            out.append(Mode(name, complex(z), wn, zeta, 2.0 * math.pi / z.imag, t_half))
        if len(osc) < 2:                                   # a real pair (e.g. a heavily damped short period)
            for z in sorted([z for z in lam if abs(z.imag) <= 1e-9], key=lambda z: -abs(z.real)):
                t = math.log(2.0) / abs(z.real) if abs(z.real) > 1e-12 else math.inf
                out.append(Mode("Aperiodic (real root)", complex(z), abs(z.real), 1.0 if z.real < 0 else -1.0, math.inf, t))
        return out

    def simulate(self, tr: Trim, duration: float = 60.0, dt: float = 0.01, kind: str = "elevator",
                 size: float = 2.0, pulse: float = 1.0) -> dict[str, np.ndarray]:
        """Response from trim to an elevator pulse (``size`` degrees for ``pulse`` seconds) or a vertical gust
        (an instant ``size``-degree change of angle of attack). Returns time histories."""
        s = np.array([tr.speed, 0.0, math.radians(tr.alpha_deg), 0.0, 0.0, 0.0])
        if kind == "gust":
            s[2] += math.radians(size)
        n = int(duration / dt) + 1
        out = np.zeros((n, 6))
        t = np.arange(n) * dt
        for k in range(n):
            out[k] = s
            de = tr.elevator_deg + (size if kind == "elevator" and t[k] < pulse else 0.0)
            k1 = self.derivatives(s, tr.thrust, de)
            k2 = self.derivatives(s + 0.5 * dt * k1, tr.thrust, de)
            k3 = self.derivatives(s + 0.5 * dt * k2, tr.thrust, de)
            k4 = self.derivatives(s + dt * k3, tr.thrust, de)
            s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.all(np.isfinite(s)) or s[0] < 1.0:
                out = out[:k + 1]
                t = t[:k + 1]
                break
        return {"t": t, "V": out[:, 0], "gamma_deg": np.degrees(out[:, 1]), "alpha_deg": np.degrees(out[:, 2]),
                "q_deg_s": np.degrees(out[:, 3]), "h": out[:, 4], "x": out[:, 5],
                "theta_deg": np.degrees(out[:, 1] + out[:, 2])}


def handling_notes(modes: list[Mode]) -> list[str]:
    """Plain-language quality notes (MIL-F-8785C style limits, category B flight phase)."""
    notes = []
    for m in modes:
        if m.name == "Short period":
            if m.damping < 0:
                notes.append("Short period UNSTABLE: the nose diverges - move the CG forward.")
            elif m.damping < 0.3:
                notes.append(f"Short period lightly damped (ζ {m.damping:.2f}): pitch bobbles after inputs.")
            elif m.damping > 2.0:
                notes.append(f"Short period very heavily damped (ζ {m.damping:.2f}): sluggish pitch response.")
            else:
                notes.append(f"Short period well damped (ζ {m.damping:.2f}, {m.omega_n:.1f} rad/s).")
        elif m.name == "Phugoid":
            if m.damping < 0:
                notes.append(f"Phugoid slowly divergent (doubles in {m.half_or_double_time:.0f} s): acceptable if "
                             "slow (> 55 s).")
            elif m.damping < 0.04:
                notes.append(f"Phugoid lightly damped (ζ {m.damping:.3f}): long, slow speed-height exchange.")
            else:
                notes.append(f"Phugoid damped (ζ {m.damping:.2f}, period {m.period:.0f} s).")
    return notes


def default_radius_of_gyration(polar: PolarData) -> float:
    """A typical pitch radius of gyration: about 0.3 × span for light aircraft and models."""
    return max(0.3 * polar.span_m, 2.0 * polar.ref_length_m)


__all__ = ["Aircraft", "FlightModel", "Trim", "Mode", "handling_notes", "default_radius_of_gyration"]
