"""Thermally activated tunnelling of the hydrogen-bond proton.

What tunnelling does and does not do
------------------------------------
It is worth being precise, because the literature is not.  For a
one-dimensional barrier, the tunnelling correction factor ``kappa(T)``
multiplies the forward and the reverse transition-state rate by the *same*
amount.  It therefore

* **changes the kinetics** -- how fast canonical and tautomeric forms
  interconvert, hence the tautomer's lifetime; but
* **cannot change the equilibrium constant** ``K_eq = k_f / k_r``, because
  detailed balance fixes that from the free-energy difference alone.

Any claim that tunnelling *raises the tautomer population* is therefore a
claim about non-equilibrium dynamics or about zero-point effects, not
about tunnelling per se.  :mod:`oncosim.quantum.tautomer` quantifies both.

Method
------
The transmission probability through the barrier uses the Kemble uniform
semiclassical form

.. math::  P(E) = \\frac{1}{1 + e^{2\\theta(E)}},\\qquad
   \\theta(E) = \\frac{1}{\\hbar}\\int_{x_-(E)}^{x_+(E)}\\sqrt{2m\\,[V(x)-E]}\\;dx ,

which correctly gives ``P = 1/2`` at the barrier top and reduces to plain
WKB deep under the barrier.  The thermal correction factor is then

.. math::  \\kappa(T) = e^{E_b/k_BT}\\,\\frac{1}{k_BT}\\int_0^\\infty P(E)\\,
   e^{-E/k_BT}\\,dE ,

normalised so that ``kappa -> 1`` when ``P`` is a step function.

The crossover temperature ``T_c = hbar*omega_b / (2 pi k_B)`` separates the
regime where thermal activation dominates (``T >> T_c``) from deep
tunnelling (``T << T_c``).  For the published G-C surface
``omega_b = 0.00277`` a.u. gives ``T_c = 139 K``, so at body temperature the
system sits well inside the *thermally activated* regime and tunnelling is
a ~40% correction, not an orders-of-magnitude one.

References
----------
Kemble, E.C. (1935) Phys. Rev. 48:549-561.
Bell, R.P. (1980) *The Tunnel Effect in Chemistry*, Chapman & Hall.
Haenggi, Talkner & Borkovec (1990) Rev. Mod. Phys. 62:251-341.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .double_well import DoubleMorsePotential
from .units import (
    AU_TIME_S,
    DEUTERON_MASS_AU,
    HARTREE_EV,
    KB_HARTREE_PER_K,
    PROTON_MASS_AU,
)

__all__ = [
    "barrier_action",
    "transmission_probability",
    "thermal_tunnelling_factor",
    "crossover_temperature",
    "RateConstants",
    "rate_constants",
    "kinetic_isotope_effect",
]


def _profile(
    potential: DoubleMorsePotential, n: int = 20001
) -> tuple[np.ndarray, np.ndarray]:
    """Potential sampled between the two minima (the reaction region)."""
    x = np.linspace(potential.x_canonical, potential.x_tautomeric, n)
    return x, potential(x)


def barrier_action(
    potential: DoubleMorsePotential,
    energy: float,
    *,
    mass: float = PROTON_MASS_AU,
    n: int = 20001,
) -> float:
    """Semiclassical action ``theta(E)`` for the barrier (dimensionless).

    ``energy`` is an absolute energy in Hartree.

    Below the barrier top this is the ordinary WKB integral over the
    classically forbidden region.  **Above** the barrier top it is
    continued analytically as ``theta = -pi (E - V_b) / (hbar omega_b)``,
    the exact parabolic-barrier result.  Returning zero there instead
    would pin ``P(E)`` at ``1/2`` for every energy above the barrier and
    make the thermal correction factor come out *below* one, which is an
    artefact rather than non-classical reflection.
    """
    v_barrier = float(potential(potential.x_barrier))
    if energy >= v_barrier:
        omega_b = potential.harmonic_frequency(potential.x_barrier, mass=mass)
        return float(-np.pi * (energy - v_barrier) / omega_b)
    x, v = _profile(potential, n)
    mask = v > energy
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(np.sqrt(2.0 * mass * (v[mask] - energy)), x[mask]))


def transmission_probability(
    potential: DoubleMorsePotential,
    energy: float | np.ndarray,
    *,
    mass: float = PROTON_MASS_AU,
) -> np.ndarray:
    """Kemble uniform transmission probability ``P(E)``."""
    energies = np.atleast_1d(np.asarray(energy, dtype=np.float64))
    out = np.empty_like(energies)
    for i, e in enumerate(energies):
        theta = barrier_action(potential, float(e), mass=mass)
        # exp overflow is not an error here: deep under the barrier
        # 2*theta is large and P underflows to 0, which is correct.
        out[i] = 1.0 / (1.0 + np.exp(min(2.0 * theta, 700.0)))
    return out


def crossover_temperature(
    potential: DoubleMorsePotential, *, mass: float = PROTON_MASS_AU
) -> float:
    """``T_c = hbar*omega_b / (2 pi k_B)`` in Kelvin.

    Above ``T_c`` the reaction is thermally activated with a modest
    tunnelling correction; below it, deep tunnelling takes over and the
    Arrhenius plot flattens.
    """
    omega_b = potential.harmonic_frequency(potential.x_barrier, mass=mass)
    return float(omega_b / (2.0 * np.pi * KB_HARTREE_PER_K))


def thermal_tunnelling_factor(
    potential: DoubleMorsePotential,
    temperature_k: float,
    *,
    mass: float = PROTON_MASS_AU,
    forward: bool = True,
    n_energy: int = 600,
) -> float:
    """Tunnelling enhancement ``kappa(T)`` of the transition-state rate.

    Identical for the forward and reverse directions (the same barrier is
    crossed), which is exactly why it cancels from ``K_eq``.  The
    ``forward`` flag only selects which well the energy zero is measured
    from, and is kept so callers can assert that cancellation explicitly.
    """
    kt = KB_HARTREE_PER_K * temperature_k
    v_barrier = float(potential(potential.x_barrier))
    v_well = float(
        potential(potential.x_canonical if forward else potential.x_tautomeric)
    )
    barrier = v_barrier - v_well

    energies = np.linspace(v_well, v_barrier + 40.0 * kt, n_energy)
    p = np.array(
        [
            1.0
            / (
                1.0
                + np.exp(
                    min(2.0 * barrier_action(potential, float(e), mass=mass), 700.0)
                )
            )
            for e in energies
        ]
    )
    integral = np.trapezoid(p * np.exp(-(energies - v_well) / kt), energies) / kt
    return float(integral / np.exp(-barrier / kt))


@dataclass(frozen=True)
class RateConstants:
    """Forward and reverse proton-transfer rate constants, in s^-1."""

    temperature_k: float
    k_forward_classical: float
    k_reverse_classical: float
    tunnelling_factor: float
    mass: float

    @property
    def k_forward(self) -> float:
        return self.k_forward_classical * self.tunnelling_factor

    @property
    def k_reverse(self) -> float:
        return self.k_reverse_classical * self.tunnelling_factor

    @property
    def equilibrium_constant(self) -> float:
        """``K_eq = k_f / k_r`` -- unchanged by tunnelling, by construction."""
        return self.k_forward / self.k_reverse

    @property
    def tautomer_lifetime_s(self) -> float:
        """``1 / k_reverse``: how long a tautomer survives once formed."""
        return 1.0 / self.k_reverse

    @property
    def canonical_lifetime_s(self) -> float:
        return 1.0 / self.k_forward

    def summary(self) -> dict[str, float]:
        return {
            "T_K": self.temperature_k,
            "k_forward_s^-1": self.k_forward,
            "k_reverse_s^-1": self.k_reverse,
            "kappa": self.tunnelling_factor,
            "K_eq": self.equilibrium_constant,
            "tautomer_lifetime_s": self.tautomer_lifetime_s,
        }


def rate_constants(
    potential: DoubleMorsePotential,
    temperature_k: float = 310.15,
    *,
    mass: float = PROTON_MASS_AU,
) -> RateConstants:
    """Transition-state rates with a semiclassical tunnelling correction.

    The classical prefactor is the harmonic well frequency divided by
    ``2*pi`` (the standard 1-D TST result ``k = (omega_0/2pi) exp(-E_b/kT)``).
    Using the full anharmonic partition function instead lowers the
    prefactor by a factor of a few; that systematic is smaller than the
    uncertainty on the DFT barrier itself, which enters exponentially.
    """
    omega_left = potential.harmonic_frequency(potential.x_canonical, mass=mass)
    omega_right = potential.harmonic_frequency(potential.x_tautomeric, mass=mass)
    kt = KB_HARTREE_PER_K * temperature_k

    kf = (omega_left / (2.0 * np.pi)) * np.exp(-potential.forward_barrier / kt)
    kr = (omega_right / (2.0 * np.pi)) * np.exp(-potential.reverse_barrier / kt)
    kappa = thermal_tunnelling_factor(potential, temperature_k, mass=mass)

    return RateConstants(
        temperature_k=temperature_k,
        k_forward_classical=float(kf / AU_TIME_S),
        k_reverse_classical=float(kr / AU_TIME_S),
        tunnelling_factor=kappa,
        mass=mass,
    )


def kinetic_isotope_effect(
    potential: DoubleMorsePotential,
    temperature_k: float = 310.15,
) -> dict[str, float]:
    """Ratio of proton to deuteron transfer rates -- the testable prediction.

    If proton tunnelling really drives spontaneous mutation, replacing the
    hydrogen-bond proton by deuterium should slow tautomerisation
    measurably.  A purely classical barrier crossing gives a KIE of
    ``sqrt(m_D/m_H) ~ 1.41`` from the prefactor alone; deep tunnelling
    gives much more.  The numbers returned here separate those two
    contributions so an experiment can be designed to distinguish them.
    """
    rc_h = rate_constants(potential, temperature_k, mass=PROTON_MASS_AU)
    rc_d = rate_constants(potential, temperature_k, mass=DEUTERON_MASS_AU)
    return {
        "kie_total": rc_h.k_forward / rc_d.k_forward,
        "kie_classical_prefactor": rc_h.k_forward_classical / rc_d.k_forward_classical,
        "kie_tunnelling_only": rc_h.tunnelling_factor / rc_d.tunnelling_factor,
        "kappa_H": rc_h.tunnelling_factor,
        "kappa_D": rc_d.tunnelling_factor,
        "crossover_temperature_H_K": crossover_temperature(potential),
        "crossover_temperature_D_K": crossover_temperature(
            potential, mass=DEUTERON_MASS_AU
        ),
        "barrier_eV": potential.forward_barrier * HARTREE_EV,
    }
