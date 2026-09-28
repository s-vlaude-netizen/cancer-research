"""Atomic units and the conversions used throughout :mod:`oncosim.quantum`.

Hartree atomic units (``hbar = m_e = e = 1``) are used internally because
the published potential-energy surfaces for base-pair proton transfer are
tabulated in them.  Everything exposed to the rest of the package is
converted to eV, Angstrom and seconds.
"""

from __future__ import annotations

__all__ = [
    "HARTREE_EV",
    "BOHR_ANGSTROM",
    "AU_TIME_S",
    "KB_HARTREE_PER_K",
    "KB_EV_PER_K",
    "PROTON_MASS_AU",
    "DEUTERON_MASS_AU",
    "BODY_TEMPERATURE_K",
    "thermal_energy_hartree",
    "thermal_energy_ev",
]

#: 1 Hartree in electronvolt (CODATA 2018).
HARTREE_EV = 27.211386245988

#: 1 Bohr radius in Angstrom.
BOHR_ANGSTROM = 0.529177210903

#: 1 atomic unit of time in seconds.
AU_TIME_S = 2.4188843265857e-17

#: Boltzmann constant in Hartree per Kelvin.
KB_HARTREE_PER_K = 3.166811563e-6

#: Boltzmann constant in eV per Kelvin.
KB_EV_PER_K = 8.617333262e-5

#: Proton mass in electron masses.
PROTON_MASS_AU = 1836.15267343

#: Deuteron mass in electron masses -- for kinetic isotope effect studies.
DEUTERON_MASS_AU = 3670.48296788

#: Human core body temperature.
BODY_TEMPERATURE_K = 310.15


def thermal_energy_hartree(temperature_k: float) -> float:
    """``k_B T`` in Hartree."""
    return KB_HARTREE_PER_K * temperature_k


def thermal_energy_ev(temperature_k: float) -> float:
    """``k_B T`` in eV -- 0.0267 eV at 310 K."""
    return KB_EV_PER_K * temperature_k
