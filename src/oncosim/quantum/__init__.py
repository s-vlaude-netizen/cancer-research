"""Quantum mechanics of spontaneous point mutation in DNA.

Solves for the hydrogen-bond proton in a Watson-Crick base pair and turns
the result into a per-base-pair mutation rate that the somatic-evolution
models can consume.
"""

from .double_well import (
    AT_ILLUSTRATIVE,
    GC_SLOCOMBE_2022,
    DoubleMorsePotential,
    ProtonEigenstates,
    solve_double_well,
)
from .tautomer import (
    TautomerAnalysis,
    analyse_tautomer,
    tautomer_occupancy_classical,
    tautomer_occupancy_quantum,
)
from .tunnelling import (
    RateConstants,
    barrier_action,
    crossover_temperature,
    kinetic_isotope_effect,
    rate_constants,
    thermal_tunnelling_factor,
    transmission_probability,
)

__all__ = [
    "AT_ILLUSTRATIVE",
    "GC_SLOCOMBE_2022",
    "DoubleMorsePotential",
    "ProtonEigenstates",
    "RateConstants",
    "TautomerAnalysis",
    "analyse_tautomer",
    "barrier_action",
    "crossover_temperature",
    "kinetic_isotope_effect",
    "rate_constants",
    "solve_double_well",
    "tautomer_occupancy_classical",
    "tautomer_occupancy_quantum",
    "thermal_tunnelling_factor",
    "transmission_probability",
]
