"""From tautomer populations to a per-base-pair mutation rate.

This module closes the loop between quantum mechanics and the somatic
evolution models in :mod:`oncosim.evolution`: it turns a proton
wavefunction into the parameter ``mu`` -- mutations per cell division --
that the branching-process machinery consumes.

The chain of reasoning
----------------------
1. The hydrogen-bond proton sits in a double well.  At thermal
   equilibrium there is some probability ``P_taut`` of finding it on the
   tautomeric side.
2. Interconversion is fast (``k_r`` of order ``10^9 s^-1``) compared with the
   replication fork's dwell time (milliseconds), so the tautomer cannot
   "survive" from strand separation to base insertion.  What matters is
   the *instantaneous equilibrium* occupancy at the moment the polymerase
   reads the template -- which is ``P_taut``.
3. A tautomeric template mispairs.  Only a fraction ``f_fix`` of those
   mispairings survives polymerase proofreading and mismatch repair to
   become a fixed point mutation.
4. Genome-wide: ``mu = N_GC * P_taut * f_fix`` mutations per replication.

Step 3 is where the biology, not the physics, dominates: ``f_fix`` is
uncertain over two orders of magnitude.  Rather than pick a value, this
module reports the *implied* ``f_fix`` needed to match the measured
spontaneous rate, which is a much more honest way to test the hypothesis.

Two competing numbers for ``P_taut``
------------------------------------
For the published G-C surface this module computes
``P_taut ~ 4.8e-8`` at 310 K from the exact canonical (Gibbs) density over
the DVR eigenstates.  It is within a factor of two of the classical
configurational integral, as it must be: tunnelling cannot shift an
equilibrium, and the small residual difference is a zero-point effect that
*lowers* the population, because the tautomeric well is the narrower of
the two and so has the higher zero-point energy.

Slocombe, Sacchi & Al-Khalili (2022) instead report ``1.73e-4`` from the
stationary state of a Wigner-Caldeira-Leggett master equation with a
low-temperature correction.  The two disagree by ~4 orders of magnitude.
The difference is methodological, not numerical:

* the WCL equation is derived in the weak-coupling, high-temperature
  limit and its stationary state is not exactly the Gibbs state, so it is
  not bound by detailed balance;
* the reported quantity is a Wigner-function integral over a half-space,
  and the Wigner function is a quasi-probability that can go negative --
  which the same paper observes;
* the paper's own rate constants satisfy ``k_f/k_r = 4.5e-8``, i.e. its
  *kinetic* equilibrium constant agrees with the value computed here, not
  with ``1.73e-4``.

Biology breaks the tie the same way.  ``4.8e-8`` per G-C pair implies ~61
tautomeric templates per replication and a required fixation probability
near 0.2, which is plausible after proofreading and mismatch repair;
``1.73e-4`` implies ~2e5 tautomeric templates per replication and would
require ``f_fix`` below ``10^-4`` to be compatible with the observed ~10^-8
per-base-pair mutation rate -- a tension the original authors themselves
flag.  This module computes both so the reader can see the size of the gap
rather than take a side on faith.

References
----------
Slocombe, Sacchi & Al-Khalili (2022) Commun. Phys. 5:109.
Kimsey et al. (2018) Nature 554:195-201 (NMR detection of transient
    Watson-Crick-like mispairs, the closest experimental handle).
Lynch et al. (2016) Nat. Rev. Genet. 17:704-714 (measured mutation rates).
Milholland et al. (2017) Nat. Commun. 8:15183 (somatic vs germline rates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .double_well import GC_SLOCOMBE_2022, DoubleMorsePotential, solve_double_well
from .tunnelling import RateConstants, rate_constants
from .units import KB_HARTREE_PER_K, PROTON_MASS_AU

__all__ = [
    "tautomer_occupancy_quantum",
    "tautomer_occupancy_classical",
    "TautomerAnalysis",
    "analyse_tautomer",
    "HUMAN_GC_PAIRS_HAPLOID",
    "OBSERVED_MUTATION_RATE_PER_BP",
    "SLOCOMBE_REPORTED_OCCUPANCY",
]

#: Number of G-C base pairs in a haploid human genome:
#: 3.1e9 bp at ~41% GC content.
HUMAN_GC_PAIRS_HAPLOID = 0.41 * 3.1e9

#: Order-of-magnitude spontaneous point mutation rate per base pair per
#: replication in human somatic cells (Lynch 2016; Milholland 2017).  Used
#: only as a consistency yardstick, never as a fitted quantity.
OBSERVED_MUTATION_RATE_PER_BP = 1e-8

#: The open-quantum-system value reported by Slocombe et al. (2022), kept
#: for explicit side-by-side comparison.
SLOCOMBE_REPORTED_OCCUPANCY = 1.73e-4


def tautomer_occupancy_quantum(
    potential: DoubleMorsePotential = GC_SLOCOMBE_2022,
    temperature_k: float = 310.15,
    *,
    mass: float = PROTON_MASS_AU,
    n_grid: int = 900,
    n_states: int = 60,
) -> float:
    """Exact canonical probability of the proton being past the barrier.

    .. math::

        P_{\\text{taut}} = \\frac{\\sum_n e^{-E_n/k_BT}
            \\int_{x>x_b}|\\psi_n|^2\\,dx}{\\sum_n e^{-E_n/k_BT}}

    This is a genuine probability (unlike a Wigner-function half-space
    integral) and it automatically respects detailed balance.

    Convergence: states more than ~30 ``k_BT`` above the ground state
    contribute below ``1e-13`` and are irrelevant at the ``1e-8`` level;
    ``n_states = 60`` is comfortably converged for the G-C surface at
    physiological temperature.
    """
    states = solve_double_well(potential, mass=mass, n_grid=n_grid, n_states=n_states)
    kt = KB_HARTREE_PER_K * temperature_k
    weights = np.exp(-(states.energies - states.energies[0]) / kt)
    p_right = states.probability_right()
    return float(np.sum(weights * p_right) / np.sum(weights))


def tautomer_occupancy_classical(
    potential: DoubleMorsePotential = GC_SLOCOMBE_2022,
    temperature_k: float = 310.15,
    *,
    n_grid: int = 200_001,
    pad: float = 2.5,
) -> float:
    """Classical configurational probability past the barrier.

    .. math::

        P_{\\text{class}} = \\frac{\\int_{x>x_b} e^{-V(x)/k_BT}dx}
                                  {\\int e^{-V(x)/k_BT}dx}

    The momentum integral is Gaussian and cancels, so only the
    configurational part survives.  Comparing this with the quantum result
    isolates the zero-point contribution.
    """
    kt = KB_HARTREE_PER_K * temperature_k
    x = np.linspace(potential.r1 - pad, potential.r2 + pad, n_grid)
    v = potential(x)
    boltz = np.exp(-(v - v.min()) / kt)
    total = np.trapezoid(boltz, x)
    right = np.trapezoid(np.where(x > potential.x_barrier, boltz, 0.0), x)
    return float(right / total)


@dataclass(frozen=True)
class TautomerAnalysis:
    """Everything needed to judge the proton-tunnelling mutation hypothesis."""

    potential_label: str
    temperature_k: float
    occupancy_quantum: float
    occupancy_classical: float
    rates: RateConstants
    n_gc_pairs: float

    # ------------------------------------------------------------------
    @property
    def zero_point_factor(self) -> float:
        """Quantum / classical occupancy.

        Below 1 means zero-point energy *suppresses* the tautomer, which is
        what happens when the product well is narrower than the reactant
        well.
        """
        return self.occupancy_quantum / self.occupancy_classical

    @property
    def kinetic_equilibrium_constant(self) -> float:
        """``k_f / k_r`` -- an independent route to the same equilibrium."""
        return self.rates.equilibrium_constant

    def survival_probability(self, wait_time_s: float) -> float:
        """Chance a tautomer formed now still exists after ``wait_time_s``.

        With ``k_r`` of order ``10^9 s^-1`` this is numerically zero for any
        biologically relevant delay, which is precisely why the *frozen
        tautomer* picture fails and the *instantaneous equilibrium* picture
        is the right one.
        """
        return float(np.exp(-self.rates.k_reverse * wait_time_s))

    # ------------------------------------------------------------------
    def tautomeric_templates_per_replication(
        self, occupancy: float | None = None
    ) -> float:
        """Expected number of G-C templates in tautomeric form genome-wide."""
        p = self.occupancy_quantum if occupancy is None else occupancy
        return self.n_gc_pairs * p

    def mutations_per_replication(
        self, fixation_probability: float, occupancy: float | None = None
    ) -> float:
        """Genome-wide fixed point mutations per replication from this channel.

        ``fixation_probability`` folds together polymerase base selection,
        3'-5' proofreading and mismatch repair.  A tautomer that mimics
        Watson-Crick geometry evades selection and proofreading, so the
        plausible range is set mostly by mismatch repair: roughly
        ``1e-3`` to ``1e-1``.
        """
        return self.tautomeric_templates_per_replication(occupancy) * float(
            fixation_probability
        )

    def implied_fixation_probability(
        self,
        occupancy: float | None = None,
        observed_rate_per_bp: float = OBSERVED_MUTATION_RATE_PER_BP,
    ) -> float:
        """``f_fix`` required for this occupancy to explain the observed rate.

        This inverts the hypothesis instead of tuning it: a value inside
        ``[1e-3, 1]`` means the mechanism can plausibly carry the observed
        mutation load; a value far below that means the predicted tautomer
        population is too large to be compatible with observation.
        """
        p = self.occupancy_quantum if occupancy is None else occupancy
        if p <= 0:
            return float("inf")
        return float(observed_rate_per_bp / p)

    # ------------------------------------------------------------------
    def report(self) -> dict[str, float | str]:
        """Flat dictionary suitable for printing or tabulating."""
        return {
            "potential": self.potential_label,
            "T_K": self.temperature_k,
            "P_taut_quantum": self.occupancy_quantum,
            "P_taut_classical": self.occupancy_classical,
            "zero_point_factor": self.zero_point_factor,
            "K_eq_kinetic": self.kinetic_equilibrium_constant,
            "kappa_tunnelling": self.rates.tunnelling_factor,
            "k_forward_s^-1": self.rates.k_forward,
            "k_reverse_s^-1": self.rates.k_reverse,
            "tautomer_lifetime_s": self.rates.tautomer_lifetime_s,
            "templates_per_replication": self.tautomeric_templates_per_replication(),
            "implied_f_fix": self.implied_fixation_probability(),
            "implied_f_fix_if_slocombe": self.implied_fixation_probability(
                SLOCOMBE_REPORTED_OCCUPANCY
            ),
        }


def analyse_tautomer(
    potential: DoubleMorsePotential = GC_SLOCOMBE_2022,
    temperature_k: float = 310.15,
    *,
    n_gc_pairs: float = HUMAN_GC_PAIRS_HAPLOID,
    mass: float = PROTON_MASS_AU,
) -> TautomerAnalysis:
    """Run the full quantum-to-mutation-rate analysis for one base pair."""
    return TautomerAnalysis(
        potential_label=potential.label,
        temperature_k=temperature_k,
        occupancy_quantum=tautomer_occupancy_quantum(
            potential, temperature_k, mass=mass
        ),
        occupancy_classical=tautomer_occupancy_classical(potential, temperature_k),
        rates=rate_constants(potential, temperature_k, mass=mass),
        n_gc_pairs=n_gc_pairs,
    )
