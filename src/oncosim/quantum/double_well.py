"""Proton transfer in a Watson-Crick base pair: potential and eigenstates.

Physical setting
----------------
Watson and Crick noted in 1953 that if a base adopts a rare tautomeric
form -- guanine as the enol, cytosine as the imino -- it mispairs, and the
mispairing is copied into a permanent point mutation.  Loewdin (1963)
proposed that the tautomer is reached by the hydrogen-bond proton
*tunnelling* across to the opposite base.  This module makes that idea
numerical: it builds the one-dimensional potential felt by the transferring
proton and solves its Schroedinger equation essentially exactly.

The potential is a back-to-back double Morse well,

.. math::

    V(x) = V_1\\left[1 - e^{-a_1 (x - r_1)}\\right]^2
         + V_2\\left[1 - e^{+a_2 (x - r_2)}\\right]^2 ,

with the deep (canonical, amino-keto) well on the left and the shallow
(tautomeric, imino-enol) well on the right.

Validation of the G-C parameter set
-----------------------------------
:data:`GC_SLOCOMBE_2022` carries the parameters that Slocombe, Sacchi and
Al-Khalili fitted to DFT/nudged-elastic-band calculations of the G-C
double proton transfer (their Table I).  The *sum* form above -- rather
than a piecewise minimum -- is what reproduces their reported surface, and
the reconstruction is verified against five further published numbers it
was not fitted to (see ``tests/test_quantum.py``):

======================================  ==========  ==========
quantity                                published    this code
======================================  ==========  ==========
forward barrier ``E_f``                  0.705 eV    0.705 eV
reverse barrier ``E_r``                  0.270 eV    0.270 eV
reaction asymmetry ``dE``                0.435 eV    0.435 eV
barrier frequency ``omega_b``          0.00277 au  0.00277 au
zero-point energy ``E_0``                0.049 eV    0.049 eV
index of first eigenstate with              7th        7th
amplitude in the shallow well
======================================  ==========  ==========

Numerical method
----------------
The Colbert-Miller sinc discrete variable representation (DVR) on a
uniform grid.  For a grid of spacing ``d`` the kinetic energy matrix is

.. math::

    T_{ij} = \\frac{\\hbar^2}{2 m d^2}(-1)^{i-j}
        \\begin{cases} \\pi^2/3 & i = j\\\\ 2/(i-j)^2 & i \\ne j\\end{cases}

and the potential is diagonal.  The DVR converges exponentially in the
grid spacing for smooth potentials, so a few hundred points already give
eigenvalues converged to ``1e-6`` eV; this is far cheaper and more accurate
than finite differences for the deeply bound states that matter here.

References
----------
Watson & Crick (1953) Nature 171:964-967.
Loewdin, P.-O. (1963) Rev. Mod. Phys. 35:724-732.
Colbert & Miller (1992) J. Chem. Phys. 96:1982-1991.
Slocombe, Al-Khalili & Sacchi (2021) Phys. Chem. Chem. Phys. 23:4141-4150.
Slocombe, Sacchi & Al-Khalili (2022) Commun. Phys. 5:109.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
import scipy.linalg as sla

from .units import BOHR_ANGSTROM, HARTREE_EV, PROTON_MASS_AU

__all__ = [
    "DoubleMorsePotential",
    "GC_SLOCOMBE_2022",
    "AT_ILLUSTRATIVE",
    "ProtonEigenstates",
    "solve_double_well",
]


@dataclass(frozen=True)
class DoubleMorsePotential:
    """Back-to-back double Morse potential, in Hartree atomic units.

    Attributes
    ----------
    v1, v2:
        Morse well depths (Hartree) for the canonical (left) and
        tautomeric (right) wells.
    a1, a2:
        Morse width parameters (inverse Bohr).  Larger means a narrower,
        stiffer well.
    r1, r2:
        Positions of the two Morse minima (Bohr).
    label:
        Human-readable provenance string.
    """

    v1: float
    v2: float
    a1: float
    a2: float
    r1: float
    r2: float
    label: str = "custom"

    # ------------------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Potential energy in Hartree at position(s) ``x`` in Bohr."""
        x = np.asarray(x, dtype=np.float64)
        left = self.v1 * (1.0 - np.exp(-self.a1 * (x - self.r1))) ** 2
        right = self.v2 * (1.0 - np.exp(self.a2 * (x - self.r2))) ** 2
        return left + right

    # ------------------------------------------------------------------
    def _grid(self, n: int = 200_001) -> tuple[np.ndarray, np.ndarray]:
        pad = 3.0
        x = np.linspace(self.r1 - pad, self.r2 + pad, n)
        return x, self(x)

    @cached_property
    def stationary_points(self) -> tuple[float, float, float]:
        """``(x_canonical, x_barrier, x_tautomeric)`` in Bohr."""
        x, v = self._grid()
        # Locate every interior local minimum and maximum by sign changes of
        # the discrete derivative.  Anchoring on ``r1``/``r2`` is unreliable:
        # in a strongly asymmetric well the true minima can sit far from the
        # nominal Morse centres, and a monotone flank then yields a spurious
        # "minimum" at the grid edge with a zero reverse barrier.
        dv = np.diff(v)
        sign_change = np.flatnonzero(np.sign(dv[:-1]) != np.sign(dv[1:])) + 1
        minima = [int(i) for i in sign_change if dv[i - 1] < 0 <= dv[i]]
        maxima = [int(i) for i in sign_change if dv[i - 1] > 0 >= dv[i]]
        if len(minima) < 2 or not maxima:
            raise ValueError(
                f"{self.label}: parameters do not define a double well "
                f"(found {len(minima)} interior minima, {len(maxima)} maxima)"
            )
        i_left, i_right = minima[0], minima[-1]
        interior = [i for i in maxima if i_left < i < i_right]
        if not interior:
            raise ValueError(f"{self.label}: no barrier between the two minima")
        i_bar = max(interior, key=lambda i: v[i])
        return (
            _polish(x, v, i_left),
            _polish(x, v, i_bar),
            _polish(x, v, i_right),
        )

    @property
    def x_canonical(self) -> float:
        return self.stationary_points[0]

    @property
    def x_barrier(self) -> float:
        """Position of the barrier top -- the transition-state dividing surface."""
        return self.stationary_points[1]

    @property
    def x_tautomeric(self) -> float:
        return self.stationary_points[2]

    @property
    def forward_barrier(self) -> float:
        """``E_f`` in Hartree: canonical minimum to barrier top."""
        return float(self(self.x_barrier) - self(self.x_canonical))

    @property
    def reverse_barrier(self) -> float:
        """``E_r`` in Hartree: tautomeric minimum to barrier top."""
        return float(self(self.x_barrier) - self(self.x_tautomeric))

    @property
    def asymmetry(self) -> float:
        """``dE = E_f - E_r`` in Hartree: how much the tautomer costs."""
        return float(self(self.x_tautomeric) - self(self.x_canonical))

    @property
    def transfer_distance_angstrom(self) -> float:
        """Distance between the two minima, in Angstrom."""
        return (self.x_tautomeric - self.x_canonical) * BOHR_ANGSTROM

    # ------------------------------------------------------------------
    def curvature(self, x: float) -> float:
        """Second derivative ``V''(x)`` in Hartree/Bohr^2 (analytic)."""
        e1 = np.exp(-self.a1 * (x - self.r1))
        e2 = np.exp(self.a2 * (x - self.r2))
        d2_left = 2.0 * self.v1 * self.a1**2 * e1 * (2.0 * e1 - 1.0)
        d2_right = 2.0 * self.v2 * self.a2**2 * e2 * (2.0 * e2 - 1.0)
        return float(d2_left + d2_right)

    def harmonic_frequency(self, x: float, mass: float = PROTON_MASS_AU) -> float:
        """``omega = sqrt(|V''| / m)`` at ``x``, in atomic units.

        At a minimum this is the well frequency (so the zero-point energy
        is ``hbar*omega/2``); at the barrier top it is the magnitude of the
        imaginary frequency that sets the tunnelling crossover temperature.
        """
        return float(np.sqrt(abs(self.curvature(x)) / mass))

    @property
    def zero_point_energy_harmonic(self) -> float:
        """Harmonic estimate ``hbar*omega/2`` in the canonical well (Hartree)."""
        return 0.5 * self.harmonic_frequency(self.x_canonical)

    # ------------------------------------------------------------------
    def summary_ev(self) -> dict[str, float]:
        """Key energies in eV plus geometry in Angstrom, for reporting."""
        return {
            "forward_barrier_eV": self.forward_barrier * HARTREE_EV,
            "reverse_barrier_eV": self.reverse_barrier * HARTREE_EV,
            "asymmetry_eV": self.asymmetry * HARTREE_EV,
            "zero_point_energy_eV": self.zero_point_energy_harmonic * HARTREE_EV,
            "barrier_frequency_au": self.harmonic_frequency(self.x_barrier),
            "transfer_distance_A": self.transfer_distance_angstrom,
        }

    def scaled(self, *, barrier_factor: float = 1.0, asymmetry_factor: float = 1.0):
        """Return a variant with the well depths rescaled.

        Useful for sensitivity analysis: the DFT surfaces carry an
        uncertainty of order 0.05 eV, and the tautomer population depends
        exponentially on the asymmetry, so a 0.05 eV shift moves it by a
        factor of ~6 at body temperature.
        """
        return replace(
            self,
            v1=self.v1 * barrier_factor,
            v2=self.v2 * barrier_factor * asymmetry_factor,
            label=f"{self.label}(scaled)",
        )


def _polish(x: np.ndarray, v: np.ndarray, i: int) -> float:
    """Sub-grid stationary point by a parabola through three samples."""
    if i <= 0 or i >= x.size - 1:
        return float(x[i])
    y0, y1, y2 = v[i - 1], v[i], v[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(x[i])
    shift = 0.5 * (y0 - y2) / denom
    if abs(shift) > 1.0:  # pathological; fall back to the grid point
        return float(x[i])
    return float(x[i] + shift * (x[1] - x[0]))


#: G-C double proton transfer surface (Slocombe, Sacchi & Al-Khalili 2022,
#: Commun. Phys. 5:109, Table I).  Fitted to DFT/ML-NEB calculations.
GC_SLOCOMBE_2022 = DoubleMorsePotential(
    v1=0.1617,
    v2=0.0820,
    a1=0.305,
    a2=0.755,
    r1=-2.7,
    r2=2.1,
    label="G-C (Slocombe et al. 2022)",
)

#: Illustrative A-T surface.  **Not** a published fit: the width and depth
#: of the shallow well were retuned until the reverse barrier came out near
#: 0.05 eV, which is the qualitative feature Slocombe et al. (2021) report
#: for A-T and the reason the A*-T* tautomer is far too short-lived to
#: survive to base insertion.  Use it for sensitivity studies and for
#: contrast with G-C, never as a quantitative A-T prediction.
AT_ILLUSTRATIVE = DoubleMorsePotential(
    v1=0.1617,
    v2=0.08354,
    a1=0.305,
    a2=0.57848,
    r1=-2.7,
    r2=2.1,
    label="A-T (illustrative, not a published fit)",
)


# ----------------------------------------------------------------------
# eigenstates
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProtonEigenstates:
    """Bound-state solution of the proton in a double well."""

    x: np.ndarray
    """Grid positions in Bohr."""
    energies: np.ndarray
    """Eigenvalues in Hartree, ascending."""
    wavefunctions: np.ndarray
    """``wavefunctions[:, n]`` is state ``n``, normalised as ``sum |psi|^2 = 1``."""
    potential: np.ndarray
    """Potential sampled on ``x``, in Hartree."""
    x_barrier: float
    """Dividing surface used to split the two wells."""
    mass: float
    """Particle mass in electron masses."""

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    def probability_right(self) -> np.ndarray:
        """Per-state probability of finding the proton past the barrier."""
        mask = self.x > self.x_barrier
        return np.asarray((self.wavefunctions[mask, :] ** 2).sum(axis=0))

    def first_tautomeric_state(self, threshold: float = 0.5) -> int:
        """Index of the lowest state localised in the shallow (right) well.

        A state counts as tautomeric when more than ``threshold`` of its
        probability lies beyond the barrier.  For the published G-C
        surface this returns 6, i.e. the *seventh* eigenstate -- exactly
        what Slocombe et al. report.
        """
        pr = self.probability_right()
        idx = np.flatnonzero(pr > threshold)
        return int(idx[0]) if idx.size else -1

    def energies_ev(self, relative_to_minimum: bool = True) -> np.ndarray:
        """Eigenvalues in eV, by default measured from the potential minimum."""
        offset = self.potential.min() if relative_to_minimum else 0.0
        return (self.energies - offset) * HARTREE_EV

    def n_below_barrier(self) -> int:
        """Number of eigenstates lying below the barrier top."""
        i = int(np.argmin(np.abs(self.x - self.x_barrier)))
        return int(np.count_nonzero(self.energies < self.potential[i]))


def solve_double_well(
    potential: DoubleMorsePotential = GC_SLOCOMBE_2022,
    *,
    mass: float = PROTON_MASS_AU,
    n_grid: int = 900,
    n_states: int | None = 60,
    pad: float = 2.5,
) -> ProtonEigenstates:
    """Solve the 1-D Schroedinger equation by Colbert-Miller sinc DVR.

    Parameters
    ----------
    potential:
        The double-well surface; defaults to the published G-C fit.
    mass:
        Transferring particle mass in electron masses.  Swap in
        :data:`~oncosim.quantum.units.DEUTERON_MASS_AU` to compute the
        kinetic isotope effect, the one prediction of the tunnelling
        hypothesis that is experimentally accessible.
    n_grid:
        Number of DVR points.  900 over the default range converges the
        low-lying eigenvalues to below 1e-6 eV.
    n_states:
        Number of lowest eigenstates to return.  ``None`` returns all,
        which costs a full dense diagonalisation.
    pad:
        Extra range in Bohr beyond each Morse minimum.  The left wall is
        steep, the right one soft, so the grid must not be so wide that
        unbound continuum-like states dominate the thermal average.

    Returns
    -------
    :class:`ProtonEigenstates`
    """
    if n_grid < 32:
        raise ValueError("n_grid must be at least 32")
    x = np.linspace(potential.r1 - pad, potential.r2 + pad, n_grid)
    d = x[1] - x[0]

    i = np.arange(n_grid)
    diff = i[:, None] - i[None, :]
    off = np.where(diff == 0, 1, diff)  # placeholder to avoid 0-division
    kinetic = np.where(diff == 0, np.pi**2 / 3.0, 2.0 / off**2) * (-1.0) ** diff
    kinetic *= 1.0 / (2.0 * mass * d**2)

    hamiltonian = kinetic + np.diag(potential(x))
    if n_states is None or n_states >= n_grid:
        energies, vectors = np.linalg.eigh(hamiltonian)
    else:
        energies, vectors = sla.eigh(hamiltonian, subset_by_index=(0, int(n_states) - 1))

    return ProtonEigenstates(
        x=x,
        energies=energies,
        wavefunctions=vectors,
        potential=potential(x),
        x_barrier=potential.x_barrier,
        mass=mass,
    )
