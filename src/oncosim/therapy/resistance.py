"""Pre-existing resistance and why combinations beat sequences.

The central result
------------------
Resistance to a targeted drug is almost never *caused* by the drug.  It is
thrown off continuously during the tumour's growth and is already present,
in a handful of cells, on the day treatment starts.  This is the Luria-
Delbruck argument transplanted to oncology, and it has a hard quantitative
consequence, first made precise by Bozic et al. (2013):

* **Sequential therapy offers no chance of cure**, even with drugs that
  share no cross-resistance.  By the time the first drug fails, the
  relapsed tumour is back at detectable size and has had ample
  opportunity to generate resistance to the second.
* **Simultaneous combination therapy can cure**, but only if no cell
  resistant to *all* drugs at once pre-exists -- which becomes exponentially
  unlikely as the tumour grows.

Where the numbers come from
---------------------------
Growing to ``M`` cells takes ``M/beta`` divisions, so resistance mutation
events number ``u M / beta``.  Each event founds a lineage that avoids
stochastic extinction with probability ``beta``.  The expected number of
*surviving* resistant lineages is therefore ``u M`` exactly, independent of
turnover, and

.. math::  P(\\text{no resistant cell}) \\approx e^{-uM}.

The *number* of resistant cells is a different matter: it is the
Luria-Delbruck variable of :mod:`oncosim.evolution.luria_delbruck`, with an
infinite-mean, heavy-tailed distribution.  Median and mean differ by orders
of magnitude, so this module reports quantiles from exact draws rather
than a mean that no patient ever experiences.

For a combination of ``k`` drugs, a cell must independently acquire all
``k`` resistance mutations.  Multiple resistance accumulates along the
lineages of already-resistant clones, so the expected number of
``k``-resistant cells picks up one factor of ``u`` and roughly one factor of
``ln M`` per additional drug -- which is why adding a third drug helps far
less than adding the second.

References
----------
Bozic et al. (2013) eLife 2:e00747.
Diaz et al. (2012) Nature 486:537-540 (pre-existing KRAS resistance).
Komarova & Wodarz (2005) PNAS 102:9714-9719.
Iwasa, Nowak & Michor (2006) Genetics 172:2557-2566.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..evolution.birth_death import BirthDeath
from ..evolution.luria_delbruck import ld_sample

__all__ = ["ResistanceModel", "expected_multi_resistant_cells", "max_curable_size"]


@dataclass(frozen=True)
class ResistanceModel:
    """Pre-existing resistance in a tumour of ``n_cells`` at detection.

    Attributes
    ----------
    n_cells:
        Tumour size when treatment starts.  A 1 cm lesion is roughly
        ``1e9`` cells; the detection limit of imaging is ``1e8``-``1e9``.
    mutation_rate:
        Probability per cell division of acquiring resistance to one
        specific drug.  Typically ``1e-6`` to ``1e-8``, depending on how
        many distinct mutations confer resistance to that agent -- a drug
        defeated by any of 50 point mutations has a much higher effective
        rate than one defeated by a single site.
    process:
        Birth-death parameters of the untreated tumour.
    resistance_fitness:
        Growth rate of resistant cells relative to sensitive ones *in the
        absence of drug*.  Values below 1 encode a cost of resistance,
        which is the entire basis of adaptive therapy.
    """

    n_cells: float = 1e9
    mutation_rate: float = 1e-7
    process: BirthDeath = field(default_factory=lambda: BirthDeath(1.0, 0.0))
    resistance_fitness: float = 1.0

    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError("n_cells must be at least 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be a probability")
        if self.resistance_fitness <= 0:
            raise ValueError("resistance_fitness must be positive")

    # ------------------------------------------------------------------
    @property
    def expected_surviving_lineages(self) -> float:
        """``u * M``: expected number of resistant lineages that avoid extinction.

        The turnover factor cancels: more divisions per net cell added
        (``1/beta``) is exactly offset by each new lineage being less
        likely to survive (``beta``).
        """
        return float(self.mutation_rate * self.n_cells)

    def probability_no_resistance(self) -> float:
        """``exp(-u M)`` -- the chance a single drug can cure."""
        return float(np.exp(-self.expected_surviving_lineages))

    def sample_resistant_cells(
        self, size: int = 1000, *, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Exact Luria-Delbruck draws of the resistant cell count at detection.

        Uses the closed-form compound-Poisson representation, so the cost
        is independent of ``n_cells``: a ``10^12``-cell tumour is as cheap as
        a ``10^3``-cell one.
        """
        m = self.mutation_rate * self.n_cells / self.process.beta
        return ld_sample(
            m,
            size,
            self.resistance_fitness,
            rng=rng,
            birth_rate=self.process.b,
            death_rate=self.process.d,
        )

    def resistance_quantiles(
        self,
        quantiles=(0.5, 0.9, 0.99),
        *,
        n_sim: int = 20_000,
        rng: np.random.Generator | None = None,
    ) -> dict[float, float]:
        """Quantiles of the resistant cell count.

        Reported instead of a mean because the Luria-Delbruck law has an
        infinite mean: the average is dominated by rare "jackpot" tumours
        that acquired resistance early, and describes no typical patient.
        """
        draws = self.sample_resistant_cells(n_sim, rng=rng)
        return {float(q): float(np.quantile(draws, q)) for q in quantiles}


def expected_multi_resistant_cells(
    n_cells: float,
    mutation_rates,
    process: BirthDeath | None = None,
) -> float:
    """Expected number of cells resistant to *all* drugs simultaneously.

    Parameters
    ----------
    mutation_rates:
        One per-division resistance rate per drug.  Cross-resistance --
        a single mutation defeating two drugs -- is *not* modelled here;
        pass a combined rate for such a pair instead, or use
        :func:`max_curable_size` with the higher effective rate.

    Notes
    -----
    Each additional drug multiplies the count by roughly ``u_i * ln M``:
    multiply-resistant cells accumulate along the lineages of
    already-resistant clones, whose total cell count carries the
    logarithmic factor of the Luria-Delbruck clone-size law.  The estimate
    is asymptotic in ``ln M`` and should be read as an order of magnitude,
    which is all that the input rates support anyway.
    """
    process = process or BirthDeath()
    rates = np.asarray(list(mutation_rates), dtype=np.float64)
    if rates.size == 0:
        raise ValueError("need at least one mutation rate")
    if np.any((rates < 0) | (rates > 1)):
        raise ValueError("mutation rates must be probabilities")
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")

    log_m = float(np.log(max(n_cells, np.e)))
    count = float(rates[0] * n_cells)
    for u in rates[1:]:
        count *= float(u) * log_m
    return count


def max_curable_size(
    mutation_rates,
    *,
    target_probability: float = 0.5,
    process: BirthDeath | None = None,
    hi: float = 1e15,
) -> float:
    """Largest tumour that a given combination can still cure.

    Returns the size ``M`` at which the probability of *no* fully resistant
    cell falls to ``target_probability``, assuming that count is Poisson.
    Solved by bisection on :func:`expected_multi_resistant_cells`.

    The practical message: for realistic rates, a single drug loses the
    ability to cure at around ``10^7`` cells -- far below the imaging
    detection limit -- while a two-drug combination pushes that boundary
    up by several orders of magnitude.
    """
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie in (0, 1)")
    target_count = -np.log(target_probability)

    lo = 1.0
    if expected_multi_resistant_cells(hi, mutation_rates, process) < target_count:
        return float(hi)
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if expected_multi_resistant_cells(mid, mutation_rates, process) < target_count:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0000001:
            break
    return float(np.sqrt(lo * hi))
