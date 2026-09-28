"""Site frequency spectra of growing tumours: theory and exact sampling.

The site frequency spectrum (SFS) -- how many mutations are carried by
what fraction of cells -- is the single most information-dense summary a
bulk-sequenced tumour offers about its own history.  Two facts drive
everything here:

**1. Neutral exponential growth gives a 1/f tail.**
In a growing population, a neutral mutation's final frequency is set by
*when* it arose.  Mutations arise at rate ``mu * b * N(t)``; a lineage
present when the population has ``n`` cells reaches a final fraction above
``f`` with probability ``beta * exp(-beta * n * f)`` (the martingale limit
``W`` is zero with probability ``d/b`` and exponential otherwise).
Integrating over the growth history,

.. math::  M(f) \\;=\\; \\frac{\\mu}{\\beta}\\left(\\frac{1}{f} - 1\\right),
   \\qquad \\beta = \\frac{b-d}{b},

with ``mu`` the mutation rate per **division event**.  The corresponding
density is ``dM/df ∝ f^{-2}``.  Note what is and is not identifiable: only
the *ratio* ``mu/beta`` appears, so bulk sequencing of a single tumour
cannot separate a high mutation rate from high cell turnover.

**2. Selection bends the tail.**
For a two-type model where a fitter subclone with net growth ``lam1``
expands inside a background growing at ``lam0 < lam1``, Durrett (2015) and
Bozic et al. (2019) give a tail ``M(f) ∝ f^{-alpha}`` with
``alpha = lam0/lam1 < 1``, plus a "shoulder" at the subclone's own
frequency.  A measured exponent significantly below 1 is therefore
evidence *against* neutrality -- this is the core of the Williams/Sottoriva
vs. Tarabichi/McGranahan debate, and the reason this module reports the
fitted exponent with a confidence interval rather than a yes/no verdict.

Algorithmic contribution
------------------------
:func:`sample_sfs_above` draws an **exact** realisation of all mutations
above a frequency threshold ``f_min`` in ``O(mu / f_min)`` time -- entirely
independent of the tumour size ``N``.  The trick is that the mutation
count above ``f_min`` is Poisson with mean ``M(f_min)``, and the
frequencies are i.i.d. from the normalised ``f^{-2}`` density, which
inverts in closed form.  Simulating a 10^11-cell tumour's detectable SFS
therefore costs the same as a 10^4-cell one.  It is validated in the test
suite against :func:`simulate_genealogy_sfs`, a brute-force cell-by-cell
simulation that builds the full mutation tree.

References
----------
Durrett, R. (2013) Ann. Appl. Probab. 23:230-250.
Williams, Werner, Barnes, Graham & Sottoriva (2016) Nat. Genet. 48:238-244.
Bozic, Paterson & Waclaw (2019) PLoS Comput. Biol. 15:e1007368.
Tarabichi et al. (2018) Nat. Genet. 50:1630-1633 (critique of neutrality tests).
Werner et al. (2020) Nat. Commun. 11:1035.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .birth_death import BirthDeath

__all__ = [
    "expected_mutations_above",
    "sfs_density",
    "sample_sfs_above",
    "expected_mutations_above_selected",
    "fit_neutral_tail",
    "TailFit",
    "vaf_from_cell_fraction",
    "simulate_genealogy_sfs",
    "GenealogyResult",
]


# ----------------------------------------------------------------------
# theory
# ----------------------------------------------------------------------
def expected_mutations_above(
    f: float | np.ndarray,
    mu: float,
    process: BirthDeath | None = None,
    *,
    n_cells: float | None = None,
) -> np.ndarray:
    """``M(f)``: expected number of subclonal mutations above cell fraction ``f``.

    Parameters
    ----------
    f:
        Cell fraction(s) in ``(0, 1]``.
    mu:
        Expected number of new mutations per **division event**, summed
        over both daughter cells.  Whole-genome values are ~1-10 per
        division for most cancers and ~100+ for mismatch-repair-deficient
        hypermutators.  If your source quotes mutations *per daughter
        genome*, double it before passing it here -- this factor of two is
        the most common silent error in SFS-based rate estimates.
    process:
        Birth-death parameters; defaults to a pure birth process.
    n_cells:
        If given, apply the finite-size correction
        ``M(f) = mu_eff * ((1 - exp(-beta*N*f))/f - 1)``.  Matters only when
        ``beta * N * f`` is of order 1, i.e. right at the detection limit.
    """
    process = process or BirthDeath()
    f = np.asarray(f, dtype=np.float64)
    if np.any((f <= 0) | (f > 1)):
        raise ValueError("f must lie in (0, 1]")
    mu_eff = mu / process.beta
    if n_cells is None:
        return mu_eff * (1.0 / f - 1.0)
    beta_n = process.beta * float(n_cells)
    return mu_eff * ((1.0 - np.exp(-beta_n * f)) / f - 1.0)


def sfs_density(
    f: float | np.ndarray, mu: float, process: BirthDeath | None = None
) -> np.ndarray:
    """``-dM/df = mu_eff / f**2`` -- the neutral SFS density."""
    process = process or BirthDeath()
    f = np.asarray(f, dtype=np.float64)
    return (mu / process.beta) / f**2


def expected_mutations_above_selected(
    f: float | np.ndarray,
    mu: float,
    lam_background: float,
    lam_subclone: float,
    *,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Tail ``M(f) ∝ f**-alpha`` with ``alpha = lam_background / lam_subclone``.

    The exponent is the observable: a neutrally growing tumour has
    ``alpha = 1``; a subclone growing twice as fast as its background
    flattens the cumulative tail to ``alpha = 0.5``.  ``amplitude`` is the
    non-universal prefactor, which depends on when the subclone arose and
    is not predicted by the exponent argument alone.
    """
    if lam_subclone <= 0 or lam_background <= 0:
        raise ValueError("net growth rates must be positive")
    if lam_subclone < lam_background:
        raise ValueError("lam_subclone must exceed lam_background for a driver")
    f = np.asarray(f, dtype=np.float64)
    alpha = lam_background / lam_subclone
    return amplitude * mu * (f ** (-alpha) - 1.0)


def vaf_from_cell_fraction(
    f: float | np.ndarray, copy_number: int = 2, mutant_copies: int = 1
) -> np.ndarray:
    """Convert cell fraction to variant allele frequency.

    ``VAF = f * mutant_copies / copy_number``.  The default is the diploid
    heterozygous case, ``VAF = f/2`` -- the factor that is silently dropped
    in half the literature and that shifts a fitted ``mu`` by 2x.
    """
    f = np.asarray(f, dtype=np.float64)
    if copy_number < 1 or mutant_copies < 1 or mutant_copies > copy_number:
        raise ValueError("require 1 <= mutant_copies <= copy_number")
    return f * mutant_copies / copy_number


# ----------------------------------------------------------------------
# exact sampling above a detection threshold -- O(mu / f_min), N-independent
# ----------------------------------------------------------------------
def sample_sfs_above(
    mu: float,
    f_min: float,
    process: BirthDeath | None = None,
    *,
    rng: np.random.Generator | None = None,
    f_max: float = 1.0,
) -> np.ndarray:
    """Exact draw of all subclonal mutation frequencies in ``[f_min, f_max]``.

    Cost is ``O(mu/beta * (1/f_min - 1))`` -- the number of mutations
    actually returned -- and does **not** depend on the tumour's cell
    count.  This is what makes whole-tumour-scale SFS inference tractable.

    The construction is exact for the neutral model: the count above
    ``f_min`` is ``Poisson(M(f_min) - M(f_max))`` and, conditional on the
    count, frequencies are i.i.d. from the density ``∝ f^-2`` restricted to
    ``[f_min, f_max]``, whose inverse CDF is available in closed form.

    Returns
    -------
    Unsorted array of cell fractions, one entry per mutation.
    """
    process = process or BirthDeath()
    if rng is None:
        rng = np.random.default_rng()
    if not 0.0 < f_min < f_max <= 1.0:
        raise ValueError("require 0 < f_min < f_max <= 1")

    mu_eff = mu / process.beta
    expected = mu_eff * (1.0 / f_min - 1.0 / f_max)
    n = int(rng.poisson(expected))
    if n == 0:
        return np.empty(0, dtype=np.float64)
    # inverse CDF of density ∝ f^-2 on [f_min, f_max]
    u = rng.random(n)
    return 1.0 / (1.0 / f_min - u * (1.0 / f_min - 1.0 / f_max))


# ----------------------------------------------------------------------
# tail fitting
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TailFit:
    """Result of fitting ``M(f) = A f^-alpha`` to a cumulative SFS."""

    alpha: float
    alpha_stderr: float
    amplitude: float
    n_mutations: int
    f_range: tuple[float, float]

    @property
    def neutral_consistent(self) -> bool:
        """Whether ``alpha = 1`` lies within two standard errors.

        A screen, not a verdict.  The interval accounts for sampling noise
        in the frequencies only; it says nothing about sequencing depth,
        copy-number confounding, or purity, all of which distort a real
        spectrum far more than Poisson noise does.
        """
        return abs(self.alpha - 1.0) <= 2.0 * self.alpha_stderr

    @property
    def implied_growth_ratio(self) -> float:
        """``lam_background / lam_subclone`` implied by the exponent.

        Under the two-type model the tail exponent *is* that ratio, so
        ``alpha = 0.5`` means the subclone grows twice as fast as its
        background.  Values at or above 1 carry no selection signal.
        """
        return self.alpha


def fit_neutral_tail(
    frequencies: np.ndarray,
    f_min: float = 0.05,
    f_max: float = 0.4,
) -> TailFit:
    """Maximum-likelihood fit of the SFS tail exponent over a window.

    The window matters: below ``f_min`` sequencing noise and the detection
    limit distort the spectrum, above ``f_max`` clonal mutations dominate.
    The conventional window for ~100x whole-genome data is 0.05-0.4 in VAF.

    Method
    ------
    The cumulative spectrum is ``M(f) = A(f**-alpha - 1)``, so the *density*
    of subclonal frequencies is a pure power law ``p(f) ∝ f**-(alpha+1)``
    with no additive offset.  Fitting the density is therefore the clean
    parameterisation, and its exponent is estimated by maximum likelihood
    for a doubly-truncated Pareto on ``[f_min, f_max]``:

    .. math::

        \\ell(\\gamma) = -\\gamma \\sum_i \\ln f_i
            - n \\ln\\!\\int_{f_{\\min}}^{f_{\\max}} f^{-\\gamma}\\,df ,
        \\qquad \\alpha = \\gamma - 1 .

    **Do not** regress ``log M(f)`` on ``log f`` instead.  The ``-1`` in the
    cumulative form is not negligible inside a realistic window: over
    ``[0.05, 0.4]`` it biases a log-log slope to ``alpha = 1.22`` for data
    that are exactly neutral, and truncating the sample at ``f_max`` before
    accumulating makes it worse still.  That bias is large enough to
    manufacture apparent selection out of neutral data.

    The standard error comes from the observed Fisher information, so it
    is a genuine likelihood-based interval rather than a regression
    artefact of an arbitrary binning.
    """
    from scipy.optimize import minimize_scalar

    freqs = np.asarray(frequencies, dtype=np.float64).ravel()
    if not 0.0 < f_min < f_max:
        raise ValueError("require 0 < f_min < f_max")
    freqs = freqs[(freqs >= f_min) & (freqs <= f_max)]
    n = freqs.size
    if n < 10:
        raise ValueError("need at least 10 mutations inside the fitting window")
    sum_log = float(np.sum(np.log(freqs)))

    def log_norm(gamma: float) -> float:
        """log of the integral of f**-gamma over the window."""
        if abs(gamma - 1.0) < 1e-9:
            return float(np.log(np.log(f_max / f_min)))
        p = 1.0 - gamma
        return float(np.log(abs(f_max**p - f_min**p)) - np.log(abs(p)))

    def neg_ll(gamma: float) -> float:
        return gamma * sum_log + n * log_norm(gamma)

    opt = minimize_scalar(neg_ll, bounds=(0.05, 12.0), method="bounded")
    gamma = float(opt.x)

    # observed information by central differences on the log-likelihood
    h = 1e-4
    second = (neg_ll(gamma + h) - 2.0 * neg_ll(gamma) + neg_ll(gamma - h)) / h**2
    stderr = float(np.sqrt(1.0 / second)) if second > 0 else float("inf")

    alpha = gamma - 1.0
    # amplitude A of M(f) = A (f^-alpha - 1), matched to the observed count
    if abs(alpha) < 1e-9:
        amplitude = float(n / np.log(f_max / f_min))
    else:
        amplitude = float(n / (f_min**-alpha - f_max**-alpha))

    return TailFit(
        alpha=alpha,
        alpha_stderr=stderr,
        amplitude=amplitude,
        n_mutations=int(n),
        f_range=(f_min, f_max),
    )


# ----------------------------------------------------------------------
# brute-force reference: full mutation genealogy
# ----------------------------------------------------------------------
@dataclass
class GenealogyResult:
    """Output of a cell-by-cell simulation with a complete mutation tree."""

    n_cells: int
    """Number of living cells at the end of the simulation."""
    mutation_parent: np.ndarray
    """``parent[j]`` = index of the mutation ancestral to mutation ``j``, or -1."""
    mutation_count: np.ndarray
    """``count[j]`` = number of living cells carrying mutation ``j``."""
    time: float
    """Simulated time elapsed."""

    @property
    def frequencies(self) -> np.ndarray:
        """Cell fraction of each mutation."""
        if self.n_cells == 0:
            return np.zeros(0)
        return self.mutation_count / self.n_cells

    def subclonal_frequencies(self, f_max: float = 0.99) -> np.ndarray:
        """Frequencies of mutations that are not (near-)clonal."""
        f = self.frequencies
        return f[f <= f_max]


def simulate_genealogy_sfs(
    n_target: int,
    mu: float,
    process: BirthDeath | None = None,
    *,
    rng: np.random.Generator | None = None,
    max_events: int | None = None,
    condition_on_survival: bool = True,
    max_attempts: int = 10_000,
) -> GenealogyResult:
    """Brute-force cell-by-cell birth-death simulation with a mutation tree.

    This is the *reference implementation*: it makes no approximation
    beyond the model itself, tracks every mutation's ancestry explicitly,
    and costs ``O(n_target / beta)`` events.  It exists to validate
    :func:`sample_sfs_above`, and is only practical up to ~10^6 cells.

    ``mu`` is the expected number of new mutations per **division event**;
    each of the two daughters independently receives ``Poisson(mu/2)``
    private mutations under the infinite-sites assumption.  This is the
    convention under which ``M(f) = (mu/beta)(1/f - 1)`` holds.

    Parameters
    ----------
    condition_on_survival:
        A supercritical lineage still dies out with probability ``d/b``
        (40% at ``d/b = 0.4``).  Because the analytic SFS describes a tumour
        that *exists*, the default restarts extinct attempts so the output
        is conditioned on reaching ``n_target``.  Set to ``False`` to see
        unconditioned outcomes, which then include ``n_cells = 0``.  Mixing
        extinct runs into an SFS average is the single easiest way to
        "disprove" a correct theory by a factor of ``1 - d/b``.

    Returns
    -------
    :class:`GenealogyResult` with per-mutation carrier counts obtained by a
    single post-order accumulation over the mutation tree.
    """
    process = process or BirthDeath()
    if rng is None:
        rng = np.random.default_rng()
    if n_target < 1:
        raise ValueError("n_target must be >= 1")
    b, d = process.b, process.d
    if b <= d:
        raise ValueError("need b > d so the population can reach n_target")
    if max_events is None:
        max_events = int(50 * n_target / process.beta) + 1000

    parents: list[int] = []
    cells: list[int] = []
    t = 0.0
    for _attempt in range(max_attempts):
        # Mutation tree stored as a parent array; -1 marks a founder mutation.
        parents = []
        # Each living cell is identified by the index of its most recent
        # mutation (-1 = none); ancestry is recovered through ``parents``.
        cells = [-1]
        t = 0.0
        events = 0
        while len(cells) < n_target and cells and events < max_events:
            n = len(cells)
            t += float(rng.exponential(1.0 / (n * (b + d))))
            idx = int(rng.integers(n))
            if rng.random() < b / (b + d):
                # division: replace the parent by two daughters
                tip = cells[idx]
                daughters = []
                for _ in range(2):
                    k = int(rng.poisson(0.5 * mu))
                    node = tip
                    for _ in range(k):
                        parents.append(node)
                        node = len(parents) - 1
                    daughters.append(node)
                cells[idx] = daughters[0]
                cells.append(daughters[1])
            else:
                # death: swap-remove keeps this O(1)
                cells[idx] = cells[-1]
                cells.pop()
            events += 1
        if not condition_on_survival or len(cells) >= n_target:
            break
    else:
        raise RuntimeError(
            f"no lineage reached {n_target} cells in {max_attempts} attempts "
            f"(extinction probability {process.extinction_probability:.3f})"
        )

    n_cells = len(cells)
    n_mut = len(parents)
    counts = np.zeros(n_mut, dtype=np.int64)
    parent_arr = np.asarray(parents, dtype=np.int64)
    if n_cells and n_mut:
        # Seed each cell's tip, then push counts up the tree.  Nodes are
        # created in increasing index order with parent index < child index,
        # so a single descending sweep accumulates all ancestors.
        tips = np.array([c for c in cells if c >= 0], dtype=np.int64)
        if tips.size:
            np.add.at(counts, tips, 1)
        for j in range(n_mut - 1, -1, -1):
            p = parent_arr[j]
            if p >= 0:
                counts[p] += counts[j]

    return GenealogyResult(
        n_cells=n_cells,
        mutation_parent=parent_arr,
        mutation_count=counts,
        time=t,
    )
