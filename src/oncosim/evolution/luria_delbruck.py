"""Luria-Delbruck fluctuation theory for pre-existing resistance.

Why this matters for cancer
---------------------------
"Does resistance to a targeted drug arise *because of* the drug, or does
it pre-exist?"  is the same question Luria and Delbruck (1943) asked about
phage resistance in bacteria.  The answer -- resistant cells are thrown
off continuously during growth, so their number is a heavy-tailed random
variable, not a Poisson one -- is the quantitative backbone of every
statement about why single-agent targeted therapy relapses (Bozic et al.
2013; Diaz et al. 2012).

Exact structure used here
-------------------------
Let a wild-type population grow deterministically to final size ``N``,
let each division produce a mutant with probability ``mu``, and let each
mutant clone then grow as a Yule (pure-birth) process with growth rate
``rho`` times the wild-type rate.  Write ``m = mu * N`` for the expected
number of mutational *events*.

A mutation event that happens when the population has reached a fraction
``u`` of its final size founds a clone whose final size is geometric with
success parameter ``u**rho``.  Because ``dN/N = du``, the founding times
are uniform in ``u``.  Marginalising ``u`` gives a clone-size law that is
available in closed form:

.. math::

    q_k \\;=\\; \\int_0^1 u^{\\rho}\\,(1-u^{\\rho})^{k-1}\\,du
        \\;=\\; \\frac{1}{\\rho}\\,B\\!\\left(1+\\tfrac{1}{\\rho},\\,k\\right),
    \\qquad k \\ge 1 .

For neutral mutants (``rho = 1``) this collapses to the classical

.. math::  q_k = \\frac{1}{k(k+1)},

a proper probability distribution with **infinite mean** -- the origin of
the notorious "jackpot" events.  The total mutant count is therefore an
exact compound Poisson variable

.. math::  X = \\sum_{i=1}^{\\mathrm{Poisson}(m)} S_i,\\qquad S_i \\sim q .

Three consequences are exploited below:

1. **O(1) exact sampling.**  Draw ``u ~ U(0,1)``, then ``S ~ Geom(u**rho)``.
   No cell-by-cell simulation is needed, so a fluctuation assay with
   ``N = 1e12`` costs the same as one with ``N = 1e6``.
2. **Exact pmf by Panjer recursion.**  The compound-Poisson structure
   yields ``p_n = (m/n) * sum_k k q_k p_{n-k}``, which for ``rho = 1`` is
   exactly the Ma-Sandri-Sarkar recursion
   ``p_n = (m/n) sum_{i<n} p_i/(n-i+1)``.  Cost ``O(K^2)``.
3. **Exact pmf by FFT.**  Inverting the generating function on a damped
   contour gives the same pmf in ``O(K log K)``; for ``rho = 1``,
   ``G(z) = exp(m (1-z) ln(1-z) / z)``.

Tail: ``p_k ~ m * q_k ~ const * k**-(1 + 1/rho)``.  Neutral mutants give the
familiar ``1/k^2``; *fitter* mutants give a heavier tail (exponent tends to
1), which is why a resistant clone with a growth advantage makes relapse
timing far less predictable.

References
----------
Luria & Delbruck (1943) Genetics 28:491-511.
Lea & Coulson (1949) J. Genet. 49:264-285.
Ma, Sandri & Sarkar (1992) J. Appl. Prob. 29:255-267.
Zheng, Q. (1999) Math. Biosci. 162:1-32 (review of LD distributions).
Bozic et al. (2013) eLife 2:e00747 (targeted combination therapy).
Abate & Whitt (1992) Queueing Syst. 10:5-88 (damped-contour FFT inversion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln

__all__ = [
    "clone_size_pmf",
    "clone_size_mean_truncated",
    "ld_pmf",
    "ld_pmf_fft",
    "ld_cdf",
    "ld_sample",
    "ld_tail_exponent",
    "estimate_m_p0",
    "estimate_m_mle",
    "FluctuationAssay",
]


# ----------------------------------------------------------------------
# clone-size law
# ----------------------------------------------------------------------
def clone_size_pmf(kmax: int, rho: float = 1.0) -> np.ndarray:
    """``q_k`` for ``k = 1..kmax`` -- final size of one mutant clone.

    Parameters
    ----------
    kmax:
        Largest clone size to tabulate.
    rho:
        Growth rate of mutants relative to wild type.  ``rho = 1`` is the
        neutral (classical) case; ``rho > 1`` a fitter resistant clone.

    Returns
    -------
    Array of length ``kmax`` with ``q[0] = q_1``.  The tabulation is a
    truncation of a proper distribution, so it sums to slightly less
    than 1 (the deficit is the ``k > kmax`` tail).
    """
    if kmax < 1:
        raise ValueError("kmax must be >= 1")
    if rho <= 0:
        raise ValueError("rho must be positive")
    k = np.arange(1, kmax + 1, dtype=np.float64)
    a = 1.0 + 1.0 / rho
    # q_k = (1/rho) * B(a, k), computed in log space for stability
    return np.exp(-np.log(rho) + betaln(a, k))


def clone_size_mean_truncated(kmax: int, rho: float = 1.0) -> float:
    """Mean clone size when sizes are truncated at ``kmax``.

    The untruncated mean diverges for ``rho >= 1``; this reports the
    truncated mean, which grows like ``log(kmax)`` at ``rho = 1``.  It is
    the honest way to quote "expected number of resistant cells" when a
    finite detection limit exists.
    """
    q = clone_size_pmf(kmax, rho)
    k = np.arange(1, kmax + 1, dtype=np.float64)
    return float(np.sum(k * q))


def ld_tail_exponent(rho: float = 1.0) -> float:
    """Power-law exponent of ``P(X = k)`` at large ``k``: ``1 + 1/rho``."""
    if rho <= 0:
        raise ValueError("rho must be positive")
    return 1.0 + 1.0 / rho


# ----------------------------------------------------------------------
# exact pmf -- Panjer recursion
# ----------------------------------------------------------------------
def ld_pmf(m: float, kmax: int, rho: float = 1.0) -> np.ndarray:
    """Exact pmf ``P(X = 0..kmax)`` by the Panjer/Ma-Sandri-Sarkar recursion.

    Parameters
    ----------
    m:
        Expected number of mutational events, ``m = mu * N``.
    kmax:
        Largest count to evaluate.
    rho:
        Relative growth rate of mutants (see :func:`clone_size_pmf`).

    Complexity ``O(kmax^2)``.  Use :func:`ld_pmf_fft` for large ``kmax``.

    Notes
    -----
    For a compound Poisson ``X = sum_{i<=Pois(m)} S_i`` with ``S >= 1``,

    ``p_0 = exp(-m)``  and  ``p_n = (m/n) * sum_{k=1..n} k q_k p_{n-k}``.

    Substituting ``q_k = 1/(k(k+1))`` (the ``rho = 1`` case) reproduces the
    textbook form ``p_n = (m/n) sum_{i=0..n-1} p_i/(n-i+1)``.
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    if kmax < 0:
        raise ValueError("kmax must be >= 0")
    p = np.zeros(kmax + 1, dtype=np.float64)
    p[0] = np.exp(-m)
    if kmax == 0 or m == 0.0:
        return p
    q = clone_size_pmf(kmax, rho)
    kq = np.arange(1, kmax + 1, dtype=np.float64) * q  # k * q_k
    for n in range(1, kmax + 1):
        # sum_{k=1..n} k q_k p_{n-k}
        p[n] = (m / n) * float(np.dot(kq[:n], p[n - 1 :: -1]))
    return p


def ld_cdf(m: float, kmax: int, rho: float = 1.0) -> np.ndarray:
    """Cumulative distribution ``P(X <= 0..kmax)``."""
    return np.cumsum(ld_pmf(m, kmax, rho))


# ----------------------------------------------------------------------
# exact pmf -- damped-contour FFT inversion
# ----------------------------------------------------------------------
def ld_pmf_fft(
    m: float,
    kmax: int,
    rho: float = 1.0,
    *,
    n_fft: int | None = None,
    radius: float | None = None,
) -> np.ndarray:
    """Exact pmf by inverting the generating function with an FFT.

    ``O(K log K)`` instead of the recursion's ``O(K^2)``.  The generating
    function of the compound Poisson law is ``G(z) = exp(m (Q(z) - 1))``
    with ``Q`` the clone-size generating function; for ``rho = 1``,
    ``Q(z) - 1 = (1-z) log(1-z) / z``, evaluated in closed form.  For
    general ``rho`` the coefficients ``q_k`` are tabulated once and
    transformed.

    Aliasing is suppressed by evaluating on a circle of radius
    ``radius < 1`` (Abate-Whitt).  The default radius trades wrap-around
    error against round-off amplification and reaches ~1e-9 absolute
    accuracy.

    Parameters
    ----------
    n_fft:
        Transform length; defaults to the next power of two above
        ``4 * (kmax + 1)``, which keeps the damping mild.
    radius:
        Contour radius in ``(0, 1)``.  Defaults to ``eps ** (1/(2*n_fft))``.
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    if kmax < 0:
        raise ValueError("kmax must be >= 0")
    if rho <= 0:
        raise ValueError("rho must be positive")

    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(8, 4 * (kmax + 1)))))
    if radius is None:
        radius = float(np.finfo(np.float64).eps ** (1.0 / (2.0 * n_fft)))
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must lie strictly in (0, 1)")

    # Contour points z_j = radius * exp(2*pi*i*j/N).  With numpy's
    # conventions, evaluating a coefficient series on this contour is one
    # inverse transform, and recovering coefficients from contour values is
    # one forward transform:
    #     sum_k c_k z_j^k        = N * ifft(c_k * radius^k)[j]
    #     c_n * radius^n         = fft(G(z_j))[n] / N
    j = np.arange(n_fft)
    z = radius * np.exp(2j * np.pi * j / n_fft)

    if abs(rho - 1.0) < 1e-12:
        # Q(z) - 1 = (1 - z) log(1 - z) / z, analytic on |z| < 1 and
        # evaluated in closed form (no truncation error at all).
        qz_minus_1 = (1.0 - z) * np.log(1.0 - z) / z
    else:
        # Tabulate q_k out to the transform length; the neglected tail is
        # O(n_fft ** (-1/rho)) and additionally damped by radius**n_fft.
        q = clone_size_pmf(n_fft - 1, rho)
        damped = np.zeros(n_fft, dtype=np.complex128)
        kk = np.arange(1, n_fft, dtype=np.float64)
        damped[1:] = q * radius**kk
        qz = n_fft * np.fft.ifft(damped)
        qz_minus_1 = qz - 1.0

    gz = np.exp(m * qz_minus_1)
    coeffs = np.fft.fft(gz).real / n_fft
    n = np.arange(n_fft, dtype=np.float64)
    p_full = coeffs / radius**n
    return np.clip(p_full[: kmax + 1], 0.0, None)


# ----------------------------------------------------------------------
# exact sampling
# ----------------------------------------------------------------------
def ld_sample(
    m: float,
    size: int = 1,
    rho: float = 1.0,
    *,
    rng: np.random.Generator | None = None,
    death_rate: float = 0.0,
    birth_rate: float = 1.0,
) -> np.ndarray:
    """Draw exact Luria-Delbruck variates in O(number of mutation events).

    No cell-by-cell simulation: each mutation event contributes one
    geometric clone size whose parameter is drawn from the uniform
    founding-time law.  The cost is independent of the final population
    size ``N``, so ``m = mu*N`` with ``N = 1e12`` is as cheap as ``N = 1e3``.

    Parameters
    ----------
    m:
        Expected number of mutation events (``mu * N``).
    size:
        Number of independent cultures / tumours to simulate.
    rho:
        Mutant growth rate relative to wild type.
    death_rate, birth_rate:
        If ``death_rate > 0`` the mutant clones follow a linear
        birth-death process rather than a Yule process, so a clone can go
        extinct before detection.  Rates are in units of the wild-type
        division rate; the mutant net growth rate is
        ``rho * (birth_rate - death_rate)``.

    Returns
    -------
    Integer array of length ``size`` with the mutant count per culture.
    """
    if rng is None:
        rng = np.random.default_rng()
    if m < 0:
        raise ValueError("m must be non-negative")
    if rho <= 0:
        raise ValueError("rho must be positive")
    if death_rate < 0 or birth_rate <= 0:
        raise ValueError("birth_rate must be positive and death_rate non-negative")
    if death_rate >= birth_rate:
        raise ValueError("death_rate must be below birth_rate for a growing clone")

    counts = rng.poisson(m, size=size)
    total = int(counts.sum())
    out = np.zeros(size, dtype=np.int64)
    if total == 0:
        return out

    tiny = np.finfo(np.float64).tiny
    u = rng.random(total)
    if death_rate == 0.0:
        # Yule clone: size ~ Geometric(w) on {1,2,...} with w = u**rho
        w = u**rho
        sizes = rng.geometric(np.clip(w, tiny, 1.0))
    else:
        # Linear birth-death run for the remaining time, with the standard
        # extinction/geometric parameters (Kendall 1948).
        b, d = birth_rate, death_rate
        net = rho * (b - d)
        s = -np.log(np.clip(u, tiny, 1.0)) / net
        e = np.exp((b - d) * s)
        denom = b * e - d
        alpha = d * (e - 1.0) / denom  # P(extinct)
        beta = b * (e - 1.0) / denom
        alive = rng.random(total) >= alpha
        sizes = np.zeros(total, dtype=np.int64)
        if int(alive.sum()):
            sizes[alive] = rng.geometric(np.clip(1.0 - beta[alive], tiny, 1.0))

    # scatter-add clone sizes back to their cultures
    culture = np.repeat(np.arange(size), counts)
    np.add.at(out, culture, sizes)
    return out


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def estimate_m_p0(counts: np.ndarray) -> float:
    """P0 estimator ``m_hat = -log(fraction of cultures with zero mutants)``.

    Robust (it ignores jackpots entirely) but only usable when a decent
    fraction of cultures are mutant-free; returns ``inf`` otherwise.
    """
    counts = np.asarray(counts)
    frac0 = float(np.mean(counts == 0))
    if frac0 <= 0.0:
        return float("inf")
    return -float(np.log(frac0))


def estimate_m_mle(
    counts: np.ndarray,
    rho: float = 1.0,
    *,
    bracket: tuple[float, float] = (1e-4, 1e4),
    tol: float = 1e-6,
) -> float:
    """Maximum-likelihood estimate of ``m`` from observed mutant counts.

    Uses the exact pmf, so it is the Ma-Sandri-Sarkar MLE (the estimator
    that actually uses the jackpot information, unlike the P0 method).
    Optimises by golden-section search on the log-likelihood, which is
    unimodal in ``m`` for this family.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty 1-D array")
    kmax = int(counts.max())

    def neg_ll(m: float) -> float:
        p = np.clip(ld_pmf(m, kmax, rho), 1e-300, None)
        return -float(np.sum(np.log(p[counts])))

    lo, hi = bracket
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = np.log(lo), np.log(hi)
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = neg_ll(np.exp(c)), neg_ll(np.exp(d))
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = neg_ll(np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = neg_ll(np.exp(d))
    return float(np.exp(0.5 * (a + b)))


@dataclass(frozen=True)
class FluctuationAssay:
    """A fluctuation experiment: ``n_cultures`` grown to ``final_size``.

    Bundles the mutation rate per division ``mu`` with the population size
    so that the derived quantity ``m = mu * final_size`` -- the only thing
    the LD law depends on -- is computed consistently.
    """

    mu: float
    final_size: float
    n_cultures: int = 100
    rho: float = 1.0

    @property
    def m(self) -> float:
        return self.mu * self.final_size

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        return ld_sample(self.m, self.n_cultures, self.rho, rng=rng)

    def pmf(self, kmax: int) -> np.ndarray:
        return ld_pmf(self.m, kmax, self.rho)

    def probability_no_resistant_cell(self) -> float:
        """``P(X = 0)`` -- the chance a lesion harbours no resistant cell.

        This is ``exp(-mu * N)`` and is the quantity that decides whether a
        single-agent targeted therapy can cure: with ``mu ~ 1e-7`` per
        division per drug-resistance-conferring site and a detectable
        lesion of ``N ~ 1e9`` cells, it is ``exp(-100) ~ 0``.
        """
        return float(np.exp(-self.m))
