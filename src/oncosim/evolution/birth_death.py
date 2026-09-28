"""Exact theory of the linear birth-death process.

Every stochastic model of tumour initiation and growth in this package
reduces, locally, to a linear birth-death process: a cell divides at rate
``b``, dies at rate ``d``, and the net growth rate is ``lam = b - d``.
Kendall (1948) solved it exactly, and those closed forms are what the
simulators in :mod:`oncosim` are validated against.

Conventions
-----------
``b > d >= 0`` for a supercritical (growing) process.  Time is in units
where ``b`` and ``d`` are per-cell rates, so ``1/b`` is the mean cell-cycle
time.  Typical human tumour values: ``b`` corresponding to a 2-4 day
cycle, with ``d/b`` between 0 and ~0.99 -- the "turnover" ratio, which is
poorly constrained experimentally and is exactly the parameter the site
frequency spectrum is most informative about.

References
----------
Kendall, D.G. (1948) Ann. Math. Statist. 19:1-15.
Athreya & Ney (1972) *Branching Processes*, Springer.
Bozic et al. (2010) PNAS 107:18545-18550 (driver/passenger accumulation).
Durrett, R. (2015) *Branching Process Models of Cancer*, Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BirthDeath"]


@dataclass(frozen=True)
class BirthDeath:
    """Linear birth-death process with birth rate ``b`` and death rate ``d``."""

    b: float = 1.0
    d: float = 0.0

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("birth rate must be positive")
        if self.d < 0:
            raise ValueError("death rate must be non-negative")

    # ------------------------------------------------------------------
    @property
    def net_growth(self) -> float:
        """``lam = b - d``.  Negative or zero means the clone dies out."""
        return self.b - self.d

    @property
    def turnover(self) -> float:
        """``d / b`` -- the fraction of divisions cancelled by death."""
        return self.d / self.b

    @property
    def beta(self) -> float:
        """``(b - d) / b`` -- net cells added per division.

        This is the factor that converts a per-division mutation rate into
        the per-*net-cell* rate that bulk sequencing actually measures, and
        is why the "effective mutation rate" read off a 1/f tail is
        ``mu / beta``, not ``mu``.
        """
        return self.net_growth / self.b

    @property
    def extinction_probability(self) -> float:
        """``P(lineage from one cell eventually dies out)`` = ``min(1, d/b)``."""
        return min(1.0, self.d / self.b)

    # ------------------------------------------------------------------
    # distribution at time t, starting from one cell (Kendall 1948)
    # ------------------------------------------------------------------
    def _alpha_beta(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=np.float64)
        b, d = self.b, self.d
        lam = b - d
        if abs(lam) < 1e-15:
            # critical case: alpha = beta = b t / (1 + b t)
            x = b * t / (1.0 + b * t)
            return x, x
        e = np.exp(lam * t)
        denom = b * e - d
        alpha = d * (e - 1.0) / denom
        beta = b * (e - 1.0) / denom
        return alpha, beta

    def prob_extinct_by(self, t: float | np.ndarray) -> np.ndarray:
        """``P(N(t) = 0 | N(0) = 1)``."""
        alpha, _ = self._alpha_beta(t)
        return alpha

    def size_pmf(self, t: float, kmax: int) -> np.ndarray:
        """``P(N(t) = 0..kmax | N(0) = 1)`` -- modified geometric."""
        alpha, beta = self._alpha_beta(t)
        alpha, beta = float(alpha), float(beta)
        p = np.empty(kmax + 1, dtype=np.float64)
        p[0] = alpha
        k = np.arange(1, kmax + 1, dtype=np.float64)
        p[1:] = (1.0 - alpha) * (1.0 - beta) * beta ** (k - 1.0)
        return p

    def mean_size(self, t: float | np.ndarray) -> np.ndarray:
        """``E[N(t)] = exp(lam t)`` -- unconditional, including extinct lines."""
        return np.exp(self.net_growth * np.asarray(t, dtype=np.float64))

    def var_size(self, t: float | np.ndarray) -> np.ndarray:
        """``Var[N(t)]`` from one cell."""
        t = np.asarray(t, dtype=np.float64)
        b, d = self.b, self.d
        lam = b - d
        if abs(lam) < 1e-15:
            return 2.0 * b * t
        e = np.exp(lam * t)
        return ((b + d) / lam) * e * (e - 1.0)

    def mean_size_conditional(self, t: float | np.ndarray) -> np.ndarray:
        """``E[N(t) | N(t) > 0]``."""
        alpha, _ = self._alpha_beta(t)
        return self.mean_size(t) / np.clip(1.0 - alpha, 1e-300, None)

    def sample_size(
        self, t: float, size: int = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Exact draws of ``N(t)`` from one cell -- O(size), no time stepping."""
        if rng is None:
            rng = np.random.default_rng()
        alpha, beta = self._alpha_beta(t)
        alpha, beta = float(alpha), float(beta)
        out = np.zeros(size, dtype=np.int64)
        alive = rng.random(size) >= alpha
        n_alive = int(alive.sum())
        if n_alive:
            out[alive] = rng.geometric(
                max(1.0 - beta, np.finfo(float).tiny), size=n_alive
            )
        return out

    # ------------------------------------------------------------------
    # time to reach a size / number of divisions
    # ------------------------------------------------------------------
    def time_to_size(self, n: float) -> float:
        """Deterministic time for the mean to reach ``n`` cells from one cell."""
        if self.net_growth <= 0:
            return float("inf")
        return float(np.log(n) / self.net_growth)

    def expected_divisions_to_size(self, n: float) -> float:
        """Total cell divisions needed to *net* produce ``n`` cells.

        ``n / beta``: with turnover, most divisions only replace cells that
        died, and every one of them is a mutational opportunity.  This is
        the reason a slowly-turning-over tissue and a fast-turning-over one
        of the same size carry very different mutation burdens.
        """
        return float(n / self.beta)

    def limit_w_mean(self) -> float:
        """Mean of ``W = lim N(t) exp(-lam t)`` conditional on survival.

        ``W`` is 0 with probability ``d/b`` and otherwise exponential with
        mean ``1 / (1 - d/b)``.  The exponential spread of ``W`` is why two
        tumours with identical parameters can differ in size by an order of
        magnitude at the same age.
        """
        if self.net_growth <= 0:
            raise ValueError("W is degenerate for a (sub)critical process")
        return 1.0 / (1.0 - self.d / self.b)

    def sample_limit_w(
        self, size: int = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draws of the martingale limit ``W`` (zero for extinct lineages)."""
        if rng is None:
            rng = np.random.default_rng()
        w = rng.exponential(self.limit_w_mean(), size=size)
        w[rng.random(size) < self.extinction_probability] = 0.0
        return w
