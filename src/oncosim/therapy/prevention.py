"""Prevention: carcinogen exposure, cessation and screening.

Prevention is where theoretical cancer modelling has its largest expected
value, and it rests on three quantitative pillars.

**1. Multistage kinetics.**  Armitage and Doll (1954) observed that
age-specific incidence for most epithelial cancers rises as a power of
age, ``I(t) ~ t^(k-1)``, and read off ``k`` -- the number of rate-limiting
steps -- as 5 to 7.  The power law is what makes prevention leveraged:
slowing *one* step out of ``k`` divides the hazard proportionally at every
age.  :func:`multistage_incidence` and :func:`stage_blocking_benefit`
make that arithmetic explicit.

**2. Exposure writes mutations, and the count is measurable.**
Alexandrov et al. (2016) quantified the somatic mutation burden per year
of smoking one pack a day, tissue by tissue -- 150 mutations per lung
cell per year, 18 per bladder cell, 6 per liver cell.  This converts a
behavioural exposure directly into the ``mu`` that the evolution models
consume, and it explains why the tissues with the largest excess burden
are the ones with the largest excess risk.

**3. Screening buys lead time, not certainty.**  With an exponentially
distributed preclinical sojourn, the fraction of cancers caught by
screening has a closed form in the screening interval and test
sensitivity.  Shortening the interval has sharply diminishing returns
once it drops well below the mean sojourn time --
:func:`optimal_screening_interval` locates the point where extra screens
stop buying detection.

References
----------
Armitage & Doll (1954) Br. J. Cancer 8:1-12.
Doll & Peto (1978) J. Epidemiol. Community Health 32:303-313.
Peto et al. (2000) BMJ 321:323-329 (benefits of stopping smoking).
Alexandrov et al. (2016) Science 354:618-622 (mutational burden of tobacco).
Zelen & Feinleib (1969) Biometrika 56:601-614 (screening lead time).
Duffy et al. (2008) Stat. Methods Med. Res. 17:333-345.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = [
    "multistage_incidence",
    "multistage_cumulative_risk",
    "stage_blocking_benefit",
    "TOBACCO_MUTATIONS_PER_PACK_YEAR",
    "smoking_mutation_burden",
    "doll_peto_lung_incidence",
    "cessation_benefit",
    "ScreeningProgramme",
    "optimal_screening_interval",
]


# ----------------------------------------------------------------------
# multistage carcinogenesis
# ----------------------------------------------------------------------
def multistage_incidence(
    age: float | np.ndarray,
    n_stages: int = 6,
    stage_rate: float = 1e-3,
    n_cells: float = 1e8,
) -> np.ndarray:
    """Armitage-Doll age-specific incidence ``I(t)``.

    With ``k`` rate-limiting steps each occurring at rate ``u`` per cell per
    year, and ``N`` cells at risk,

    .. math::  I(t) = N\\,\\frac{u^k\\,t^{k-1}}{(k-1)!}

    Returns incidence per person per year.  The characteristic signature
    is a straight line of slope ``k-1`` on a log-log plot of incidence
    against age -- the observation that motivated the model and still the
    first thing to check against registry data.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1")
    if stage_rate <= 0 or n_cells <= 0:
        raise ValueError("stage_rate and n_cells must be positive")
    t = np.asarray(age, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("age must be non-negative")
    k = n_stages
    log_i = (
        np.log(n_cells)
        + k * np.log(stage_rate)
        + (k - 1) * np.log(np.maximum(t, 1e-12))
        - gammaln(k)
    )
    return np.exp(log_i)


def multistage_cumulative_risk(
    age: float,
    n_stages: int = 6,
    stage_rate: float = 1e-3,
    n_cells: float = 1e8,
) -> float:
    """Cumulative probability of at least one malignant cell by ``age``.

    Integrates the hazard and applies ``1 - exp(-H)``, which is the correct
    conversion; treating incidence as a probability directly overstates
    risk badly once cumulative hazard approaches 1.
    """
    k = n_stages
    log_h = (
        np.log(n_cells)
        + k * np.log(stage_rate)
        + k * np.log(max(age, 1e-12))
        - gammaln(k + 1)
    )
    return float(1.0 - np.exp(-np.exp(log_h)))


def stage_blocking_benefit(
    age: float,
    n_stages: int = 6,
    stage_rate: float = 1e-3,
    n_cells: float = 1e8,
    *,
    blocked_fraction: float = 0.5,
    n_blocked_stages: int = 1,
) -> dict[str, float]:
    """Risk reduction from slowing some rate-limiting steps.

    A chemopreventive agent that cuts the rate of ``n_blocked_stages`` steps
    by ``blocked_fraction`` multiplies the hazard by
    ``(1 - blocked_fraction) ** n_blocked_stages``.  Because the hazard is a
    ``k``-fold product, blocking even one step yields a proportional
    reduction at every age -- but the *absolute* benefit is concentrated in
    the years when incidence is highest, i.e. late, which is why
    prevention trials need long follow-up to show anything.
    """
    if not 0.0 <= blocked_fraction < 1.0:
        raise ValueError("blocked_fraction must lie in [0, 1)")
    if n_blocked_stages > n_stages:
        raise ValueError("cannot block more stages than exist")
    factor = (1.0 - blocked_fraction) ** n_blocked_stages
    baseline = multistage_cumulative_risk(age, n_stages, stage_rate, n_cells)
    reduced = multistage_cumulative_risk(age, n_stages, stage_rate, n_cells * factor)
    return {
        "baseline_risk": baseline,
        "reduced_risk": reduced,
        "hazard_ratio": factor,
        "absolute_reduction": baseline - reduced,
        "relative_reduction": (baseline - reduced) / baseline if baseline > 0 else 0.0,
    }


# ----------------------------------------------------------------------
# tobacco
# ----------------------------------------------------------------------
#: Somatic mutations per cell per year of smoking one pack per day, by
#: tissue (Alexandrov et al. 2016, Science 354:618-622).
TOBACCO_MUTATIONS_PER_PACK_YEAR: dict[str, float] = {
    "lung": 150.0,
    "larynx": 97.0,
    "pharynx": 39.0,
    "oral_cavity": 23.0,
    "bladder": 18.0,
    "liver": 6.0,
}


def smoking_mutation_burden(
    pack_years: float,
    tissue: str = "lung",
    *,
    baseline_per_year: float = 20.0,
    age: float | None = None,
) -> dict[str, float]:
    """Excess somatic mutation burden per cell attributable to smoking.

    Parameters
    ----------
    pack_years:
        Packs per day multiplied by years smoked.
    tissue:
        One of :data:`TOBACCO_MUTATIONS_PER_PACK_YEAR`.
    baseline_per_year:
        Age-related (clock-like, SBS1/SBS5) mutations per cell per year in
        the absence of smoking, for context.

    Returns the excess burden, the baseline burden at ``age`` if given, and
    their ratio -- the number that makes the exposure concrete: 30
    pack-years writes about 4500 extra mutations into every lung cell,
    several times the entire age-related burden at 60.
    """
    if tissue not in TOBACCO_MUTATIONS_PER_PACK_YEAR:
        raise ValueError(
            f"unknown tissue {tissue!r}; known: "
            f"{sorted(TOBACCO_MUTATIONS_PER_PACK_YEAR)}"
        )
    if pack_years < 0:
        raise ValueError("pack_years must be non-negative")
    excess = TOBACCO_MUTATIONS_PER_PACK_YEAR[tissue] * pack_years
    out = {
        "tissue_rate_per_pack_year": TOBACCO_MUTATIONS_PER_PACK_YEAR[tissue],
        "excess_mutations_per_cell": excess,
    }
    if age is not None:
        baseline = baseline_per_year * age
        out["baseline_mutations_per_cell"] = baseline
        out["fold_increase"] = (baseline + excess) / baseline if baseline > 0 else np.inf
    return out


def doll_peto_lung_incidence(
    age: float | np.ndarray,
    cigarettes_per_day: float,
    age_started: float = 17.5,
    *,
    age_quit: float | None = None,
) -> np.ndarray:
    """Doll-Peto lung cancer incidence, per person per year.

    .. math::

        I = 0.273\\times10^{-12}\\,(c + 6)^2\\,d^{4.5}

    with ``c`` cigarettes per day and ``d`` years of smoking.  The
    remarkable feature -- and the reason cessation works so well -- is that
    incidence scales with *duration* to the 4.5th power but with *dose*
    only quadratically.  Smoking half as much for twice as long is far
    worse than the reverse.

    If ``age_quit`` is given, the duration term stops accumulating at
    cessation, so a former smoker's hazard is frozen at the level reached
    at quitting while a continuing smoker's keeps climbing steeply.

    This is a deliberately crude treatment of cessation and it
    **overstates** the benefit: real ex-smokers keep ageing, so their
    absolute incidence continues to rise after quitting even as their
    excess relative risk declines.  Read the numbers as an upper bound on
    the benefit, and compare the *ordering* across quitting ages rather
    than the absolute risks.
    """
    if cigarettes_per_day < 0:
        raise ValueError("cigarettes_per_day must be non-negative")
    t = np.asarray(age, dtype=np.float64)
    stop = t if age_quit is None else np.minimum(t, age_quit)
    duration = np.maximum(stop - age_started, 0.0)
    return 0.273e-12 * (cigarettes_per_day + 6.0) ** 2 * duration**4.5


def cessation_benefit(
    age_quit: float,
    cigarettes_per_day: float = 20.0,
    *,
    age_started: float = 17.5,
    age_evaluated: float = 75.0,
    n_grid: int = 4000,
) -> dict[str, float]:
    """Cumulative lung cancer risk to ``age_evaluated`` with and without quitting.

    Integrates the Doll-Peto hazard and converts to a probability via
    ``1 - exp(-H)``.

    The qualitative ordering matches Peto et al. (2000) -- quitting in the
    thirties avoids nearly all of the excess risk, quitting at fifty avoids
    a large part of it, quitting at sixty much less -- but because
    :func:`doll_peto_lung_incidence` freezes the hazard at cessation, the
    computed benefit is an upper bound.  Peto et al. found stopping at 50
    roughly *halved* the risk; this model gives a considerably larger
    reduction for that age.
    """
    if age_evaluated <= age_started:
        raise ValueError("age_evaluated must exceed age_started")
    ages = np.linspace(age_started, age_evaluated, n_grid)
    h_continue = np.trapezoid(
        doll_peto_lung_incidence(ages, cigarettes_per_day, age_started), ages
    )
    h_quit = np.trapezoid(
        doll_peto_lung_incidence(ages, cigarettes_per_day, age_started, age_quit=age_quit),
        ages,
    )
    risk_continue = 1.0 - np.exp(-h_continue)
    risk_quit = 1.0 - np.exp(-h_quit)
    return {
        "age_quit": float(age_quit),
        "risk_if_continue": float(risk_continue),
        "risk_if_quit": float(risk_quit),
        "absolute_risk_avoided": float(risk_continue - risk_quit),
        "relative_risk": float(risk_quit / risk_continue) if risk_continue > 0 else 0.0,
    }


# ----------------------------------------------------------------------
# screening
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ScreeningProgramme:
    """Screening with an exponentially distributed preclinical sojourn.

    Attributes
    ----------
    mean_sojourn_time:
        Average duration of the preclinical screen-detectable phase, in
        years.  Estimates: ~2-4 years for breast cancer, ~3-5 for
        colorectal adenoma-carcinoma progression, well under a year for
        aggressive subtypes -- which is exactly why interval cancers are
        enriched for aggressive biology.
    sensitivity:
        Probability that a screen detects a preclinical cancer that is
        present.
    interval:
        Years between screens.
    """

    mean_sojourn_time: float = 3.0
    sensitivity: float = 0.85
    interval: float = 2.0

    def __post_init__(self) -> None:
        if self.mean_sojourn_time <= 0:
            raise ValueError("mean_sojourn_time must be positive")
        if not 0.0 < self.sensitivity <= 1.0:
            raise ValueError("sensitivity must lie in (0, 1]")
        if self.interval <= 0:
            raise ValueError("interval must be positive")

    @property
    def _lambda(self) -> float:
        return 1.0 / self.mean_sojourn_time

    def screen_detected_fraction(self) -> float:
        """Fraction of cancers caught by screening rather than symptoms.

        .. math::

            P = \\frac{\\beta\\,(1 - e^{-\\lambda\\Delta})}
                     {\\lambda\\Delta\\,\\left(1 - (1-\\beta)e^{-\\lambda\\Delta}\\right)}

        Derivation: a case entering the detectable phase sees the next
        screen after ``U ~ Uniform(0, interval)`` and turns clinical after
        ``Exp(lambda)``; summing over successive screens that the test may
        miss gives the geometric factor in the denominator.

        Note that this can exceed the single-test ``sensitivity``: a cancer
        missed at one screen is still caught at the next, as long as it has
        not gone clinical.  The bound is 1, not ``sensitivity``.
        """
        lam, d, beta = self._lambda, self.interval, self.sensitivity
        x = lam * d
        first = (1.0 - np.exp(-x)) / x
        return float(beta * first / (1.0 - (1.0 - beta) * np.exp(-x)))

    def interval_cancer_fraction(self) -> float:
        """Fraction presenting symptomatically between screens."""
        return 1.0 - self.screen_detected_fraction()

    def mean_lead_time(self) -> float:
        """Average time by which screening advances diagnosis, in years.

        Conditional on screen detection, the lead time is the remaining
        sojourn, which for an exponential sojourn is memoryless and hence
        has mean ``mean_sojourn_time``.  The apparent survival benefit this
        creates in *uncorrected* comparisons is lead-time bias -- the
        single largest trap in evaluating a screening programme.
        """
        return self.mean_sojourn_time

    def screens_per_lifetime(
        self, start_age: float = 50.0, end_age: float = 74.0
    ) -> float:
        if end_age <= start_age:
            raise ValueError("end_age must exceed start_age")
        return (end_age - start_age) / self.interval

    def summary(
        self, start_age: float = 50.0, end_age: float = 74.0
    ) -> dict[str, float]:
        n = self.screens_per_lifetime(start_age, end_age)
        detected = self.screen_detected_fraction()
        return {
            "interval_years": self.interval,
            "screen_detected_fraction": detected,
            "interval_cancer_fraction": 1.0 - detected,
            "screens_per_person": n,
            "detected_per_screen": detected / n if n > 0 else 0.0,
        }


def optimal_screening_interval(
    mean_sojourn_time: float = 3.0,
    sensitivity: float = 0.85,
    *,
    cost_per_screen: float = 1.0,
    value_per_detection: float = 100.0,
    intervals=None,
    start_age: float = 50.0,
    end_age: float = 74.0,
) -> dict[str, np.ndarray | float]:
    """Sweep screening intervals and locate the net-benefit optimum.

    Net benefit is ``value_per_detection * screen_detected_fraction -
    cost_per_screen * screens_per_person``.  The units are deliberately
    abstract: the point is the *shape* of the trade-off, not a health
    economic claim, and the location of the optimum depends entirely on
    the ``value_per_detection / cost_per_screen`` ratio you supply.

    What is robust, and independent of that ratio, is the shape: detection
    rises steeply as the interval shortens towards the mean sojourn time
    and then saturates, while the number of screens keeps growing as
    ``1/interval``.  So there is always an interior optimum, and pushing
    the interval far below the sojourn time buys almost no extra detection
    at steeply rising cost.
    """
    if intervals is None:
        intervals = np.linspace(0.25, 10.0, 60)
    intervals = np.asarray(intervals, dtype=np.float64)
    detected = np.empty(intervals.size)
    n_screens = np.empty(intervals.size)
    for i, d in enumerate(intervals):
        prog = ScreeningProgramme(mean_sojourn_time, sensitivity, float(d))
        detected[i] = prog.screen_detected_fraction()
        n_screens[i] = prog.screens_per_lifetime(start_age, end_age)
    net = value_per_detection * detected - cost_per_screen * n_screens
    best = int(np.argmax(net))
    return {
        "intervals": intervals,
        "screen_detected_fraction": detected,
        "screens_per_person": n_screens,
        "net_benefit": net,
        "optimal_interval": float(intervals[best]),
        "optimal_detected_fraction": float(detected[best]),
    }
