"""Somatic evolution: every simulator against a closed form."""

from __future__ import annotations

import numpy as np
import pytest

from oncosim.evolution import (
    BirthDeath,
    DriverModel,
    clone_size_pmf,
    estimate_m_mle,
    estimate_m_p0,
    expected_mutations_above,
    fit_neutral_tail,
    ld_pmf,
    ld_pmf_fft,
    ld_sample,
    ld_tail_exponent,
    sample_sfs_above,
    simulate_genealogy_sfs,
    simulate_ssa,
    simulate_tau_leap,
    vaf_from_cell_fraction,
)


# ----------------------------------------------------------------------
# birth-death theory
# ----------------------------------------------------------------------
def test_extinction_probability_and_beta():
    proc = BirthDeath(1.0, 0.4)
    assert proc.extinction_probability == pytest.approx(0.4)
    assert proc.beta == pytest.approx(0.6)
    assert BirthDeath(1.0, 0.0).extinction_probability == 0.0


def test_size_pmf_is_a_distribution_and_matches_moments():
    proc = BirthDeath(1.0, 0.3)
    t = 2.0
    p = proc.size_pmf(t, 4000)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    k = np.arange(p.size)
    assert float(p @ k) == pytest.approx(float(proc.mean_size(t)), rel=1e-6)
    second = float(p @ (k.astype(float) ** 2))
    var = second - float(p @ k) ** 2
    assert var == pytest.approx(float(proc.var_size(t)), rel=1e-5)


def test_sampled_sizes_match_the_pmf():
    proc = BirthDeath(1.0, 0.5)
    rng = np.random.default_rng(0)
    draws = proc.sample_size(1.5, size=200_000, rng=rng)
    assert (draws == 0).mean() == pytest.approx(
        float(proc.prob_extinct_by(1.5)), abs=3e-3
    )
    assert draws.mean() == pytest.approx(float(proc.mean_size(1.5)), rel=0.03)


# ----------------------------------------------------------------------
# Luria-Delbruck
# ----------------------------------------------------------------------
def test_clone_size_law_reduces_to_the_classical_case():
    q = clone_size_pmf(2000, rho=1.0)
    k = np.arange(1, 2001, dtype=float)
    assert np.allclose(q, 1.0 / (k * (k + 1.0)), rtol=1e-10)
    assert q.sum() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("rho", [0.7, 1.0, 1.5])
@pytest.mark.parametrize("m", [0.5, 2.0, 8.0])
def test_fft_inversion_agrees_with_the_panjer_recursion(m, rho):
    a = ld_pmf(m, 300, rho)
    b = ld_pmf_fft(m, 300, rho)
    assert np.max(np.abs(a - b)) < 1e-10


def test_p0_is_exactly_exp_minus_m():
    for m in (0.1, 1.0, 5.0):
        assert ld_pmf(m, 3)[0] == pytest.approx(np.exp(-m), rel=1e-12)


def test_heavy_tail_exponent():
    """Neutral mutants give p_k ~ m/k^2; the exponent is 1 + 1/rho."""
    m = 4.0
    p = ld_pmf(m, 2000)
    k = np.arange(p.size, dtype=float)
    assert np.mean(k[1000:] ** 2 * p[1000:]) == pytest.approx(m, rel=0.1)
    assert ld_tail_exponent(1.0) == pytest.approx(2.0)
    assert ld_tail_exponent(2.0) == pytest.approx(1.5)


def test_exact_sampler_matches_the_pmf():
    rng = np.random.default_rng(1)
    m = 3.0
    draws = ld_sample(m, 400_000, rng=rng)
    empirical = np.bincount(np.clip(draws, 0, 20), minlength=21) / draws.size
    assert np.allclose(empirical[:8], ld_pmf(m, 20)[:8], atol=3e-3)


def test_sampler_cost_is_independent_of_population_size():
    """m = mu*N with N = 1e12 must be as cheap as N = 1e3."""
    rng = np.random.default_rng(2)
    small = ld_sample(1e-9 * 1e3, 1000, rng=rng)
    large = ld_sample(1e-9 * 1e12, 1000, rng=rng)
    assert small.sum() == 0  # m = 1e-6, essentially never a mutant
    assert large.mean() > 0  # m = 1e3, always many


def test_mle_recovers_the_mutation_rate():
    rng = np.random.default_rng(3)
    m_true = 2.5
    counts = ld_sample(m_true, 400, rng=rng)
    assert estimate_m_mle(counts) == pytest.approx(m_true, rel=0.25)
    assert estimate_m_p0(counts) == pytest.approx(m_true, rel=0.35)


# ----------------------------------------------------------------------
# site frequency spectrum
# ----------------------------------------------------------------------
def test_vaf_conversion():
    assert vaf_from_cell_fraction(0.5) == pytest.approx(0.25)
    assert vaf_from_cell_fraction(0.5, copy_number=4, mutant_copies=2) == pytest.approx(
        0.25
    )


@pytest.mark.parametrize("death", [0.0, 0.4, 0.8])
def test_genealogy_simulation_matches_the_one_over_f_law(death):
    """The brute-force mutation tree must reproduce M(f) = (mu/beta)(1/f - 1)."""
    rng = np.random.default_rng(11)
    proc = BirthDeath(1.0, death)
    # Counts above a frequency are strongly correlated within one tumour
    # (a single early clone contributes many mutations at similar
    # frequency), so the per-run coefficient of variation reaches ~0.22 at
    # f = 0.1.  40 replicates put the standard error of the mean near 3.4%,
    # which is what the tolerance below is sized against.
    mu, n, reps = 4.0, 15_000, 40
    fs = np.array([0.02, 0.05, 0.1])
    observed = np.zeros(fs.size)
    for _ in range(reps):
        result = simulate_genealogy_sfs(n, mu, proc, rng=rng)
        f = result.frequencies
        clonal = int((f >= 0.99).sum())
        observed += np.array([(f >= x).sum() - clonal for x in fs])
    observed /= reps
    theory = expected_mutations_above(fs, mu, proc)
    assert np.allclose(observed / theory, 1.0, atol=0.12)


def test_fast_sampler_agrees_with_the_analytic_expectation():
    rng = np.random.default_rng(12)
    proc = BirthDeath(1.0, 0.4)
    mu, f_min = 4.0, 0.05
    counts = [sample_sfs_above(mu, f_min, proc, rng=rng).size for _ in range(3000)]
    expected = float(expected_mutations_above(f_min, mu, proc))
    assert np.mean(counts) == pytest.approx(expected, rel=0.03)


def test_fast_sampler_frequencies_follow_the_inverse_square_density():
    rng = np.random.default_rng(13)
    freqs = sample_sfs_above(2000.0, 0.02, rng=rng)
    # cumulative count above f must scale as 1/f - 1
    for f in (0.05, 0.1, 0.2):
        observed = (freqs >= f).sum()
        predicted = (1.0 / f - 1.0) / (1.0 / 0.02 - 1.0) * freqs.size
        assert observed == pytest.approx(predicted, rel=0.08)


def test_tail_fit_recovers_a_neutral_exponent():
    rng = np.random.default_rng(14)
    freqs = sample_sfs_above(5000.0, 0.01, rng=rng)
    fit = fit_neutral_tail(freqs, 0.05, 0.4)
    assert fit.alpha == pytest.approx(1.0, abs=0.03)
    assert fit.neutral_consistent


@pytest.mark.parametrize("alpha_true", [0.5, 0.75, 1.0, 1.4])
def test_tail_fit_recovers_a_known_exponent(alpha_true):
    """MLE on a doubly-truncated power law must be unbiased.

    Guards against the log-log-regression formulation, which returned
    alpha = 1.22 for exactly neutral data because of the additive -1 in
    the cumulative spectrum.
    """
    rng = np.random.default_rng(15)
    f_min, f_max, n = 0.05, 0.4, 200_000
    gamma = alpha_true + 1.0
    # inverse CDF of a density proportional to f**-gamma on [f_min, f_max]
    u = rng.random(n)
    p = 1.0 - gamma
    freqs = (f_min**p + u * (f_max**p - f_min**p)) ** (1.0 / p)
    fit = fit_neutral_tail(freqs, f_min, f_max)
    assert fit.alpha == pytest.approx(alpha_true, abs=4.0 * fit.alpha_stderr)
    assert fit.alpha == pytest.approx(alpha_true, abs=0.02)


def test_tail_fit_needs_enough_mutations():
    with pytest.raises(ValueError):
        fit_neutral_tail(np.array([0.1, 0.2, 0.3]), 0.05, 0.4)


def test_genealogy_rejects_a_subcritical_process():
    with pytest.raises(ValueError):
        simulate_genealogy_sfs(100, 1.0, BirthDeath(1.0, 1.5))


# ----------------------------------------------------------------------
# clonal dynamics: SSA and tau-leaping
# ----------------------------------------------------------------------
def test_ssa_event_count_matches_the_exact_expectation():
    """Divisions minus deaths equals N-1, so events = (N-1)(b+d)/(b-d)."""
    rng = np.random.default_rng(20)
    b, d, n = 1.0, 0.3, 4000
    model = DriverModel(b, d, driver_rate=0.0, selection_coefficient=0.0)
    events = np.array(
        [simulate_ssa(model, n, rng=rng).n_events for _ in range(60)], dtype=float
    )
    theory = (n - 1) * (b + d) / (b - d)
    assert events.mean() == pytest.approx(theory, rel=0.02)


def test_tau_leap_agrees_with_ssa_on_time_to_target():
    rng = np.random.default_rng(21)
    model = DriverModel(1.0, 0.3, driver_rate=0.0, selection_coefficient=0.0)
    n = 3000
    exact = np.array([simulate_ssa(model, n, rng=rng).time for _ in range(60)])
    leaped = np.array([simulate_tau_leap(model, n, rng=rng).time for _ in range(60)])
    se = np.hypot(exact.std() / np.sqrt(exact.size), leaped.std() / np.sqrt(leaped.size))
    assert abs(exact.mean() - leaped.mean()) < 3.0 * se


def test_both_simulators_reach_the_target_and_stay_consistent():
    rng = np.random.default_rng(22)
    model = DriverModel(1.0, 0.2, driver_rate=2e-4, selection_coefficient=0.15)
    for state in (
        simulate_ssa(model, 8000, rng=rng),
        simulate_tau_leap(model, 8000, rng=rng),
    ):
        assert state.total_cells >= 8000
        assert state.n_clones >= 1
        assert state.mean_drivers() >= 0.0
        assert state.driver_distribution().sum() == state.total_cells
        assert state.clone_frequencies().sum() == pytest.approx(1.0, abs=1e-9)


def test_driver_selection_increases_the_driver_load():
    rng = np.random.default_rng(23)
    neutral = DriverModel(1.0, 0.2, driver_rate=1e-3, selection_coefficient=0.0)
    selected = DriverModel(1.0, 0.2, driver_rate=1e-3, selection_coefficient=0.5)
    loads = [
        np.mean(
            [simulate_tau_leap(m, 20_000, rng=rng).mean_drivers() for _ in range(8)]
        )
        for m in (neutral, selected)
    ]
    assert loads[1] > loads[0]


def test_driver_model_rejects_a_subcritical_process():
    with pytest.raises(ValueError):
        DriverModel(1.0, 1.2)
