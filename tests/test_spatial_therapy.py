"""Spatial growth, resistance, adaptive therapy and prevention."""

from __future__ import annotations

import numpy as np
import pytest

from oncosim.evolution import BirthDeath
from oncosim.spatial import SpatialModel, simulate_spatial
from oncosim.therapy import (
    AdaptiveProtocol,
    CompetitionModel,
    ResistanceModel,
    ScreeningProgramme,
    cessation_benefit,
    compare_strategies,
    doll_peto_lung_incidence,
    expected_multi_resistant_cells,
    max_curable_size,
    multistage_cumulative_risk,
    multistage_incidence,
    optimal_screening_interval,
    scan_containment_threshold,
    simulate_treatment,
    smoking_mutation_burden,
    stage_blocking_benefit,
)


# ----------------------------------------------------------------------
# spatial growth
# ----------------------------------------------------------------------
def test_surface_growth_produces_a_compact_sphere():
    """Radius of gyration of a uniform ball is sqrt(3/5) R."""
    rng = np.random.default_rng(3)
    n = 12_000
    tumour = simulate_spatial(SpatialModel(1.0, 0.0, driver_rate=0.0), n, rng=rng)
    assert tumour.n_cells >= n
    radius = (3.0 * tumour.n_cells / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert tumour.radius_of_gyration() == pytest.approx(np.sqrt(0.6) * radius, rel=0.12)


def test_no_two_cells_share_a_lattice_site():
    rng = np.random.default_rng(4)
    tumour = simulate_spatial(SpatialModel(1.0, 0.3, driver_rate=1e-3), 4000, rng=rng)
    assert len({tuple(p) for p in tumour.positions}) == tumour.n_cells


def test_clone_bookkeeping_is_consistent():
    rng = np.random.default_rng(5)
    tumour = simulate_spatial(SpatialModel(1.0, 0.2, driver_rate=1e-3), 4000, rng=rng)
    assert tumour.clone_sizes().sum() == tumour.n_cells
    assert tumour.clone_parent.size == tumour.clone_drivers.size
    assert tumour.clone_parent[0] == -1
    # every non-founder clone carries exactly one more driver than its parent
    for c in range(1, tumour.clone_drivers.size):
        parent = tumour.clone_parent[c]
        assert tumour.clone_drivers[c] == tumour.clone_drivers[parent] + 1


def test_dispersal_makes_the_mass_less_compact():
    """Dispersal seeds microlesions, so the same cell count spreads further.

    This is the low-variance consequence of dispersal (roughly 9 standard
    errors at 6 replicates).  The heterogeneity reduction that motivates
    the model is real but only ~2 standard errors at this many replicates,
    so it lives in ``experiments/spatial_heterogeneity.py`` rather than
    being asserted here.
    """
    rng = np.random.default_rng(6)
    common = dict(driver_rate=0.01, selection_coefficient=0.0)
    without = SpatialModel(1.0, 0.5, **common)
    with_dispersal = SpatialModel(
        1.0, 0.5, dispersal_probability=0.05, dispersal_radius=6, **common
    )
    reps = 6
    rg_without = np.mean(
        [simulate_spatial(without, 5000, rng=rng).radius_of_gyration() for _ in range(reps)]
    )
    rg_with = np.mean(
        [
            simulate_spatial(with_dispersal, 5000, rng=rng).radius_of_gyration()
            for _ in range(reps)
        ]
    )
    assert rg_with > rg_without * 1.05


def test_lattice_grows_automatically_for_strongly_dispersing_tumours():
    """Regression: the size heuristic used to abort at high turnover."""
    rng = np.random.default_rng(9)
    model = SpatialModel(
        1.0, 0.8, driver_rate=0.0, dispersal_probability=0.05, dispersal_radius=6
    )
    assert simulate_spatial(model, 4000, rng=rng).n_cells >= 4000


def test_undersized_lattice_is_reported_not_silently_wrapped():
    rng = np.random.default_rng(10)
    with pytest.raises(RuntimeError, match="lattice"):
        simulate_spatial(
            SpatialModel(1.0, 0.0, driver_rate=0.0), 5000, rng=rng, lattice_size=15
        )


def test_biopsy_never_exceeds_the_whole_tumour():
    rng = np.random.default_rng(7)
    tumour = simulate_spatial(SpatialModel(1.0, 0.2, driver_rate=1e-3), 4000, rng=rng)
    sample = tumour.biopsy(radius=6.0, rng=rng)
    assert sample.sum() <= tumour.n_cells
    assert np.all(sample <= tumour.clone_sizes())


# ----------------------------------------------------------------------
# resistance
# ----------------------------------------------------------------------
def test_probability_of_no_resistance_is_exp_minus_um():
    model = ResistanceModel(1e7, 1e-7, BirthDeath(1.0, 0.5))
    assert model.probability_no_resistance() == pytest.approx(np.exp(-1.0))
    # turnover cancels: 1/beta more divisions, beta lower lineage survival
    assert ResistanceModel(
        1e7, 1e-7, BirthDeath(1.0, 0.0)
    ).probability_no_resistance() == pytest.approx(model.probability_no_resistance())


def test_resistant_counts_are_heavy_tailed():
    rng = np.random.default_rng(8)
    model = ResistanceModel(1e9, 1e-7, BirthDeath(1.0, 0.5))
    q = model.resistance_quantiles((0.5, 0.9, 0.99), n_sim=20_000, rng=rng)
    assert q[0.5] < q[0.9] < q[0.99]
    # a heavy tail means the 99th percentile dwarfs the median
    assert q[0.99] > 5.0 * q[0.5]


def test_combination_therapy_extends_the_curable_range():
    one = max_curable_size([1e-7])
    two = max_curable_size([1e-7, 1e-7])
    three = max_curable_size([1e-7] * 3)
    assert one < two < three
    # a single drug already fails far below the imaging detection limit
    assert one < 1e8


def test_multi_resistant_count_grows_with_tumour_size():
    assert expected_multi_resistant_cells(
        1e12, [1e-7, 1e-7]
    ) > expected_multi_resistant_cells(1e6, [1e-7, 1e-7])
    assert expected_multi_resistant_cells(1e9, [1e-7]) == pytest.approx(1e2)


# ----------------------------------------------------------------------
# adaptive therapy
# ----------------------------------------------------------------------
def test_containment_beats_mtd_near_carrying_capacity():
    results = compare_strategies(CompetitionModel())
    assert results["mtd"].time_to_progression > results["none"].time_to_progression
    assert results["adaptive"].time_to_progression > results["mtd"].time_to_progression
    assert (
        results["containment"].time_to_progression
        > results["adaptive"].time_to_progression
    )


def test_containment_uses_less_drug_than_mtd():
    results = compare_strategies(CompetitionModel())
    assert results["mtd"].treatment_fraction == pytest.approx(1.0, abs=0.02)
    assert results["containment"].treatment_fraction < 0.9


def test_competition_is_required_for_containment_to_help():
    """Far below carrying capacity there is nothing to contain."""
    model = CompetitionModel(carrying_capacity=1e11)  # burden is now 0.08 K
    assert model.competition_strength(8e9) < 0.1
    results = compare_strategies(model)
    assert results["containment"].time_to_progression <= results["mtd"].time_to_progression


def test_resistance_cost_lengthens_time_to_progression():
    free = CompetitionModel(growth_resistant=0.01)
    costly = CompetitionModel(growth_resistant=0.007)
    assert free.resistance_cost == pytest.approx(0.0)
    assert costly.resistance_cost == pytest.approx(0.3)
    a = simulate_treatment(free, AdaptiveProtocol("containment"))
    b = simulate_treatment(costly, AdaptiveProtocol("containment"))
    assert b.time_to_progression > a.time_to_progression


def test_untreated_tumour_progresses_fastest():
    results = compare_strategies(CompetitionModel())
    assert results["none"].time_to_progression == min(
        r.time_to_progression for r in results.values()
    )


def test_containment_scan_is_monotone_in_the_useful_range():
    scan = scan_containment_threshold(CompetitionModel())
    ttp = scan["time_to_progression"][np.isfinite(scan["time_to_progression"])]
    # holding a higher burden preserves more sensitive competitors
    assert ttp[-1] > ttp[0]


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError):
        AdaptiveProtocol("aggressive")


# ----------------------------------------------------------------------
# prevention
# ----------------------------------------------------------------------
def test_armitage_doll_log_log_slope_is_k_minus_one():
    for k in (3, 5, 7):
        ages = np.array([40.0, 80.0])
        inc = multistage_incidence(ages, n_stages=k)
        slope = np.diff(np.log(inc))[0] / np.diff(np.log(ages))[0]
        assert slope == pytest.approx(k - 1, rel=1e-9)


def test_cumulative_risk_is_a_probability():
    for age in (20.0, 50.0, 90.0):
        assert 0.0 <= multistage_cumulative_risk(age, 6) <= 1.0
    assert multistage_cumulative_risk(90.0, 6) > multistage_cumulative_risk(50.0, 6)


def test_blocking_a_stage_reduces_risk():
    out = stage_blocking_benefit(70.0, n_stages=6, blocked_fraction=0.5)
    assert out["reduced_risk"] < out["baseline_risk"]
    assert 0.0 < out["relative_reduction"] < 1.0
    assert out["hazard_ratio"] == pytest.approx(0.5)


def test_doll_peto_scales_as_duration_to_the_four_point_five():
    a = doll_peto_lung_incidence(37.5, 20.0, age_started=17.5)  # 20 years
    b = doll_peto_lung_incidence(57.5, 20.0, age_started=17.5)  # 40 years
    assert float(b / a) == pytest.approx(2.0**4.5, rel=1e-9)


def test_duration_matters_more_than_dose():
    """Half the cigarettes for twice as long is worse, not better."""
    heavy_short = doll_peto_lung_incidence(37.5, 40.0, age_started=17.5)
    light_long = doll_peto_lung_incidence(57.5, 20.0, age_started=17.5)
    assert float(light_long) > float(heavy_short)


def test_quitting_earlier_avoids_more_risk():
    risks = [cessation_benefit(a)["risk_if_quit"] for a in (30.0, 40.0, 50.0, 60.0)]
    assert risks == sorted(risks)
    assert all(r < cessation_benefit(60.0)["risk_if_continue"] for r in risks)


def test_tobacco_burden_matches_the_published_per_pack_year_rate():
    out = smoking_mutation_burden(30.0, "lung", age=60.0)
    assert out["excess_mutations_per_cell"] == pytest.approx(150.0 * 30.0)
    assert out["fold_increase"] > 1.0
    with pytest.raises(ValueError):
        smoking_mutation_burden(10.0, "pancreas")


def test_screening_detection_saturates_below_the_sojourn_time():
    fractions = [
        ScreeningProgramme(3.0, 0.85, d).screen_detected_fraction()
        for d in (0.25, 0.5, 1.0, 3.0, 6.0)
    ]
    assert fractions == sorted(fractions, reverse=True)
    # halving the interval well below the sojourn buys almost nothing
    assert fractions[0] - fractions[1] < 0.5 * (fractions[2] - fractions[3])
    # Repeated screening can exceed the single-test sensitivity: a cancer
    # missed once is caught at the next screen as long as it has not gone
    # clinical.  The bound is 1, not the per-test sensitivity.
    assert fractions[0] > 0.85
    assert all(f <= 1.0 + 1e-12 for f in fractions)


def test_screening_fractions_sum_to_one():
    prog = ScreeningProgramme(3.0, 0.85, 2.0)
    assert prog.screen_detected_fraction() + prog.interval_cancer_fraction() == (
        pytest.approx(1.0)
    )
    assert prog.mean_lead_time() == pytest.approx(3.0)


def test_optimal_interval_is_interior():
    out = optimal_screening_interval(
        3.0, 0.85, cost_per_screen=1.0, value_per_detection=100.0
    )
    assert out["intervals"][0] < out["optimal_interval"] < out["intervals"][-1]


def test_screening_parameters_are_validated():
    with pytest.raises(ValueError):
        ScreeningProgramme(mean_sojourn_time=-1.0)
    with pytest.raises(ValueError):
        ScreeningProgramme(sensitivity=1.5)
