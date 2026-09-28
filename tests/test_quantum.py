"""Quantum module: reproduce published values, and enforce detailed balance.

The G-C potential is reconstructed from Table I of Slocombe, Sacchi &
Al-Khalili (2022).  These tests check it against five numbers reported in
that paper that were *not* used to build it, which is what makes the
reconstruction verifiable rather than merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from oncosim.quantum import (
    AT_ILLUSTRATIVE,
    GC_SLOCOMBE_2022,
    DoubleMorsePotential,
    analyse_tautomer,
    barrier_action,
    crossover_temperature,
    kinetic_isotope_effect,
    rate_constants,
    solve_double_well,
    tautomer_occupancy_classical,
    tautomer_occupancy_quantum,
    thermal_tunnelling_factor,
)
from oncosim.quantum.units import DEUTERON_MASS_AU, HARTREE_EV, PROTON_MASS_AU

BODY_T = 310.15


# ----------------------------------------------------------------------
# reconstruction of the published surface
# ----------------------------------------------------------------------
def test_published_barrier_heights_and_asymmetry():
    s = GC_SLOCOMBE_2022.summary_ev()
    assert s["forward_barrier_eV"] == pytest.approx(0.705, abs=0.002)
    assert s["reverse_barrier_eV"] == pytest.approx(0.270, abs=0.002)
    assert s["asymmetry_eV"] == pytest.approx(0.435, abs=0.002)


def test_published_barrier_frequency():
    omega_b = GC_SLOCOMBE_2022.harmonic_frequency(GC_SLOCOMBE_2022.x_barrier)
    assert omega_b == pytest.approx(0.00277, rel=0.01)


def test_published_zero_point_energy():
    """E_0 = 0.049 eV above the potential minimum."""
    states = solve_double_well(GC_SLOCOMBE_2022, n_grid=900, n_states=20)
    assert states.energies_ev()[0] == pytest.approx(0.049, abs=0.002)


def test_seventh_eigenstate_is_the_first_tautomeric_one():
    """Slocombe et al. report exactly this; index 6 is the 7th state."""
    states = solve_double_well(GC_SLOCOMBE_2022, n_grid=900, n_states=30)
    assert states.first_tautomeric_state() == 6


def test_eigenvalues_are_converged_in_the_grid():
    coarse = solve_double_well(GC_SLOCOMBE_2022, n_grid=600, n_states=10)
    fine = solve_double_well(GC_SLOCOMBE_2022, n_grid=1400, n_states=10)
    assert np.allclose(coarse.energies_ev(), fine.energies_ev(), atol=1e-4)


def test_wavefunctions_are_normalised_and_ordered():
    states = solve_double_well(GC_SLOCOMBE_2022, n_grid=700, n_states=15)
    assert np.allclose((states.wavefunctions**2).sum(axis=0), 1.0, atol=1e-10)
    assert np.all(np.diff(states.energies) > 0)


# ----------------------------------------------------------------------
# physics that must hold regardless of parameters
# ----------------------------------------------------------------------
def test_tunnelling_cannot_shift_the_equilibrium():
    """kappa is identical forward and backward, so it cancels from K_eq."""
    kf = thermal_tunnelling_factor(GC_SLOCOMBE_2022, BODY_T, forward=True)
    kr = thermal_tunnelling_factor(GC_SLOCOMBE_2022, BODY_T, forward=False)
    assert kf == pytest.approx(kr, rel=1e-5)


def test_tunnelling_factor_exceeds_one_and_grows_as_temperature_falls():
    temps = [400.0, 310.0, 250.0, 200.0, 150.0]
    kappas = [thermal_tunnelling_factor(GC_SLOCOMBE_2022, t) for t in temps]
    assert all(k >= 1.0 for k in kappas)
    assert all(a < b for a, b in zip(kappas, kappas[1:]))


def test_body_temperature_is_far_above_the_crossover():
    """T_c = 139 K, so 310 K is thermally activated, not deep tunnelling."""
    assert crossover_temperature(GC_SLOCOMBE_2022) == pytest.approx(139.0, abs=3.0)
    assert thermal_tunnelling_factor(GC_SLOCOMBE_2022, BODY_T) < 2.0


def test_barrier_action_matches_the_parabolic_limit_near_the_top():
    """Near the top the exact WKB integral must equal ``pi*eps/omega_b``.

    This simultaneously validates the numerical integral below the barrier
    and the analytic continuation above it, and confirms the two join
    antisymmetrically rather than the action being clamped to zero above
    the top (which would pin P(E) at 1/2 and push kappa below 1).
    """
    pot = GC_SLOCOMBE_2022
    v_b = float(pot(pot.x_barrier))
    omega_b = pot.harmonic_frequency(pot.x_barrier)
    for eps in (1e-6, 1e-5, 1e-4):
        predicted = np.pi * eps / omega_b
        assert barrier_action(pot, v_b - eps) == pytest.approx(predicted, rel=0.02)
        assert barrier_action(pot, v_b + eps) == pytest.approx(-predicted, rel=1e-9)
    assert barrier_action(pot, v_b + 0.01) < 0.0  # transmission above 1/2
    assert barrier_action(pot, v_b - 0.01) > 0.0


def test_heavier_isotope_tunnels_less():
    kappa_h = thermal_tunnelling_factor(GC_SLOCOMBE_2022, BODY_T, mass=PROTON_MASS_AU)
    kappa_d = thermal_tunnelling_factor(GC_SLOCOMBE_2022, BODY_T, mass=DEUTERON_MASS_AU)
    assert 1.0 <= kappa_d < kappa_h
    assert crossover_temperature(
        GC_SLOCOMBE_2022, mass=DEUTERON_MASS_AU
    ) < crossover_temperature(GC_SLOCOMBE_2022)


def test_kinetic_isotope_effect_is_modest_at_body_temperature():
    kie = kinetic_isotope_effect(GC_SLOCOMBE_2022, BODY_T)
    assert 1.2 < kie["kie_total"] < 3.0
    # the classical prefactor alone contributes sqrt(m_D/m_H)
    assert kie["kie_classical_prefactor"] == pytest.approx(np.sqrt(2.0), rel=0.02)


# ----------------------------------------------------------------------
# tautomer populations
# ----------------------------------------------------------------------
def test_quantum_and_classical_occupancies_agree_to_within_a_small_factor():
    """Detailed balance forbids tunnelling from inflating the population."""
    q = tautomer_occupancy_quantum(GC_SLOCOMBE_2022, BODY_T)
    c = tautomer_occupancy_classical(GC_SLOCOMBE_2022, BODY_T)
    assert 0.3 < q / c < 3.0


def test_occupancy_matches_the_kinetic_equilibrium_constant():
    """Two independent routes to K_eq must land in the same place."""
    analysis = analyse_tautomer(GC_SLOCOMBE_2022, BODY_T)
    assert analysis.occupancy_quantum == pytest.approx(
        analysis.kinetic_equilibrium_constant, rel=0.6
    )


def test_occupancy_is_near_1e_minus_8_not_1e_minus_4():
    """The published open-system value 1.73e-4 is four orders away."""
    p = tautomer_occupancy_quantum(GC_SLOCOMBE_2022, BODY_T)
    assert 1e-9 < p < 1e-6


def test_occupancy_follows_the_boltzmann_asymmetry():
    warm = tautomer_occupancy_quantum(GC_SLOCOMBE_2022, 350.0)
    cold = tautomer_occupancy_quantum(GC_SLOCOMBE_2022, 280.0)
    assert warm > cold


def test_tautomer_cannot_survive_to_base_insertion():
    analysis = analyse_tautomer(GC_SLOCOMBE_2022, BODY_T)
    assert analysis.rates.tautomer_lifetime_s < 1e-6
    assert analysis.survival_probability(1e-3) == 0.0


def test_implied_fixation_probability_is_plausible():
    """~61 tautomeric templates per replication needs f_fix of order 0.1."""
    analysis = analyse_tautomer(GC_SLOCOMBE_2022, BODY_T)
    assert 10.0 < analysis.tautomeric_templates_per_replication() < 300.0
    assert 0.01 < analysis.implied_fixation_probability() < 1.0
    # the open-system occupancy would demand an implausibly small value
    assert analysis.report()["implied_f_fix_if_slocombe"] < 1e-3


def test_at_tautomer_is_far_shorter_lived_than_gc():
    """A small reverse barrier is why A*-T* cannot matter."""
    gc = analyse_tautomer(GC_SLOCOMBE_2022, BODY_T)
    at = analyse_tautomer(AT_ILLUSTRATIVE, BODY_T)
    assert AT_ILLUSTRATIVE.reverse_barrier * HARTREE_EV < 0.1
    assert at.rates.tautomer_lifetime_s < gc.rates.tautomer_lifetime_s


# ----------------------------------------------------------------------
# guard rails
# ----------------------------------------------------------------------
def test_non_double_well_parameters_are_rejected():
    """A monotone right flank used to report a zero reverse barrier."""
    bad = DoubleMorsePotential(0.1617, 0.0001, 0.305, 0.755, -2.7, 2.1, "degenerate")
    with pytest.raises(ValueError, match="double well|barrier"):
        _ = bad.forward_barrier


def test_rate_constants_are_positive_and_ordered():
    rc = rate_constants(GC_SLOCOMBE_2022, BODY_T)
    assert rc.k_forward > 0 and rc.k_reverse > 0
    # the reverse barrier is far lower, so the reverse rate is far higher
    assert rc.k_reverse > 1e5 * rc.k_forward
    assert rc.equilibrium_constant == pytest.approx(
        rc.k_forward_classical / rc.k_reverse_classical, rel=1e-9
    )
