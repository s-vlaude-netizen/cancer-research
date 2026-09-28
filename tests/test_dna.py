"""DNA layer: channel bookkeeping and signature recovery."""

from __future__ import annotations

import numpy as np
import pytest

from oncosim.dna import (
    N_CHANNELS,
    bootstrap_exposures,
    channel_context_counts,
    channel_index,
    channel_labels,
    context_of_channel,
    cosine_similarity,
    count_trinucleotides,
    encode_sequence,
    extract_signatures,
    fit_exposures,
    pyrimidine_context_counts,
    refit_with_selection,
    reverse_complement,
    simulate_spectrum,
    stylized_signature_set,
    substitution_of_channel,
)


# ----------------------------------------------------------------------
# channels
# ----------------------------------------------------------------------
def test_channel_labels_follow_the_cosmic_order():
    labels = channel_labels()
    assert len(labels) == N_CHANNELS
    assert len(set(labels)) == N_CHANNELS
    assert labels[0] == "A[C>A]A"
    assert labels[15] == "T[C>A]T"
    assert labels[16] == "A[C>G]A"
    assert labels[95] == "T[T>G]T"


def test_purine_reference_folds_to_the_pyrimidine_strand():
    """C[G>A]T and A[C>T]G are the same event on opposite strands."""
    i = channel_index("C", "G", "A", "T")
    assert channel_labels()[i] == "A[C>T]G"
    assert i == channel_index("A", "C", "T", "G")


def test_channel_accessors_are_mutually_consistent():
    for i, label in enumerate(channel_labels()):
        five, three = label[0], label[-1]
        sub = label[2:5]
        assert substitution_of_channel(i) == sub
        assert context_of_channel(i) == f"{five}{sub[0]}{three}"


def test_invalid_substitutions_are_rejected():
    with pytest.raises(ValueError):
        channel_index("A", "C", "C", "G")  # ref == alt
    with pytest.raises(ValueError):
        channel_index("A", "C", "T", "N")


# ----------------------------------------------------------------------
# sequence composition
# ----------------------------------------------------------------------
def test_trinucleotide_counts_are_complete():
    seq = "ACGTACGTTTGCAACGCGATCGATCGGGCCCTTTAAA"
    assert count_trinucleotides(encode_sequence(seq)).sum() == len(seq) - 2


def test_unknown_bases_are_skipped():
    assert count_trinucleotides(encode_sequence("ACNGT")).sum() == 0
    assert count_trinucleotides(encode_sequence("ACGNACG")).sum() == 2


def test_pyrimidine_contexts_are_reverse_complement_invariant():
    """A mutation is strand-agnostic, so its context counts must be too."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=5000))
    forward = pyrimidine_context_counts(encode_sequence(seq))
    reverse = pyrimidine_context_counts(encode_sequence(reverse_complement(seq)))
    assert np.array_equal(forward, reverse)
    assert forward.sum() == len(seq) - 2


def test_channel_context_counts_triple_the_context_counts():
    codes = encode_sequence("ACGTACGTTTGCAACGCGATCGATCGGG")
    assert channel_context_counts(codes).sum() == 3 * pyrimidine_context_counts(
        codes
    ).sum()


# ----------------------------------------------------------------------
# signatures
# ----------------------------------------------------------------------
def test_signature_profiles_are_normalised():
    s = stylized_signature_set()
    assert np.allclose(s.matrix.sum(axis=0), 1.0)
    assert np.all(s.matrix >= 0)


def test_stylized_signatures_carry_their_intended_peaks():
    s = stylized_signature_set()
    cpg_ct = [
        i
        for i in range(N_CHANNELS)
        if substitution_of_channel(i) == "C>T" and context_of_channel(i)[2] == "G"
    ]
    assert s["SBS1"][cpg_ct].sum() > 0.7  # SBS1 lives at CpG
    tcw_cg = [
        i
        for i in range(N_CHANNELS)
        if substitution_of_channel(i) == "C>G"
        and context_of_channel(i)[0] == "T"
        and context_of_channel(i)[2] in ("A", "T")
    ]
    assert s["SBS13"][tcw_cg].sum() > 0.7  # APOBEC transversion arm at TCW
    assert np.allclose(s["SBS3"], 1.0 / N_CHANNELS)  # HRD is flat


def test_cosine_similarity_bounds():
    s = stylized_signature_set()
    assert cosine_similarity(s["SBS1"], s["SBS1"]) == pytest.approx(1.0)
    sim = s.similarity_matrix()
    assert np.allclose(np.diag(sim), 1.0)
    assert np.all(sim >= -1e-12) and np.all(sim <= 1.0 + 1e-12)


def test_refitting_recovers_a_known_mixture():
    rng = np.random.default_rng(5)
    s = stylized_signature_set()
    truth = {"SBS1": 300.0, "SBS4": 1200.0, "SBS5": 500.0}
    spectrum = simulate_spectrum(s, truth, rng=rng)
    fit = fit_exposures(spectrum, s)
    assert fit.cosine > 0.98
    recovered = fit.as_dict()
    for name, value in truth.items():
        assert recovered[name] == pytest.approx(value, rel=0.35)
    # signatures that were not used must stay small
    assert recovered["SBS7a"] < 0.05 * spectrum.sum()


def test_greedy_selection_picks_a_small_subset():
    rng = np.random.default_rng(6)
    s = stylized_signature_set()
    spectrum = simulate_spectrum(s, {"SBS4": 2000.0}, rng=rng)
    fit = refit_with_selection(spectrum, s)
    assert fit.cosine > 0.97
    assert len(fit.names) <= 3
    assert "SBS4" in fit.names


def test_bootstrap_brackets_the_truth_for_a_well_separated_signature():
    rng = np.random.default_rng(7)
    s = stylized_signature_set()
    spectrum = simulate_spectrum(s, {"SBS4": 1500.0, "SBS7a": 800.0}, rng=rng)
    ci = bootstrap_exposures(spectrum, s, n_boot=80, rng=rng)
    lo, hi = ci["SBS4"][1], ci["SBS4"][2]
    assert lo <= 1500.0 <= hi
    assert hi > lo


def test_opportunity_weighting_changes_the_expected_spectrum():
    """CpG depletion must reweight a CpG-driven signature."""
    rng = np.random.default_rng(8)
    s = stylized_signature_set(["SBS1"])
    codes = encode_sequence("".join(["ATATATATAT"] * 200))  # no CpG at all
    opportunities = channel_context_counts(codes)
    plain = simulate_spectrum(s, {"SBS1": 5000.0}, rng=rng)
    weighted = simulate_spectrum(
        s, {"SBS1": 5000.0}, context_counts=opportunities, rng=rng
    )
    assert cosine_similarity(plain, weighted) < 0.5


def test_nmf_recovers_planted_signatures():
    rng = np.random.default_rng(1)
    s = stylized_signature_set(["SBS1", "SBS4", "SBS7a", "SBS13"])
    catalogue = rng.poisson(s.matrix @ rng.gamma(2.0, 400.0, size=(4, 60))).astype(float)
    result = extract_signatures(catalogue, 4, n_restarts=6, rng=rng)
    assert np.allclose(result.signatures.sum(axis=0), 1.0)
    for j in range(4):
        best = max(
            cosine_similarity(result.signatures[:, k], s.matrix[:, j]) for k in range(4)
        )
        assert best > 0.9


def test_nmf_convergence_check_does_not_stop_immediately():
    """Regression: prev = inf satisfied `inf <= tol*inf` on the first check."""
    rng = np.random.default_rng(2)
    s = stylized_signature_set(["SBS1", "SBS4"])
    catalogue = rng.poisson(s.matrix @ rng.gamma(2.0, 300.0, size=(2, 20))).astype(float)
    result = extract_signatures(catalogue, 2, n_restarts=2, max_iter=500, rng=rng)
    assert result.n_iter > 25


def test_spectrum_length_is_validated():
    s = stylized_signature_set()
    with pytest.raises(ValueError):
        fit_exposures(np.ones(50), s)
    with pytest.raises(ValueError):
        fit_exposures(-np.ones(N_CHANNELS), s)
