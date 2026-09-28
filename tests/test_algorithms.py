"""Fenwick tree: exactness against a brute-force reference."""

from __future__ import annotations

import numpy as np
import pytest

from oncosim.algorithms import FenwickTree


def test_total_and_prefix_match_brute_force():
    rng = np.random.default_rng(0)
    w = rng.random(257) * 5.0
    tree = FenwickTree(w)
    assert tree.total() == pytest.approx(w.sum())
    for i in (0, 1, 13, 128, 200, 257):
        assert tree.prefix_sum(i) == pytest.approx(w[:i].sum())


def test_update_keeps_weights_consistent():
    rng = np.random.default_rng(1)
    w = rng.random(64) + 0.1
    tree = FenwickTree(w)
    for _ in range(200):
        i = int(rng.integers(w.size))
        delta = float(rng.normal(0.0, 0.05))
        if w[i] + delta < 0:
            continue
        tree.update(i, delta)
        w[i] += delta
    assert np.allclose(tree.weights, w)
    assert tree.total() == pytest.approx(w.sum())


def test_append_past_initial_capacity():
    """Regression: rebuild() must use the live prefix, not the whole buffer."""
    tree = FenwickTree(np.array([1.0]))
    reference = [1.0]
    rng = np.random.default_rng(2)
    for _ in range(300):
        v = float(rng.random() * 3.0)
        tree.append(v)
        reference.append(v)
    reference = np.array(reference)
    assert len(tree) == reference.size
    assert np.allclose(tree.weights, reference)
    assert tree.total() == pytest.approx(reference.sum())


def test_sampling_reproduces_the_weight_distribution():
    rng = np.random.default_rng(3)
    w = np.array([5.0, 1.0, 0.0, 3.0, 11.0, 0.5])
    tree = FenwickTree(w)
    draws = tree.sample_many(rng, 200_000)
    counts = np.bincount(draws, minlength=w.size) / draws.size
    assert np.allclose(counts, w / w.sum(), atol=5e-3)
    # a zero-weight entry must never be selected
    assert counts[2] == 0.0


def test_rejects_negative_weights():
    with pytest.raises(ValueError):
        FenwickTree(np.array([1.0, -1.0]))
    tree = FenwickTree(np.array([1.0]))
    with pytest.raises(ValueError):
        tree.update(0, -5.0)
