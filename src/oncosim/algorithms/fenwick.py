"""Fenwick (binary indexed) tree for weighted categorical sampling.

The inner loop of every exact stochastic simulation of a growing cell
population is "pick a clone with probability proportional to its
propensity, then change that propensity".  A naive implementation
recomputes the cumulative sum on every step and costs ``O(K)`` per event
for ``K`` clones; over ``E`` events that is ``O(E*K)``, which is the
reason most naive tumour simulators die at ~10^4 clones.

A Fenwick tree stores partial sums of the weight vector so that

* ``update(i, delta)``  -- ``O(log K)``
* ``total()``           -- ``O(log K)``
* ``sample(u)``         -- ``O(log K)`` via a descent over the implicit tree

Total cost ``O(E log K)``.  The descent used here (`_find_prefix`) walks
the tree top-down using the binary structure directly rather than doing a
binary search with ``O(log^2 K)`` prefix-sum queries.

References
----------
Fenwick, P. (1994) "A new data structure for cumulative frequency tables",
Software: Practice and Experience 24(3):327-336.

Gillespie, D.T. (1976) J. Comput. Phys. 22:403-434 -- the direct method
whose search step this accelerates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FenwickTree"]


class FenwickTree:
    """Mutable weight vector supporting O(log n) proportional sampling.

    Parameters
    ----------
    weights:
        Initial non-negative weights.  May be a length (int) for an
        all-zero tree of that capacity.

    Notes
    -----
    Weights must stay non-negative.  Floating point cancellation in
    ``update`` is bounded because each update touches ``O(log n)`` nodes;
    :meth:`rebuild` recomputes the tree from the stored leaf weights and
    is cheap enough to call periodically in very long runs.
    """

    __slots__ = ("_n", "_cap", "_leaf", "_tree", "_logn")

    def __init__(self, weights):
        if isinstance(weights, (int, np.integer)):
            leaf = np.zeros(int(weights), dtype=np.float64)
        else:
            leaf = np.asarray(weights, dtype=np.float64).copy()
        if leaf.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if np.any(leaf < 0):
            raise ValueError("weights must be non-negative")
        self._n = leaf.size
        self._cap = max(1, self._n)
        self._leaf = leaf
        self._tree = np.zeros(self._cap + 1, dtype=np.float64)
        self._logn = max(0, int(np.floor(np.log2(self._cap))))
        self.rebuild()

    # ------------------------------------------------------------------
    # construction / maintenance
    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        """Recompute internal partial sums from the leaf weights."""
        tree = self._tree
        tree[:] = 0.0
        # ``_leaf`` is a capacity-sized buffer; only the first ``_n``
        # entries are live weights.
        tree[1 : self._n + 1] = self._leaf[: self._n]
        # in-place linear-time build: push each node into its parent
        for i in range(1, self._cap + 1):
            j = i + (i & -i)
            if j <= self._cap:
                tree[j] += tree[i]

    def __len__(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        """Read-only view of the current leaf weights."""
        view = self._leaf[: self._n].view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def total(self) -> float:
        """Sum of all weights."""
        s = 0.0
        i = self._cap
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def prefix_sum(self, i: int) -> float:
        """Sum of weights ``[0, i)`` in O(log n)."""
        s = 0.0
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def get(self, i: int) -> float:
        return float(self._leaf[i])

    # ------------------------------------------------------------------
    # mutation
    # ------------------------------------------------------------------
    def update(self, i: int, delta: float) -> None:
        """Add ``delta`` to weight ``i`` in O(log n)."""
        new = self._leaf[i] + delta
        if new < 0.0:
            if new < -1e-9 * max(1.0, abs(self._leaf[i])):
                raise ValueError(f"weight {i} would become negative ({new})")
            new = 0.0
            delta = new - self._leaf[i]
        self._leaf[i] = new
        j = i + 1
        tree = self._tree
        cap = self._cap
        while j <= cap:
            tree[j] += delta
            j += j & -j

    def set(self, i: int, value: float) -> None:
        """Set weight ``i`` to ``value`` in O(log n)."""
        self.update(i, float(value) - self._leaf[i])

    def append(self, value: float) -> int:
        """Append a new weight, growing capacity geometrically.

        Returns the index of the new entry.  Amortised ``O(log n)``.
        """
        if self._n == self._leaf.size:
            new_cap = max(2, self._leaf.size * 2)
            grown = np.zeros(new_cap, dtype=np.float64)
            grown[: self._n] = self._leaf
            self._leaf = grown
            self._cap = new_cap
            self._tree = np.zeros(new_cap + 1, dtype=np.float64)
            self._logn = max(0, int(np.floor(np.log2(new_cap))))
            idx = self._n
            self._n += 1
            self._leaf[idx] = float(value)
            self.rebuild()
            return idx
        idx = self._n
        self._n += 1
        self._leaf[idx] = 0.0
        self.update(idx, float(value))
        return idx

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def sample(self, u: float) -> int:
        """Return the index whose weight interval contains ``u * total``.

        ``u`` must lie in ``[0, 1)``.  Cost ``O(log n)`` with a single
        top-down descent (no repeated prefix-sum queries).
        """
        return self._find_prefix(u * self.total())

    def sample_rng(self, rng: np.random.Generator) -> int:
        return self.sample(rng.random())

    def sample_many(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Vectorised i.i.d. draws (weights held fixed)."""
        us = rng.random(size)
        out = np.empty(size, dtype=np.int64)
        tot = self.total()
        for k in range(size):
            out[k] = self._find_prefix(us[k] * tot)
        return out

    def _find_prefix(self, target: float) -> int:
        """Largest ``i`` with ``prefix_sum(i) <= target``; returns index ``i``."""
        tree = self._tree
        pos = 0
        step = 1 << self._logn
        cap = self._cap
        while step > 0:
            nxt = pos + step
            if nxt <= cap and tree[nxt] <= target:
                target -= tree[nxt]
                pos = nxt
            step >>= 1
        # ``pos`` is the number of leaves fully consumed -> index ``pos``
        if pos >= self._n:  # only reachable through floating-point slack
            pos = self._n - 1
            while pos > 0 and self._leaf[pos] == 0.0:
                pos -= 1
        return pos
