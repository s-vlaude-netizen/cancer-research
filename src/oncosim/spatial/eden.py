"""Spatial tumour growth on a lattice: surface growth, turnover, dispersal.

Why space changes the answer
----------------------------
A well-mixed branching process says a driver with a 1% advantage sweeps a
10^9-cell tumour in a few hundred generations.  A solid tumour is not well
mixed: only cells at the surface have room to divide, so a mutant arising
in the interior is frozen where it is and a mutant at the surface expands
as a *sector*, not as a sweep.  Purely spatial growth therefore predicts
tumours far more heterogeneous than the ones actually sequenced.

Waclaw, Bozic, Pittman, Hruban, Vogelstein & Nowak (Nature 2015) showed
what closes the gap: short-range *dispersal*, in which a cell occasionally
migrates a few cell diameters and founds a new microlesion, combined with
cell *turnover*.  Together they restore enough mixing for a small
advantage to take over the mass within a clinically relevant time, and
they reproduce the observed low intratumour heterogeneity.

What this implementation reproduces
-----------------------------------
Measured at 5000 cells with neutral clone tagging (6 replicates per cell,
``dispersal_radius = 6``, seed 41), Simpson diversity and radius of
gyration:

======================  =================  =================
turnover ``d/b``        no dispersal       ``p_disp = 0.05``
======================  =================  =================
0.0                     0.172, Rg 8.41     0.209, Rg 9.50
0.5                     0.292, Rg 8.89     0.236, Rg 10.12
0.8                     0.550, Rg 9.92     0.437, Rg 11.54
======================  =================  =================

Read the turnover dependence, not any single row.  Turnover on its own
raises heterogeneity sharply (0.17 to 0.55), and dispersal pulls it back
down by an amount that grows with turnover -- which is the Waclaw et al.
claim, that the two ingredients act *together*.  The zero-turnover row
goes the other way here and is within Monte-Carlo noise: with no turnover
there is nothing for dispersal to mix.

The heterogeneity effect is real but noisy (roughly 2 standard errors at
this many replicates), so the test suite asserts only the sharp,
low-variance consequence -- dispersal makes the mass measurably less
compact, ~9 standard errors -- and
``experiments/spatial_heterogeneity.py`` runs the diversity comparison
with enough replicates to resolve it.

Pure surface growth is a good sphere: at 5000 cells the predicted radius
of gyration of a uniform ball, ``sqrt(3/5) R = 8.22``, is matched to
within 2%.

Model
-----
Cells occupy sites of a 3-D cubic lattice.

* **Division** at rate ``birth_rate`` (scaled by ``(1+s)`` per driver), but
  only for cells with at least one empty neighbouring site.  The daughter
  takes a uniformly chosen empty neighbour.
* **Death** at rate ``death_rate``, for any cell, anywhere.
* **Dispersal**: with probability ``dispersal_probability`` a daughter is
  instead placed at a uniformly chosen empty site within
  ``dispersal_radius`` lattice units, founding a microlesion.
* **Drivers** arise at each division with probability ``driver_rate``.
  Setting ``selection_coefficient = 0`` turns this into neutral clone
  tagging, which is how genetic diversity is measured above.

Implementation
--------------
The cost driver is picking a dividing cell in proportion to its rate while
the surface changes constantly.  A Fenwick tree over per-cell weights
(``birth rate`` if the cell has an empty neighbour, ``0`` otherwise) makes
selection ``O(log N)``, and each event touches only the 6 neighbours of one
site.  The lattice is addressed by *flat* indices, so a neighbourhood
query is one length-6 fancy-index with no coordinate arithmetic.
Practical to a few times ``10^5`` cells in pure Python.

References
----------
Waclaw et al. (2015) Nature 525:261-264.
Eden, M. (1961) Proc. 4th Berkeley Symp. 4:223-239.
Sottoriva et al. (2015) Nat. Genet. 47:209-216 (Big Bang colorectal growth).
Noble, Burri et al. (2022) Nat. Ecol. Evol. 6:207-217 (spatial structure
    and the mode of tumour evolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..algorithms.fenwick import FenwickTree

__all__ = ["SpatialModel", "SpatialTumour", "simulate_spatial"]

#: Largest cubic lattice the auto-sizing logic will grow to, per edge.
#: 512**3 int32 cells is 0.5 GB, which is already at the edge of comfort.
_MAX_LATTICE_SIZE = 512


class _BoundaryReached(RuntimeError):
    """Raised internally when growth touches the lattice edge."""


@dataclass(frozen=True)
class SpatialModel:
    """Parameters of the spatial growth model."""

    birth_rate: float = 1.0
    death_rate: float = 0.0
    driver_rate: float = 1e-5
    selection_coefficient: float = 0.1
    dispersal_probability: float = 0.0
    """Chance that a daughter migrates instead of taking an adjacent site."""
    dispersal_radius: int = 5
    """Maximum Chebyshev distance of a dispersal hop, in lattice units."""
    max_drivers: int = 20

    def __post_init__(self) -> None:
        if self.birth_rate <= 0:
            raise ValueError("birth_rate must be positive")
        if self.death_rate < 0:
            raise ValueError("death_rate must be non-negative")
        if not 0.0 <= self.dispersal_probability <= 1.0:
            raise ValueError("dispersal_probability must be a probability")
        if self.dispersal_radius < 1:
            raise ValueError("dispersal_radius must be at least 1")

    def birth_rate_with(self, n_drivers: int) -> float:
        k = min(int(n_drivers), self.max_drivers)
        return self.birth_rate * (1.0 + self.selection_coefficient) ** k


@dataclass
class SpatialTumour:
    """Final state of a spatial simulation."""

    positions: np.ndarray
    """``(n_cells, 3)`` integer lattice coordinates."""
    clone_of: np.ndarray
    """Clone index of each cell."""
    clone_drivers: np.ndarray
    """Number of drivers carried by each clone."""
    clone_parent: np.ndarray
    """Parent clone of each clone; ``-1`` for the founder."""
    time: float
    n_events: int
    model: SpatialModel
    metadata: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.positions.shape[0])

    def clone_sizes(self) -> np.ndarray:
        return np.bincount(self.clone_of, minlength=self.clone_drivers.size)

    def radius_of_gyration(self) -> float:
        """RMS distance of cells from their centroid -- a shape summary.

        For a uniform ball of radius ``R`` this is ``sqrt(3/5) R``, which is
        what pure surface growth reproduces; dispersal raises it because
        the mass is no longer compact.
        """
        if self.n_cells == 0:
            return 0.0
        centre = self.positions.mean(axis=0)
        return float(np.sqrt(((self.positions - centre) ** 2).sum(axis=1).mean()))

    def mean_drivers(self) -> float:
        if self.n_cells == 0:
            return 0.0
        return float(self.clone_drivers[self.clone_of].mean())

    def heterogeneity(self) -> float:
        """Simpson diversity ``1 - sum f_i^2`` over clones.

        0 means one clone owns everything; values near 1 mean no clone
        dominates.  Bulk-sequenced human tumours mostly land below 0.5,
        which purely surface-limited growth without dispersal struggles to
        reproduce.
        """
        sizes = self.clone_sizes()
        if sizes.sum() == 0:
            return 0.0
        f = sizes / sizes.sum()
        return float(1.0 - np.sum(f**2))

    def biopsy(
        self,
        centre: np.ndarray | None = None,
        radius: float = 5.0,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Clone composition of a spherical virtual biopsy.

        Returns clone sizes within the sampled region.  Multi-region
        sequencing studies take a handful of such samples; comparing them
        is how spatial models are confronted with data.
        """
        if self.n_cells == 0:
            return np.zeros(self.clone_drivers.size, dtype=np.int64)
        if centre is None:
            if rng is None:
                rng = np.random.default_rng()
            centre = self.positions[rng.integers(self.n_cells)]
        centre = np.asarray(centre, dtype=np.float64)
        inside = ((self.positions - centre) ** 2).sum(axis=1) <= radius**2
        return np.bincount(
            self.clone_of[inside], minlength=self.clone_drivers.size
        ).astype(np.int64)


class _Lattice:
    """Dense lattice of cell indices with ``-1`` for empty sites.

    Sites are addressed by a *flat* index into the raveled grid, which is
    what makes the inner loop fast: the six neighbours of a site are just
    ``flat + offsets`` for a fixed offset vector, so a neighbourhood query
    is one fancy-index of length 6 with no coordinate arithmetic and no
    per-axis bounds checking.  Bounds safety comes from the simulator
    aborting as soon as the tumour approaches the boundary, so the
    neighbourhood of every occupied site is guaranteed to be interior.
    """

    __slots__ = ("grid", "flat", "size", "origin", "offsets")

    def __init__(self, size: int) -> None:
        self.size = size
        self.grid = np.full((size, size, size), -1, dtype=np.int32)
        self.flat = self.grid.reshape(-1)
        self.origin = size // 2
        s = size
        self.offsets = np.array([s * s, -s * s, s, -s, 1, -1], dtype=np.int64)

    def to_flat(self, p: np.ndarray) -> int:
        return int((p[0] * self.size + p[1]) * self.size + p[2])

    def to_coords(self, flat: int) -> np.ndarray:
        s = self.size
        return np.array([flat // (s * s), (flat // s) % s, flat % s], dtype=np.int64)

    def in_bounds(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= 0) and np.all(p < self.size))


def simulate_spatial(
    model: SpatialModel,
    n_target: int,
    *,
    rng: np.random.Generator | None = None,
    lattice_size: int | None = None,
    max_events: int | None = None,
    condition_on_survival: bool = True,
    max_attempts: int = 200,
) -> SpatialTumour:
    """Grow a spatial tumour until it reaches ``n_target`` cells.

    Parameters
    ----------
    lattice_size:
        Edge length of the cubic lattice.  Left unset, it is chosen for a
        sphere of ``n_target`` cells plus a dispersal allowance, and grown
        automatically if the tumour still reaches the edge.  Passed
        explicitly, a tumour that reaches the boundary raises rather than
        silently wrapping around.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_target < 1:
        raise ValueError("n_target must be >= 1")

    auto_size = lattice_size is None
    if auto_size:
        radius = (3.0 * n_target / (4.0 * np.pi)) ** (1.0 / 3.0)
        # Dispersal makes the mass non-compact: microlesions seed outwards
        # and can themselves seed further, so the required box grows with
        # both the hop length and how often hops happen.
        spread = model.dispersal_radius * (3.0 + 60.0 * model.dispersal_probability)
        lattice_size = int(2 * (radius + 6.0 + spread)) + 1
    if max_events is None:
        beta = (model.birth_rate - model.death_rate) / model.birth_rate
        max_events = int(200 * n_target / max(beta, 1e-3)) + 10_000

    for _attempt in range(max_attempts):
        try:
            result = _grow_once(model, n_target, rng, lattice_size, max_events)
        except _BoundaryReached:
            if not auto_size:
                raise RuntimeError(
                    f"tumour reached the boundary of the {lattice_size}-cell "
                    "lattice; pass a larger lattice_size"
                ) from None
            # The heuristic under-estimates the spread for strongly
            # dispersing, high-turnover tumours.  Grow the box rather than
            # failing, but never silently: the memory cost is cubic.
            lattice_size *= 2
            if lattice_size > _MAX_LATTICE_SIZE:
                raise RuntimeError(
                    f"tumour needs a lattice larger than {_MAX_LATTICE_SIZE} "
                    f"cells per edge ({_MAX_LATTICE_SIZE**3 * 4 / 1e9:.1f} GB); "
                    "reduce n_target, dispersal_radius or dispersal_probability"
                ) from None
            continue
        if result is not None and (
            not condition_on_survival or result.n_cells >= n_target
        ):
            return result
    raise RuntimeError(f"no lineage reached {n_target} cells in {max_attempts} attempts")


def _grow_once(
    model: SpatialModel,
    n_target: int,
    rng: np.random.Generator,
    lattice_size: int,
    max_events: int,
) -> SpatialTumour | None:
    lat = _Lattice(lattice_size)
    cap = max(1024, n_target * 2)
    site = np.zeros(cap, dtype=np.int64)  # flat lattice index of each cell
    clone_of = np.zeros(cap, dtype=np.int64)

    clone_drivers = [0]
    clone_parent = [-1]

    start = np.array([lat.origin, lat.origin, lat.origin], dtype=np.int64)
    site[0] = lat.to_flat(start)
    lat.flat[site[0]] = 0
    n_cells = 1

    tree = FenwickTree(np.array([model.birth_rate_with(0)]))
    t = 0.0
    events = 0

    grid_flat = lat.flat
    offsets = lat.offsets
    # Per-driver-count birth rates, precomputed: the inner loop looks these
    # up on every weight refresh and recomputing a power each time is the
    # single biggest avoidable cost.
    rate_cache = [model.birth_rate_with(k) for k in range(model.max_drivers + 2)]

    def free_neighbour_sites(flat: int) -> np.ndarray:
        cand = flat + offsets
        return cand[grid_flat[cand] == -1]

    def refresh_weight(cell: int) -> None:
        """Set a cell's Fenwick weight to its birth rate iff it can divide."""
        cand = site[cell] + offsets
        if np.any(grid_flat[cand] == -1):
            tree.set(cell, rate_cache[clone_drivers[clone_of[cell]]])
        else:
            tree.set(cell, 0.0)

    def refresh_around(flat: int) -> None:
        occupants = grid_flat[flat + offsets]
        for c in occupants[occupants >= 0]:
            refresh_weight(int(c))

    while n_cells < n_target and n_cells > 0 and events < max_events:
        birth_total = tree.total()
        death_total = model.death_rate * n_cells
        rate_total = birth_total + death_total
        if rate_total <= 0.0:
            # fully jammed: every cell is interior and nothing can die
            break
        t += float(rng.exponential(1.0 / rate_total))

        if rng.random() < birth_total / rate_total:
            parent = tree.sample(rng.random())
            flat_parent = int(site[parent])
            target = -1
            if (
                model.dispersal_probability > 0.0
                and rng.random() < model.dispersal_probability
            ):
                target = _random_free_site_near(
                    lat, flat_parent, model.dispersal_radius, rng
                )
            if target < 0:
                options = free_neighbour_sites(flat_parent)
                if options.shape[0] == 0:
                    refresh_weight(parent)
                    events += 1
                    continue
                target = int(options[rng.integers(options.shape[0])])

            coords = lat.to_coords(target)
            # Two sites of slack keeps every occupied cell's six-neighbour
            # flat-index lookup inside the array and free of wrap-around.
            if np.any(coords <= 1) or np.any(coords >= lat.size - 2):
                raise _BoundaryReached

            child_clone = int(clone_of[parent])
            if (
                model.driver_rate > 0.0
                and clone_drivers[child_clone] < model.max_drivers
                and rng.random() < model.driver_rate
            ):
                clone_drivers.append(clone_drivers[child_clone] + 1)
                clone_parent.append(child_clone)
                child_clone = len(clone_drivers) - 1

            if n_cells == site.shape[0]:
                site = np.concatenate([site, np.zeros_like(site)])
                clone_of = np.concatenate([clone_of, np.zeros_like(clone_of)])
            idx = n_cells
            site[idx] = target
            clone_of[idx] = child_clone
            grid_flat[target] = idx
            n_cells += 1
            tree.append(0.0)
            refresh_weight(idx)
            refresh_weight(parent)
            refresh_around(target)
        else:
            victim = int(rng.integers(n_cells))
            flat_victim = int(site[victim])
            last = n_cells - 1
            grid_flat[flat_victim] = -1
            if victim != last:
                # swap-remove: move the last cell into the vacated slot
                site[victim] = site[last]
                clone_of[victim] = clone_of[last]
                grid_flat[site[victim]] = victim
                tree.set(victim, tree.get(last))
            tree.set(last, 0.0)
            n_cells -= 1
            if n_cells == 0:
                break
            if victim < n_cells:
                refresh_weight(victim)
            refresh_around(flat_victim)
        events += 1

    if n_cells == 0:
        return None
    s = lat.size
    flats = site[:n_cells]
    positions = np.stack([flats // (s * s), (flats // s) % s, flats % s], axis=1)
    return SpatialTumour(
        positions=positions.astype(np.int64),
        clone_of=clone_of[:n_cells].copy(),
        clone_drivers=np.array(clone_drivers, dtype=np.int64),
        clone_parent=np.array(clone_parent, dtype=np.int64),
        time=t,
        n_events=events,
        model=model,
        metadata={"lattice_size": lat.size},
    )


def _random_free_site_near(
    lat: _Lattice, flat: int, radius: int, rng: np.random.Generator
) -> int:
    """Rejection-sample an empty site within Chebyshev distance ``radius``.

    Returns the flat index, or ``-1`` if no empty site was found in the
    allotted attempts (the caller then falls back to adjacent placement).
    """
    p = lat.to_coords(flat)
    for _ in range(30):
        offset = rng.integers(-radius, radius + 1, size=3)
        if not np.any(offset):
            continue
        cand = p + offset
        if not lat.in_bounds(cand):
            continue
        c = lat.to_flat(cand)
        if lat.flat[c] == -1:
            return c
    return -1
