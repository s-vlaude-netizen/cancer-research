"""Mutational signatures: reference profiles, refitting and de novo extraction.

A tumour's 96-channel mutation spectrum is, to a good approximation, a
non-negative mixture of a handful of *signatures*, each the fingerprint of
one mutational process: 5-methylcytosine deamination, tobacco adducts, UV
photodimers, APOBEC cytidine deaminases, failed mismatch repair, and so
on.  Recovering the mixture from the spectrum is the workhorse
computation of cancer genome analysis, and it is what tells you *which
carcinogen* wrote a given tumour -- which is the entry point for
prevention.

Two problems live here and they are not the same:

**Refitting** -- given known reference signatures, find the exposures.
This is non-negative least squares, convex, and well posed as long as the
reference set is not over-complete.  :func:`fit_exposures` solves it, and
:func:`refit_with_selection` adds a greedy forward selection step because
throwing all ~80 COSMIC signatures at one tumour reliably invents
processes that are not there.

**De novo extraction** -- given many tumours, discover the signatures
themselves.  This is non-negative matrix factorisation, non-convex, and
sensitive to initialisation.  :func:`extract_signatures` implements
multiplicative updates under a Poisson (Kullback-Leibler) likelihood,
which is the right noise model for counts, with multiple restarts.

About the bundled reference profiles
------------------------------------
:data:`STYLIZED_SIGNATURES` contains **stylized** profiles built from the
published *descriptions* of each process (which channels carry the peaks),
not the COSMIC numerical catalogue, which carries its own licence.  They
are good enough to exercise and validate the algorithms and to reason
qualitatively; they are **not** a substitute for the real catalogue.  For
quantitative work download the COSMIC signature matrix and load it with
:func:`load_signature_matrix`, which validates channel order on the way in.

References
----------
Alexandrov et al. (2013) Nature 500:415-421.
Alexandrov et al. (2020) Nature 578:94-101.
Lee & Seung (2001) NIPS 13:556-562 (multiplicative NMF updates).
Degasperi et al. (2022) Science 376:science.abl9283 (signature reference sets).
Maura et al. (2019) Nat. Commun. 10:2969 (pitfalls of over-fitting exposures).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .channels import (
    N_CHANNELS,
    channel_labels,
    context_of_channel,
    substitution_of_channel,
)

__all__ = [
    "SignatureSet",
    "STYLIZED_SIGNATURES",
    "SIGNATURE_AETIOLOGY",
    "stylized_signature_set",
    "cosine_similarity",
    "ExposureFit",
    "ExtractionResult",
    "fit_exposures",
    "refit_with_selection",
    "extract_signatures",
    "bootstrap_exposures",
    "load_signature_matrix",
    "simulate_spectrum",
]


# ----------------------------------------------------------------------
# stylized reference profiles
# ----------------------------------------------------------------------
def _profile(rule) -> np.ndarray:
    """Build a normalised 96-vector from a ``(sub, context) -> weight`` rule."""
    w = np.zeros(N_CHANNELS, dtype=np.float64)
    for i in range(N_CHANNELS):
        w[i] = float(rule(substitution_of_channel(i), context_of_channel(i)))
    total = w.sum()
    if total <= 0:
        raise ValueError("signature rule produced an all-zero profile")
    return w / total


def _sbs1(sub: str, ctx: str) -> float:
    """5-methylcytosine deamination: C>T at CpG, the 'clock-like' signature."""
    if sub == "C>T" and ctx[2] == "G":
        return 20.0
    if sub == "C>T":
        return 0.6
    if sub == "T>C":
        return 0.25
    return 0.08


def _sbs2(sub: str, ctx: str) -> float:
    """APOBEC3 deamination, transition arm: C>T at TCW (W = A or T)."""
    if sub == "C>T" and ctx[0] == "T" and ctx[2] in ("A", "T"):
        return 30.0
    if sub == "C>T" and ctx[0] == "T":
        return 2.0
    if sub == "C>G" and ctx[0] == "T" and ctx[2] in ("A", "T"):
        return 2.0
    return 0.05


def _sbs13(sub: str, ctx: str) -> float:
    """APOBEC3, transversion arm: C>G at TCW.  Co-occurs with SBS2."""
    if sub == "C>G" and ctx[0] == "T" and ctx[2] in ("A", "T"):
        return 30.0
    if sub == "C>G" and ctx[0] == "T":
        return 2.0
    if sub == "C>A" and ctx[0] == "T" and ctx[2] in ("A", "T"):
        return 2.5
    return 0.05


def _sbs4(sub: str, ctx: str) -> float:
    """Tobacco smoke: broad C>A from bulky benzo[a]pyrene adducts."""
    if sub == "C>A":
        return 4.0 + (1.5 if ctx[2] in ("A", "C") else 0.0)
    if sub == "C>T":
        return 0.5
    return 0.2


def _sbs5(sub: str, ctx: str) -> float:
    """Flat 'clock-like' signature of unknown aetiology, mild T>C at ApTpN."""
    base = 1.0
    if sub == "T>C" and ctx[0] == "A":
        base += 1.4
    if sub == "C>T":
        base += 0.5
    return base


def _sbs7a(sub: str, ctx: str) -> float:
    """UV: C>T at dipyrimidines, i.e. a pyrimidine immediately 5'."""
    if sub == "C>T" and ctx[0] in ("C", "T"):
        return 15.0 + (6.0 if ctx[0] == "C" else 0.0)
    if sub == "C>T":
        return 1.0
    return 0.05


def _sbs3(sub: str, ctx: str) -> float:
    """Homologous-recombination deficiency: essentially featureless."""
    del sub, ctx
    return 1.0


def _sbs6(sub: str, ctx: str) -> float:
    """Mismatch-repair deficiency: C>T at CpG plus a broad T>C component."""
    if sub == "C>T" and ctx[2] == "G":
        return 8.0
    if sub == "C>T":
        return 3.0
    if sub == "T>C":
        return 2.5
    return 0.3


def _sbs18(sub: str, ctx: str) -> float:
    """Reactive oxygen species via 8-oxo-guanine: C>A, strongest at NCA."""
    if sub == "C>A":
        return 5.0 + (5.0 if ctx[2] == "A" else 0.0)
    return 0.2


#: Stylized reference profiles keyed by COSMIC-like name.  **Not** the
#: COSMIC numerical catalogue -- see the module docstring.
STYLIZED_SIGNATURES: dict[str, np.ndarray] = {
    "SBS1": _profile(_sbs1),
    "SBS2": _profile(_sbs2),
    "SBS3": _profile(_sbs3),
    "SBS4": _profile(_sbs4),
    "SBS5": _profile(_sbs5),
    "SBS6": _profile(_sbs6),
    "SBS7a": _profile(_sbs7a),
    "SBS13": _profile(_sbs13),
    "SBS18": _profile(_sbs18),
}

#: Short aetiology notes, used in reports so a fitted exposure carries its
#: interpretation with it.
SIGNATURE_AETIOLOGY: dict[str, str] = {
    "SBS1": "spontaneous deamination of 5-methylcytosine (clock-like, age)",
    "SBS2": "APOBEC3 cytidine deaminase, transition arm",
    "SBS3": "homologous recombination deficiency (BRCA1/2)",
    "SBS4": "tobacco smoking, bulky adducts",
    "SBS5": "clock-like, aetiology unknown",
    "SBS6": "defective DNA mismatch repair, microsatellite instability",
    "SBS7a": "ultraviolet light, pyrimidine dimers",
    "SBS13": "APOBEC3 cytidine deaminase, transversion arm",
    "SBS18": "reactive oxygen species, 8-oxo-guanine",
}


@dataclass
class SignatureSet:
    """A named collection of 96-channel signature profiles."""

    names: list[str]
    matrix: np.ndarray
    """Shape ``(96, n_signatures)``; each column sums to 1."""
    provenance: str = "unspecified"
    aetiology: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape[0] != N_CHANNELS:
            raise ValueError(
                f"signature matrix must have {N_CHANNELS} rows, "
                f"got {self.matrix.shape[0]}"
            )
        if self.matrix.shape[1] != len(self.names):
            raise ValueError("number of names must match number of columns")
        if np.any(self.matrix < 0):
            raise ValueError("signature profiles must be non-negative")
        sums = self.matrix.sum(axis=0)
        if np.any(sums <= 0):
            raise ValueError("every signature must carry positive mass")
        self.matrix = self.matrix / sums

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.matrix[:, self.names.index(name)]

    def subset(self, names) -> "SignatureSet":
        idx = [self.names.index(n) for n in names]
        return SignatureSet(
            names=list(names),
            matrix=self.matrix[:, idx],
            provenance=self.provenance,
            aetiology=self.aetiology,
        )

    def similarity_matrix(self) -> np.ndarray:
        """Pairwise cosine similarity between signatures.

        Values above ~0.85 mean the pair is nearly collinear and their
        exposures cannot be separated reliably -- the reason refitting a
        full catalogue to one tumour is ill-conditioned.
        """
        m = self.matrix
        norms = np.linalg.norm(m, axis=0)
        return (m.T @ m) / np.outer(norms, norms)

    def condition_number(self) -> float:
        """Condition number of the signature matrix.

        A large value warns that the refitting problem is ill-posed even
        though NNLS will happily return an answer.
        """
        return float(np.linalg.cond(self.matrix))


def stylized_signature_set(names=None) -> SignatureSet:
    """The bundled stylized reference set (see module docstring caveat)."""
    names = list(names) if names is not None else list(STYLIZED_SIGNATURES)
    return SignatureSet(
        names=names,
        matrix=np.column_stack([STYLIZED_SIGNATURES[n] for n in names]),
        provenance="stylized profiles built from published aetiologies, "
        "NOT the COSMIC numerical catalogue",
        aetiology={n: SIGNATURE_AETIOLOGY[n] for n in names},
    )


def load_signature_matrix(path, *, sep: str = "\t") -> SignatureSet:
    """Load a COSMIC-format signature table and validate its channel order.

    The file is expected to have a first column of channel labels
    (``A[C>A]A`` style) and one further column per signature.  Rows are
    reordered to the canonical order rather than trusted, because COSMIC
    downloads have shipped in more than one row order.
    """
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=sep)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]

    names = [h.strip() for h in header[1:]]
    canonical = channel_labels()
    index = {lab: i for i, lab in enumerate(canonical)}
    matrix = np.zeros((N_CHANNELS, len(names)), dtype=np.float64)
    seen = set()
    for row in rows:
        label = row[0].strip()
        if label not in index:
            raise ValueError(f"unrecognised channel label {label!r}")
        seen.add(label)
        matrix[index[label], :] = [float(v) for v in row[1 : len(names) + 1]]
    missing = set(canonical) - seen
    if missing:
        raise ValueError(f"signature file is missing {len(missing)} channels")
    return SignatureSet(names=names, matrix=matrix, provenance=str(path))


# ----------------------------------------------------------------------
# similarity and refitting
# ----------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two spectra; 1 means identical shape."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


@dataclass(frozen=True)
class ExposureFit:
    """Result of refitting a spectrum onto a signature set."""

    names: list[str]
    exposures: np.ndarray
    """Non-negative mutation counts attributed to each signature."""
    reconstruction: np.ndarray
    residual_norm: float
    cosine: float
    """Cosine similarity between the observed and reconstructed spectrum."""

    @property
    def proportions(self) -> np.ndarray:
        total = self.exposures.sum()
        return self.exposures / total if total > 0 else self.exposures

    def as_dict(self, min_proportion: float = 0.0) -> dict[str, float]:
        return {
            n: float(e)
            for n, e, p in zip(self.names, self.exposures, self.proportions)
            if p >= min_proportion
        }


def fit_exposures(spectrum: np.ndarray, signatures: SignatureSet) -> ExposureFit:
    """Non-negative least squares refit of one spectrum.

    Parameters
    ----------
    spectrum:
        Length-96 vector of mutation counts in canonical channel order.
    signatures:
        Reference set to fit against.

    Notes
    -----
    NNLS minimises ``||S x - y||_2`` subject to ``x >= 0``.  Least squares
    implicitly assumes homoscedastic Gaussian noise, whereas mutation
    counts are Poisson; for spectra with a few hundred mutations the
    difference is small, but for sparse spectra prefer
    :func:`refit_with_selection`, whose selection step is driven by
    explained shape rather than by raw residual size.
    """
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    if y.size != N_CHANNELS:
        raise ValueError(f"spectrum must have {N_CHANNELS} entries, got {y.size}")
    if np.any(y < 0):
        raise ValueError("mutation counts must be non-negative")
    x, residual = nnls(signatures.matrix, y)
    recon = signatures.matrix @ x
    return ExposureFit(
        names=list(signatures.names),
        exposures=x,
        reconstruction=recon,
        residual_norm=float(residual),
        cosine=cosine_similarity(y, recon),
    )


def refit_with_selection(
    spectrum: np.ndarray,
    signatures: SignatureSet,
    *,
    min_cosine_gain: float = 0.01,
    max_signatures: int | None = None,
) -> ExposureFit:
    """Greedy forward selection, then a final NNLS on the chosen subset.

    Adds the signature that most improves the reconstruction cosine, and
    stops when the best remaining candidate gains less than
    ``min_cosine_gain``.  This is the standard defence against the
    well-documented failure mode in which fitting a large catalogue to a
    single tumour attributes mutations to processes that were never
    active (Maura et al. 2019): with 80 nearly-collinear columns, NNLS can
    reach a near-perfect fit using an essentially arbitrary subset.
    """
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    chosen: list[int] = []
    best_cos = 0.0
    limit = max_signatures or len(signatures)
    while len(chosen) < limit:
        best_gain, best_idx, best_new_cos = 0.0, None, best_cos
        for j in range(len(signatures)):
            if j in chosen:
                continue
            trial = chosen + [j]
            x, _ = nnls(signatures.matrix[:, trial], y)
            cos = cosine_similarity(y, signatures.matrix[:, trial] @ x)
            if cos - best_cos > best_gain:
                best_gain, best_idx, best_new_cos = cos - best_cos, j, cos
        if best_idx is None or best_gain < min_cosine_gain:
            break
        chosen.append(best_idx)
        best_cos = best_new_cos
    if not chosen:
        chosen = [int(np.argmax(signatures.matrix.T @ y))]
    subset = signatures.subset([signatures.names[j] for j in chosen])
    return fit_exposures(y, subset)


def bootstrap_exposures(
    spectrum: np.ndarray,
    signatures: SignatureSet,
    *,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Parametric bootstrap confidence intervals on exposures.

    Resamples each spectrum from ``Poisson(observed)`` and refits.  Returns
    ``{name: (median, lower 2.5%, upper 97.5%)}``.  Wide intervals on a
    signature whose point estimate looks solid are the honest signal that
    the reference set is too collinear to resolve it.
    """
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(spectrum, dtype=np.float64).ravel()
    draws = np.empty((n_boot, len(signatures)), dtype=np.float64)
    for b in range(n_boot):
        resampled = rng.poisson(y).astype(np.float64)
        draws[b] = fit_exposures(resampled, signatures).exposures
    return {
        name: (
            float(np.median(draws[:, j])),
            float(np.percentile(draws[:, j], 2.5)),
            float(np.percentile(draws[:, j], 97.5)),
        )
        for j, name in enumerate(signatures.names)
    }


# ----------------------------------------------------------------------
# de novo extraction
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractionResult:
    """Output of de novo NMF signature extraction."""

    signatures: np.ndarray
    """Shape ``(96, k)``, columns normalised to sum to 1."""
    exposures: np.ndarray
    """Shape ``(k, n_samples)``, in mutation counts."""
    divergence: float
    n_iter: int
    n_restarts: int

    def reconstruction(self) -> np.ndarray:
        return self.signatures @ self.exposures


def _kl_divergence(v: np.ndarray, wh: np.ndarray) -> float:
    wh = np.clip(wh, 1e-12, None)
    mask = v > 0
    return float(np.sum(v[mask] * np.log(v[mask] / wh[mask])) - v.sum() + wh.sum())


def extract_signatures(
    catalogue: np.ndarray,
    n_signatures: int,
    *,
    n_restarts: int = 8,
    max_iter: int = 4000,
    tol: float = 1e-7,
    rng: np.random.Generator | None = None,
) -> ExtractionResult:
    """De novo NMF extraction under a Poisson (KL) likelihood.

    Parameters
    ----------
    catalogue:
        Shape ``(96, n_samples)`` matrix of per-sample mutation counts.
    n_signatures:
        Rank ``k`` of the factorisation.  Choosing ``k`` is the hard part:
        run several values and compare stability across restarts, not the
        reconstruction error, which decreases monotonically in ``k``.

    Notes
    -----
    Multiplicative updates (Lee & Seung) for the KL objective:

    ``W <- W * ((V / WH) H^T) / (1 H^T)``, ``H <- H * (W^T (V / WH)) / (W^T 1)``

    which is the correct noise model for counts.  NMF is non-convex, so
    ``n_restarts`` random initialisations are run and the best objective
    kept; the spread across restarts is itself the diagnostic for whether
    ``k`` is supportable.
    """
    v = np.asarray(catalogue, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != N_CHANNELS:
        raise ValueError(f"catalogue must have shape ({N_CHANNELS}, n_samples)")
    if np.any(v < 0):
        raise ValueError("catalogue must be non-negative")
    if n_signatures < 1:
        raise ValueError("n_signatures must be >= 1")
    if rng is None:
        rng = np.random.default_rng()

    n_channels, n_samples = v.shape
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    scale = max(v.mean(), 1e-6)

    for _ in range(n_restarts):
        w = rng.random((n_channels, n_signatures)) * scale + 1e-6
        h = rng.random((n_signatures, n_samples)) * scale + 1e-6
        # ``prev is None`` marks "no objective evaluated yet"; seeding it
        # with inf instead would satisfy ``inf <= tol * inf`` and stop the
        # very first time the convergence test runs.
        prev: float | None = None
        it = 0
        for it in range(1, max_iter + 1):
            wh = np.clip(w @ h, 1e-12, None)
            h *= (w.T @ (v / wh)) / np.clip(w.sum(axis=0)[:, None], 1e-12, None)
            wh = np.clip(w @ h, 1e-12, None)
            w *= ((v / wh) @ h.T) / np.clip(h.sum(axis=1)[None, :], 1e-12, None)
            if it % 25 == 0:
                obj = _kl_divergence(v, w @ h)
                if prev is not None and abs(prev - obj) <= tol * max(1.0, abs(prev)):
                    prev = obj
                    break
                prev = obj
        obj = _kl_divergence(v, w @ h)
        if best is None or obj < best[0]:
            best = (obj, w.copy(), h.copy(), it)

    assert best is not None
    obj, w, h, n_iter = best
    norms = np.clip(w.sum(axis=0), 1e-12, None)
    return ExtractionResult(
        signatures=w / norms,
        exposures=h * norms[:, None],
        divergence=obj,
        n_iter=n_iter,
        n_restarts=n_restarts,
    )


# ----------------------------------------------------------------------
# forward simulation
# ----------------------------------------------------------------------
def simulate_spectrum(
    signatures: SignatureSet,
    exposures: dict[str, float] | np.ndarray,
    *,
    context_counts: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw a Poisson mutation spectrum from a signature mixture.

    Parameters
    ----------
    exposures:
        Expected mutation count per signature, either as a dict keyed by
        signature name or an array aligned with ``signatures.names``.
    context_counts:
        Optional length-96 vector of available trinucleotide sites (from
        :func:`~oncosim.dna.channels.channel_context_counts`).  When given,
        the signature profiles are reweighted by genome composition and
        renormalised -- the difference between a signature defined *per
        opportunity* and one defined *per observed mutation*.  Ignoring it
        biases CpG-driven signatures most, because CpG is ~5x depleted in
        vertebrate genomes.
    """
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(exposures, dict):
        vec = np.array(
            [float(exposures.get(n, 0.0)) for n in signatures.names], dtype=np.float64
        )
    else:
        vec = np.asarray(exposures, dtype=np.float64)
        if vec.size != len(signatures):
            raise ValueError("exposure vector length must match the signature set")
    if np.any(vec < 0):
        raise ValueError("exposures must be non-negative")

    profiles = signatures.matrix
    if context_counts is not None:
        opp = np.asarray(context_counts, dtype=np.float64)
        if opp.size != N_CHANNELS:
            raise ValueError(f"context_counts must have {N_CHANNELS} entries")
        profiles = profiles * opp[:, None]
        profiles = profiles / np.clip(profiles.sum(axis=0), 1e-12, None)

    return rng.poisson(profiles @ vec).astype(np.int64)
