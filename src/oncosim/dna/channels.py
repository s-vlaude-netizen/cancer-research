"""The 96 single-base-substitution channels and trinucleotide contexts.

A somatic point mutation is classified by its substitution *and* by the
bases immediately 5' and 3' of it.  Because a mutation on one strand is
the same event as its complement on the other, the reference base is
always written as a pyrimidine (C or T); purine references are reverse
complemented.  That leaves ``6 substitutions x 4 x 4 flanks = 96``
channels, the representation in which mutational signatures are defined.

Channel ordering follows the COSMIC convention exactly:

    A[C>A]A, A[C>A]C, A[C>A]G, A[C>A]T, C[C>A]A, ..., T[T>G]T

i.e. substitution type varies slowest (C>A, C>G, C>T, T>A, T>C, T>G),
then the 5' base, then the 3' base, each in alphabetical order.  Getting
this order wrong silently permutes every signature, so
:func:`channel_labels` is the single source of truth and everything else
indexes through it.

References
----------
Alexandrov et al. (2013) Nature 500:415-421 (the 96-channel framework).
Alexandrov et al. (2020) Nature 578:94-101 (PCAWG signature catalogue).
COSMIC Mutational Signatures, https://cancer.sanger.ac.uk/signatures/
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BASES",
    "SUBSTITUTIONS",
    "N_CHANNELS",
    "channel_labels",
    "channel_index",
    "substitution_of_channel",
    "context_of_channel",
    "encode_sequence",
    "reverse_complement",
    "count_trinucleotides",
    "trinucleotide_labels",
    "pyrimidine_context_counts",
    "channel_context_counts",
]

BASES = ("A", "C", "G", "T")

#: The six pyrimidine-referenced substitution types, in COSMIC order.
SUBSTITUTIONS = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")

N_CHANNELS = 96

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def reverse_complement(seq: str) -> str:
    """Reverse complement of an IUPAC-free DNA string; ``N`` is preserved."""
    return "".join(_COMPLEMENT[b] for b in reversed(seq.upper()))


def channel_labels() -> list[str]:
    """The 96 channel labels in canonical COSMIC order, e.g. ``'A[C>A]A'``."""
    labels = []
    for sub in SUBSTITUTIONS:
        for five in BASES:
            for three in BASES:
                labels.append(f"{five}[{sub}]{three}")
    return labels


_LABELS = channel_labels()
_LABEL_TO_INDEX = {lab: i for i, lab in enumerate(_LABELS)}


def channel_index(five: str, ref: str, alt: str, three: str) -> int:
    """Index of the channel for ``five[ref>alt]three``, folding purines.

    Raises
    ------
    ValueError
        If the substitution is not a real substitution (``ref == alt``) or
        any base is outside ``ACGT``.
    """
    five, ref, alt, three = (b.upper() for b in (five, ref, alt, three))
    if ref == alt:
        raise ValueError("reference and alternate base are identical")
    for b in (five, ref, alt, three):
        if b not in _BASE_CODE:
            raise ValueError(f"unsupported base {b!r}; expected one of ACGT")
    if ref in ("A", "G"):  # fold to the pyrimidine strand
        five, ref, alt, three = (
            _COMPLEMENT[three],
            _COMPLEMENT[ref],
            _COMPLEMENT[alt],
            _COMPLEMENT[five],
        )
    return _LABEL_TO_INDEX[f"{five}[{ref}>{alt}]{three}"]


def substitution_of_channel(index: int) -> str:
    """``'C>T'`` etc. for a channel index."""
    return SUBSTITUTIONS[index // 16]


def context_of_channel(index: int) -> str:
    """The trinucleotide context of a channel, e.g. ``'ACG'`` for ``A[C>T]G``."""
    sub = SUBSTITUTIONS[index // 16]
    rem = index % 16
    return f"{BASES[rem // 4]}{sub[0]}{BASES[rem % 4]}"


# ----------------------------------------------------------------------
# sequence handling
# ----------------------------------------------------------------------
def encode_sequence(seq: str) -> np.ndarray:
    """Encode a DNA string as ``int8`` codes ``A=0, C=1, G=2, T=3``; other -> -1.

    Vectorised through a 256-entry lookup table, so a 250 Mb chromosome
    encodes in well under a second and without a Python-level loop.
    """
    table = np.full(256, -1, dtype=np.int8)
    for base, code in _BASE_CODE.items():
        table[ord(base)] = code
        table[ord(base.lower())] = code
    raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    return table[raw]


def trinucleotide_labels() -> list[str]:
    """All 64 trinucleotides in ``AAA, AAC, ..., TTT`` order."""
    return [f"{a}{b}{c}" for a in BASES for b in BASES for c in BASES]


def count_trinucleotides(codes: np.ndarray) -> np.ndarray:
    """Counts of all 64 trinucleotides in an encoded sequence.

    ``codes`` is the output of :func:`encode_sequence`.  Positions
    involving an unknown base (code ``-1``) are skipped.  Cost is ``O(L)``
    with three vector operations and one ``bincount`` -- no sliding
    window in Python.
    """
    codes = np.asarray(codes)
    if codes.size < 3:
        return np.zeros(64, dtype=np.int64)
    left, mid, right = codes[:-2], codes[1:-1], codes[2:]
    valid = (left >= 0) & (mid >= 0) & (right >= 0)
    idx = (left[valid].astype(np.int64) * 16) + (mid[valid] * 4) + right[valid]
    return np.bincount(idx, minlength=64).astype(np.int64)


def pyrimidine_context_counts(codes: np.ndarray) -> np.ndarray:
    """Counts of the 32 pyrimidine-referenced trinucleotide contexts.

    Returns a length-32 vector ordered as ``NCN`` contexts first (16 of
    them, 5' base slowest) then ``NTN`` contexts, matching the layout that
    the 96 channels use in blocks of 16.  A ``NGN`` context on the given
    strand is counted as its reverse complement ``NCN``, because the
    mutation would be reported on the pyrimidine strand.

    This is the denominator every signature analysis needs: a genome is not
    uniform in trinucleotide composition (CpG is ~5x depleted in
    vertebrates), so raw channel counts must be normalised by it before
    they can be compared with an exposure model.
    """
    tri = count_trinucleotides(codes)
    out = np.zeros(32, dtype=np.int64)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                n = tri[a * 16 + b * 4 + c]
                if n == 0:
                    continue
                if b == 1:  # C in the middle
                    out[a * 4 + c] += n
                elif b == 3:  # T in the middle
                    out[16 + a * 4 + c] += n
                elif b == 2:  # G -> revcomp gives C, flanks swap and complement
                    out[(3 - c) * 4 + (3 - a)] += n
                else:  # A -> revcomp gives T
                    out[16 + (3 - c) * 4 + (3 - a)] += n
    return out


def channel_context_counts(codes: np.ndarray) -> np.ndarray:
    """The 32 context counts expanded to the 96 channels.

    Each context appears in three channels (one per alternate base), so
    the length-96 result repeats the 32-vector in the right pattern.  Use
    it to convert a per-site mutation probability into an expected channel
    count.
    """
    ctx32 = pyrimidine_context_counts(codes)
    out = np.empty(N_CHANNELS, dtype=np.int64)
    for i in range(N_CHANNELS):
        block = i // 16  # 0,1,2 -> C>*, 3,4,5 -> T>*
        within = i % 16
        out[i] = ctx32[within] if block < 3 else ctx32[16 + within]
    return out
