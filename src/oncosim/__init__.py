"""oncosim -- theoretical cancer research toolkit.

A numerical laboratory for the question "how does a cancer arise, and what
would stop it", spanning four scales that are usually studied in
isolation:

``oncosim.quantum``
    Proton transfer in Watson-Crick base pairs: the quantum-mechanical
    origin of spontaneous point mutations.  Solves the Schroedinger
    equation for the hydrogen-bond proton and converts tautomer
    populations into a per-base-pair, per-replication mutation rate.

``oncosim.dna``
    Sequence-level mutational processes: trinucleotide-context-dependent
    rates, 96-channel mutational signatures, exposure refitting and de
    novo extraction.

``oncosim.evolution``
    Somatic evolution: linear birth-death theory, site frequency spectra,
    Luria-Delbruck fluctuation theory, multi-type clonal dynamics.

``oncosim.spatial``
    Spatial tumour growth: surface-limited growth, turnover and
    short-range dispersal, virtual biopsies.

``oncosim.therapy``
    Resistance evolution, combination and adaptive therapy, and
    prevention: carcinogen exposure, cessation, screening.

Design rules
------------
1. Every simulator is validated against a closed-form result in
   ``tests/``; where no closed form exists, against a slower brute-force
   reference implementation in the same module.
2. Every model states its parameter conventions in the docstring,
   especially the ones the literature is inconsistent about (mutations per
   division vs. per daughter genome; cell fraction vs. VAF).
3. Randomness always flows through an explicit
   :class:`numpy.random.Generator`, so every figure is reproducible from
   its seed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
