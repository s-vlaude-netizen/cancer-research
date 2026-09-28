# oncosim — theoretical cancer research toolkit

A numerical laboratory for one question: **how does a cancer arise, and what
would stop it?** The package spans four scales that are normally studied in
separate literatures, and it connects them with explicit numbers rather than
hand-waving.

| scale | module | what it computes |
|---|---|---|
| quantum | `oncosim.quantum` | the hydrogen-bond proton's wavefunction in a Watson-Crick base pair → tautomer population → per-base-pair mutation rate |
| sequence | `oncosim.dna` | 96-channel mutational signatures, exposure refitting, de novo extraction |
| population | `oncosim.evolution` | birth-death theory, site frequency spectra, Luria-Delbrück, driver dynamics |
| tissue | `oncosim.spatial` | surface-limited growth, turnover, short-range dispersal, virtual biopsies |
| clinic | `oncosim.therapy` | resistance, combination and adaptive therapy, exposure, cessation, screening |

```bash
pip install -e .
pytest                  # 109 tests, ~70 s
oncosim quantum         # or: sfs, resistance, therapy, prevention
```

---

## The cross-scale consistency check

The reason to build all of this in one package is that the scales constrain
each other, and the constraint is quantitative.

The quantum module says a G-C base pair sits in its tautomeric form with
probability **4.8 × 10⁻⁸** at body temperature. A haploid human genome has
~1.3 × 10⁹ G-C pairs, so about **61 templates per replication** present the
polymerase with a mispairing geometry. For that channel to produce the
observed ~10⁻⁸ mutations per base pair per division, the fraction surviving
proofreading and mismatch repair must be about **0.2** — squarely in the
plausible range.

Multiply through and you get a few mutations per cell division genome-wide,
which is exactly the `mu ≈ 1–10` that the site frequency spectrum module needs
to reproduce real tumour spectra. The scales agree without either being tuned
to the other.

---

## Five things this code establishes

**1. Tunnelling changes the clock, not the equilibrium.**
For a 1-D barrier the tunnelling correction `kappa(T)` multiplies the forward
and reverse rates by the *same* factor, so it cancels exactly from `K_eq` —
verified here to nine digits. Any claim that proton tunnelling *raises* the
tautomer population is a claim about non-equilibrium dynamics or zero-point
energy, not about tunnelling. At body temperature `kappa = 1.42`; the crossover
temperature for this barrier is 139 K, so 310 K is thermally activated, not a
deep-tunnelling regime.

**2. The tautomer population is ~10⁻⁸, not ~10⁻⁴.**
Three independent routes — the exact canonical density over DVR eigenstates
(4.8e-8), the classical configurational integral (6.5e-8), and the kinetic
ratio `k_f/k_r` (6.4e-8) — agree. The widely cited 1.73e-4 from a
Wigner–Caldeira–Leggett steady state sits four orders of magnitude above all
three; that paper's own rate constants give `k_f/k_r = 4.5e-8`. Zero-point
energy in fact *suppresses* the population here, because the tautomeric well is
the narrower of the two. See `oncosim/quantum/tautomer.py` for the full
argument, including the biological consistency check that breaks the tie.

**3. Don't fit an SFS tail with log-log regression.**
The cumulative spectrum is `M(f) = A(f^-alpha - 1)`. That additive `-1` is not
negligible inside a realistic fitting window: over VAF 0.05–0.4 it biases a
log-log slope to `alpha = 1.22` on data that are *exactly neutral* — enough to
manufacture apparent selection out of nothing. `fit_neutral_tail` uses maximum
likelihood on the doubly-truncated Pareto density instead, and is verified
unbiased at `alpha` = 0.5, 0.75, 1.0 and 1.4.

**4. Containment beats maximum tolerated dose — but only near carrying capacity.**
At 80% of carrying capacity, holding the tumour at its initial burden reaches
progression at 2255 days versus 1176 for continuous MTD, using **23% less
drug**. Two mechanisms are usually conflated: competition for capacity (which
operates even at zero resistance cost and dominates here) and the cost of
resistance (which adds ~40% on top). Below ~50% of carrying capacity the
logistic term is near 1, there is no competition to exploit, and containment is
no better than no treatment at all — `CompetitionModel.competition_strength()`
is the diagnostic.

**5. A single targeted drug cannot cure a detectable tumour.**
`P(no resistant cell) = exp(-uM)` exactly, with the turnover factor cancelling
(more divisions per net cell added, but each new lineage less likely to
survive). At `u = 10⁻⁷` a single drug loses curative potential above ~7 × 10⁶
cells — two orders of magnitude *below* the imaging detection limit. A two-drug
combination pushes that to ~2 × 10¹². This is the Luria–Delbrück argument
transplanted to oncology, and the resistant cell count is reported as
quantiles, never a mean: the law has an infinite mean.

---

## Algorithms

Three pieces of numerical machinery do most of the work.

**Exact Luria–Delbrück in closed form.** The clone-size law for arbitrary
mutant fitness `rho` is `q_k = B(1+1/rho, k)/rho`, which for `rho = 1` collapses
to the classical `1/(k(k+1))`. This yields (a) **O(1) exact sampling** — a
fluctuation assay with N = 10¹² costs the same as N = 10³; (b) the
Ma–Sandri–Sarkar recursion as a special case of Panjer's; (c) an O(K log K) FFT
inversion on a damped contour, agreeing with the recursion to 10⁻¹⁰.

**Size-independent SFS sampling.** All mutations above a detection threshold
`f_min` can be drawn exactly in `O(mu/f_min)` — *independent of tumour size*.
The count is Poisson with mean `M(f_min)` and the frequencies are i.i.d. from
the normalised `f^-2` density, which inverts in closed form. A 10¹¹-cell
tumour's detectable spectrum costs the same as a 10⁴-cell one, and the result
is validated against a brute-force cell-by-cell genealogy.

**Hybrid adaptive tau-leaping.** Clone selection runs through a Fenwick tree
(O(log K) per event), and the tau-leap advances all large clones by Poisson
draws with Cao–Gillespie–Petzold step selection while clones below
`n_critical` are advanced by exact SSA within the same interval. That
partition is not a refinement but a requirement: a Poisson leap can drive a
clone of size 3 negative, and small clones carry the extinction dynamics that
decide whether a driver establishes. Result: 26× faster than exact SSA at 10⁵
cells, agreeing on time-to-target within 0.06 standard errors.

---

## Validation

Every simulator is checked against a closed form, or against a slower
brute-force reference in the same module. A few anchors:

| claim | check |
|---|---|
| SSA is exact | event count matches `(N-1)(b+d)/(b-d)` to 0.03% |
| the 1/f law holds with turnover | genealogy vs `(mu/beta)(1/f - 1)` within 2–4% at `d/b` = 0, 0.4, 0.8 |
| the G-C surface is the published one | 5 numbers it was not fitted to (barriers, asymmetry, `omega_b`, zero-point energy, the 7th eigenstate being the first tautomeric one) |
| tunnelling respects detailed balance | `kappa_forward / kappa_reverse = 1` to 9 digits |
| spatial growth is a real sphere | radius of gyration within 2% of `sqrt(3/5) R` |
| Armitage–Doll is implemented right | log-log incidence slope exactly `k-1` |

Where an effect is real but noisy, it is **not** asserted as a unit test.
Dispersal's reduction of intratumour heterogeneity is ~2 standard errors at 8
replicates, so the test suite asserts only its sharp consequence (dispersal
makes the mass measurably less compact, ~9 standard errors) and the diversity
comparison lives in the experiments directory with enough replicates to resolve
it.

---

## Caveats worth reading before citing anything

- **The bundled mutational signatures are stylized**, built from published
  descriptions of which channels carry each process's peaks — not the COSMIC
  numerical catalogue, which carries its own licence. Use
  `load_signature_matrix()` with a real catalogue for quantitative work.
- **The A-T proton-transfer surface is illustrative**, retuned to have a small
  reverse barrier. Only the G-C surface is a published fit.
- **The cessation model overstates the benefit of quitting.** It freezes the
  Doll–Peto hazard at cessation; real ex-smokers keep ageing. Read the ordering
  across quitting ages, not the absolute risks.
- **`mu` means mutations per division event**, summed over both daughters. If
  your source quotes mutations per daughter genome, double it. This factor of
  two is the most common silent error in SFS-based rate estimates.
- **Cell fraction is not VAF.** Diploid heterozygous gives `VAF = f/2`; use
  `vaf_from_cell_fraction()`.

## Layout

```
src/oncosim/
  algorithms/   Fenwick tree for O(log n) weighted sampling
  quantum/      double-well potential, DVR solver, tunnelling, tautomer → mutation rate
  dna/          96 channels, context counting, signature refitting and NMF
  evolution/    birth-death theory, Luria-Delbrück, SFS, clonal SSA and tau-leap
  spatial/      3-D lattice growth with turnover and dispersal
  therapy/      resistance, adaptive therapy, prevention
tests/          109 tests, all against closed forms or brute-force references
experiments/    runnable studies that produce figures
docs/           literature notes and derivations
```

## References

The primary sources each model is built on are cited in the module docstrings,
with the specific result being used. The main ones: Kendall (1948) for the
birth-death process; Luria & Delbrück (1943) and Ma, Sandri & Sarkar (1992);
Armitage & Doll (1954); Löwdin (1963) and Slocombe, Sacchi & Al-Khalili (2022)
for proton transfer; Alexandrov et al. (2013, 2016, 2020) for signatures;
Bozic et al. (2010, 2013); Durrett (2013, 2015) and Williams et al. (2016) for
site frequency spectra; Waclaw et al. (2015) for spatial structure; Gatenby et
al. (2009) and Zhang et al. (2017) for adaptive therapy.
