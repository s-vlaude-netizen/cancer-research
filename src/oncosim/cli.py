"""Command-line entry point: ``oncosim <command>``.

Each subcommand runs one self-contained analysis and prints a table.  They
exist so the headline numbers can be reproduced without writing any code,
and so the claims in the README can be checked in one line.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _table(rows: list[tuple[str, object]], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        shown = value if isinstance(value, str) else _fmt(float(value))
        print(f"  {key:<{width}}  {shown}")


# ----------------------------------------------------------------------
def cmd_quantum(args: argparse.Namespace) -> int:
    from .quantum import (
        AT_ILLUSTRATIVE,
        GC_SLOCOMBE_2022,
        analyse_tautomer,
        crossover_temperature,
        kinetic_isotope_effect,
    )

    for potential in (GC_SLOCOMBE_2022, AT_ILLUSTRATIVE):
        analysis = analyse_tautomer(potential, args.temperature)
        rows = list(analysis.report().items())
        rows.append(("crossover_temperature_K", crossover_temperature(potential)))
        _table(rows, f"Proton transfer: {potential.label} at {args.temperature} K")
    _table(
        list(kinetic_isotope_effect(GC_SLOCOMBE_2022, args.temperature).items()),
        "Kinetic isotope effect (the testable prediction)",
    )
    return 0


def cmd_sfs(args: argparse.Namespace) -> int:
    from .evolution import (
        BirthDeath,
        expected_mutations_above,
        fit_neutral_tail,
        sample_sfs_above,
    )

    rng = np.random.default_rng(args.seed)
    process = BirthDeath(1.0, args.death)
    freqs = sample_sfs_above(args.mu, args.f_min, process, rng=rng)
    fit = fit_neutral_tail(freqs, 0.05, 0.4)
    _table(
        [
            ("mutations per division (mu)", args.mu),
            ("turnover d/b", process.turnover),
            ("effective rate mu/beta", args.mu / process.beta),
            ("mutations above f_min", len(freqs)),
            (
                "expected above f_min",
                float(expected_mutations_above(args.f_min, args.mu, process)),
            ),
            ("fitted tail exponent alpha", fit.alpha),
            ("alpha standard error", fit.alpha_stderr),
            ("consistent with neutrality", str(fit.neutral_consistent)),
        ],
        "Site frequency spectrum of a neutrally growing tumour",
    )
    return 0


def cmd_resistance(args: argparse.Namespace) -> int:
    from .evolution import BirthDeath
    from .therapy import ResistanceModel, max_curable_size

    rng = np.random.default_rng(args.seed)
    model = ResistanceModel(args.n_cells, args.mutation_rate, BirthDeath(1.0, args.death))
    quantiles = model.resistance_quantiles(n_sim=20_000, rng=rng)
    rows: list[tuple[str, object]] = [
        ("tumour size at detection", args.n_cells),
        ("resistance rate per division", args.mutation_rate),
        ("P(no resistant cell)", model.probability_no_resistance()),
    ]
    rows += [(f"resistant cells, {int(q * 100)}th pct", v) for q, v in quantiles.items()]
    rows += [
        ("max curable size, 1 drug", max_curable_size([args.mutation_rate])),
        ("max curable size, 2 drugs", max_curable_size([args.mutation_rate] * 2)),
        ("max curable size, 3 drugs", max_curable_size([args.mutation_rate] * 3)),
    ]
    _table(rows, "Pre-existing resistance at treatment start")
    return 0


def cmd_therapy(args: argparse.Namespace) -> int:
    from .therapy import CompetitionModel, compare_strategies

    del args
    results = compare_strategies(CompetitionModel())
    print("\nTreatment strategies (Lotka-Volterra, tumour at 80% of capacity)")
    print("-" * 63)
    print(f"  {'strategy':<14}{'TTP (days)':>12}{'drug fraction':>16}{'final R frac':>15}")
    for name, r in results.items():
        print(
            f"  {name:<14}{r.time_to_progression:>12.0f}"
            f"{r.treatment_fraction:>16.2f}{r.final_resistant_fraction():>15.3f}"
        )
    return 0


def cmd_prevention(args: argparse.Namespace) -> int:
    from .therapy import ScreeningProgramme, cessation_benefit, smoking_mutation_burden

    print("\nSmoking cessation (Doll-Peto, upper bound on benefit)")
    print("-" * 53)
    print(f"  {'quit at age':<14}{'lifetime risk':>16}{'vs continuing':>16}")
    for age in (30.0, 40.0, 50.0, 60.0):
        b = cessation_benefit(age, args.cigarettes)
        print(f"  {age:<14.0f}{b['risk_if_quit']:>16.4f}{b['relative_risk']:>16.3f}")

    _table(
        list(smoking_mutation_burden(args.pack_years, "lung", age=60.0).items()),
        f"Mutational burden of {args.pack_years:.0f} pack-years (lung)",
    )

    print("\nScreening interval trade-off (mean sojourn 3 y, sensitivity 0.85)")
    print("-" * 64)
    print(f"  {'interval (y)':<14}{'screen-detected':>18}{'screens/person':>17}")
    for interval in (0.5, 1.0, 2.0, 3.0, 5.0):
        s = ScreeningProgramme(3.0, 0.85, interval).summary()
        print(
            f"  {interval:<14.1f}{s['screen_detected_fraction']:>18.3f}"
            f"{s['screens_per_person']:>17.1f}"
        )
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oncosim",
        description="Theoretical cancer research toolkit: reproduce the headline numbers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("quantum", help="proton transfer and spontaneous mutation")
    p.add_argument("--temperature", type=float, default=310.15)
    p.set_defaults(func=cmd_quantum)

    p = sub.add_parser("sfs", help="site frequency spectrum of a growing tumour")
    p.add_argument("--mu", type=float, default=4.0)
    p.add_argument("--death", type=float, default=0.4)
    p.add_argument("--f-min", dest="f_min", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sfs)

    p = sub.add_parser("resistance", help="pre-existing resistance and combinations")
    p.add_argument("--n-cells", dest="n_cells", type=float, default=1e9)
    p.add_argument("--mutation-rate", dest="mutation_rate", type=float, default=1e-7)
    p.add_argument("--death", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_resistance)

    p = sub.add_parser("therapy", help="compare treatment strategies")
    p.set_defaults(func=cmd_therapy)

    p = sub.add_parser("prevention", help="cessation, exposure and screening")
    p.add_argument("--cigarettes", type=float, default=20.0)
    p.add_argument("--pack-years", dest="pack_years", type=float, default=30.0)
    p.set_defaults(func=cmd_prevention)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
