"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from grain.data import SOURCES, load_dataset
from grain.manuscript import ABLATION, COMPONENT_GAINS, MAIN_RESULTS, RELATED, verify_manuscript
from grain.plots import save_manuscript_figures, save_simulation_figures, write_round_csv
from grain.simulate import ABLATION_SWITCHES, SimSummary, default_config, run_grain

def _print_checks() -> int:
    checks = verify_manuscript()
    bad = [c for c in checks if not c.ok]
    for check in checks:
        mark = "ok " if check.ok else "FAIL"
        print(f"[{mark}] {check.id}: {check.detail}")
    print(f"{len(checks) - len(bad)}/{len(checks)} manuscript digits match")
    return 1 if bad else 0

def _summary_dict(summary: SimSummary) -> dict[str, float]:
    return {
        "social_welfare_ratio": summary.social_welfare_ratio,
        "dynamic_regret": summary.dynamic_regret,
        "mean_truth_cost": summary.mean_truth_cost,
        "mean_truth_quality": summary.mean_truth_quality,
        "mean_utility_gain": summary.mean_utility_gain,
        "mean_participation": summary.mean_participation,
        "mean_fairness": summary.mean_fairness,
        "mean_energy": summary.mean_energy,
        "mean_coverage": summary.mean_coverage,
        "mean_aoi_score": summary.mean_aoi_score,
        "final_path": summary.final_path,
        "final_sigma": summary.final_sigma,
        "final_alpha": summary.final_alpha,
        "lipschitz": summary.lipschitz,
        "mean_step": summary.mean_step,
        "monotone_fraction": summary.monotone_fraction,
        "epsilon_bound_unit_constant": summary.epsilon_bound,
        "hypervolume": summary.hypervolume,
        **{f"obj_{k}": v for k, v in summary.objectives.items()},
    }

def _print_summary(tag: str, summary: SimSummary) -> None:
    print(f"\n== reference run: {tag} ==")
    for key, value in _summary_dict(summary).items():
        print(f"  {key:32s} {value:.6f}")

def _write_manuscript_tables(out: Path) -> None:
    table = out / "tables"
    table.mkdir(parents=True, exist_ok=True)
    lines = ["method,sw,sw_std,dyn_regret,truth_dev"]
    for row in MAIN_RESULTS:
        lines.append(f"{row.method},{row.sw:.2f},{row.sw_std:.2f},{row.regret:.1f},{row.truth:.2f}")
    (table / "knowair_main.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["variant,sw,dyn_regret"]
    for name, sw, regret, _ours in ABLATION:
        lines.append(f"\"{name}\",{sw:.2f},{regret:.1f}")
    (table / "ablation.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["module,metric,printed"]
    for module, metric, printed in COMPONENT_GAINS:
        lines.append(f"\"{module}\",\"{metric}\",\"{printed}\"")
    (table / "component_gains.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    header = "method,year,multi_param,approx_dsic,dyn_regret,graph,pareto,collusion"
    body = [",".join(row) for row in RELATED]
    (table / "related.csv").write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing"
    )
    parser.add_argument("--dataset", choices=["air", "traffic"], default="air")
    parser.add_argument("--sigma0", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260301)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--check", action="store_true", help="verify printed manuscript digits and exit")
    parser.add_argument("--all", action="store_true", help="manuscript figures, both datasets, and ablations")
    parser.add_argument("--ablation", action="store_true", help="run every ablation switch on --dataset")
    args = parser.parse_args(argv)
    code = _print_checks()
    if args.check and not args.all:
        return code
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    save_manuscript_figures(out)
    _write_manuscript_tables(out)
    (out / "sources.json").write_text(json.dumps(SOURCES, indent=2) + "\n", encoding="utf-8")
    datasets = ["air", "traffic"] if args.all else [args.dataset]
    for name in datasets:
        trace = load_dataset(name)
        per = 3 if trace.kind == "air" else 2
        summary = run_grain(trace, default_config(sigma0=args.sigma0, seed=args.seed, workers_per_region=per))
        _print_summary(name, summary)
        write_round_csv(summary, out / f"rounds_{name}.csv")
        save_simulation_figures(summary, out, name)
        (out / f"summary_{name}.json").write_text(
            json.dumps(_summary_dict(summary), indent=2) + "\n", encoding="utf-8"
        )
    if args.all or args.ablation:
        trace = load_dataset(args.dataset if not args.all else "air")
        per = 3 if trace.kind == "air" else 2
        lines = ["variant,social_welfare_ratio,dynamic_regret,mean_utility_gain,mean_truth_cost"]
        print("\n== ablations on air ==")
        for name, switches in ABLATION_SWITCHES.items():
            cfg = default_config(sigma0=args.sigma0, seed=args.seed, workers_per_region=per, **switches)
            summary = run_grain(trace, cfg)
            lines.append(
                f"\"{name}\",{summary.social_welfare_ratio:.6f},{summary.dynamic_regret:.6f},"
                f"{summary.mean_utility_gain:.6f},{summary.mean_truth_cost:.6f}"
            )
            print(
                f"  {name:36s} SW {summary.social_welfare_ratio:.4f}  "
                f"regret {summary.dynamic_regret:.3f}"
            )
        (out / "tables" / "reference_ablation_air.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote figures and tables under {out.resolve()}")
    return code
if __name__ == "__main__":
    sys.exit(main())
