"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grain.manuscript import ABLATION, MAIN_RESULTS, REGRET_CURVE
from grain.simulate import SimSummary

INK = "#1a1814"
SEA = "#1e3a4c"
ACCENT = "#8f3d1e"
MUTED = "#6d655c"
PAPER = "#f7f4ee"

def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#e3dacb",
            "grid.linewidth": 0.6,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

def save_manuscript_figures(out_dir: Path) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ux, uy = zip(*REGRET_CURVE["grain"])
    bx, by = zip(*REGRET_CURVE["baseline"])
    ax.plot(ux, uy, color=SEA, lw=2.0, marker="o", label="GRAIN")
    ax.plot(bx, by, color=ACCENT, lw=1.6, ls="--", marker="s", label="Strongest baseline")
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative dynamic regret")
    ax.set_title("GRAIN — manuscript regret curves (exact TikZ knots)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fig_manuscript_regret.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    names = [row.method for row in MAIN_RESULTS]
    sw = [row.sw for row in MAIN_RESULTS]
    colors = [ACCENT if row.ours else SEA for row in MAIN_RESULTS]
    ax.bar(names, sw, color=colors)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Social welfare")
    ax.set_title("GRAIN — Beijing KnowAir social welfare")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "fig_manuscript_sw.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    labels = [row[0] for row in ABLATION]
    xs = range(len(labels))
    ax1.bar(xs, [row[1] for row in ABLATION], color=SEA, label="SW")
    ax1.set_ylim(0.7, 1.0)
    ax1.set_ylabel("Social welfare", color=SEA)
    ax2 = ax1.twinx()
    ax2.plot(xs, [row[2] for row in ABLATION], color=ACCENT, marker="o", label="Regret")
    ax2.set_ylabel("Dynamic regret", color=ACCENT)
    ax1.set_xticks(list(xs), labels, rotation=30, ha="right")
    ax1.set_title("GRAIN — manuscript ablation")
    fig.tight_layout()
    path = out_dir / "fig_manuscript_ablation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written

def save_simulation_figures(summary: SimSummary, out_dir: Path, tag: str) -> list[Path]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ts = [p.t for p in summary.rounds]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ts, [p.cum_regret for p in summary.rounds], color=ACCENT, lw=1.6)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative dynamic regret")
    ax.set_title(f"GRAIN reference run — {tag}")
    fig.tight_layout()
    path = out_dir / f"fig_sim_regret_{tag}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    keys = ["sw", "part", "fair", "eng", "cov", "aoi"]
    labels = ["SW ratio", "Participation", "Fairness", "Energy score", "Coverage", "AoI score"]
    vals = [summary.objectives[k] for k in keys]
    ax.bar(labels, vals, color=SEA)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title(f"GRAIN multi-objective snapshot — {tag}")
    fig.tight_layout()
    path = out_dir / f"fig_sim_objectives_{tag}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written

def write_round_csv(summary: SimSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t",
        "sigma",
        "alpha",
        "path",
        "sw",
        "sw_oracle",
        "cum_regret",
        "truth_cost",
        "truth_quality",
        "utility_gain",
        "participation",
        "fairness",
        "energy",
        "coverage",
        "aoi",
        "gamma",
        "monotone",
        "step",
    ]
    lines = [",".join(header)]
    for p in summary.rounds:
        row = [
            p.t,
            p.sigma,
            p.alpha,
            p.path,
            p.sw,
            p.sw_oracle,
            p.cum_regret,
            p.truth_cost,
            p.truth_quality,
            p.utility_gain,
            p.participation,
            p.fairness,
            p.energy,
            p.coverage,
            p.aoi,
            p.gamma,
            int(p.monotone),
            p.step,
        ]
        lines.append(",".join(f"{v:.8f}" if isinstance(v, float) else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
