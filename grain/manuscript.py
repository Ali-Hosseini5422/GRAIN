"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from grain.formulas import round1

@dataclass(frozen=True)

class MainRow:
    method: str
    sw: float
    sw_std: float
    regret: float
    truth: float
    ours: bool = False
MAIN_RESULTS: tuple[MainRow, ...] = (
    MainRow("VCG", 0.72, 0.04, 148.2, 0.31),
    MainRow("Stackelberg", 0.75, 0.03, 132.7, 0.28),
    MainRow("SA-WGRA", 0.78, 0.03, 119.4, 0.22),
    MainRow("RA-ABC", 0.81, 0.02, 105.6, 0.19),
    MainRow("SQIM", 0.83, 0.02, 98.1, 0.17),
    MainRow("MHDIM", 0.84, 0.02, 94.3, 0.15),
    MainRow("GRAIN", 0.96, 0.01, 68.7, 0.04, ours=True),
)
ABLATION: tuple[tuple[str, float, float, bool], ...] = (
    ("Full GRAIN", 0.93, 42.1, True),
    ("w/o GSSM", 0.84, 56.4, False),
    ("w/o RCAS", 0.86, 48.9, False),
    ("w/o Hierarchy", 0.83, 49.7, False),
    ("Exact (non-smoothed) payments", 0.91, 43.8, False),
    ("w/o Collusion-Resist.", 0.90, 44.2, False),
    ("w/o Anticipation", 0.88, 47.1, False),
    ("w/o Adaptive MoE", 0.89, 45.6, False),
    ("Fixed learning rate", 0.88, 51.2, False),
    ("Single-parameter reduction", 0.79, 61.5, False),
)
REGRET_CURVE = {
    "grain": ((0, 0), (200, 18), (400, 32), (600, 41), (800, 48), (1000, 54)),
    "baseline": ((0, 0), (200, 28), (400, 52), (600, 71), (800, 88), (1000, 102)),
}
RELATED: tuple[tuple[str, ...], ...] = (
    ("MST/MMT", "2015", "No", "Exact (1-param)", "—", "No", "No", "No"),
    ("SA-WGRA", "2024", "No", "No", "—", "Social", "No", "No"),
    ("MHDIM", "2025", "Partial", "No", "—", "Multi-plat.", "No", "No"),
    ("HMOM-MCS", "2026", "Partial", "No", "—", "No", "Yes", "No"),
    ("MOFARO", "2026", "Partial", "No", "—", "No", "Yes", "No"),
    ("DPI-MDR", "2026", "Yes", "Detection", "—", "No", "Partial", "Partial"),
    ("GC-MoE", "2026", "No", "No", "—", "Partial", "No", "No"),
    ("GRAIN", "2026", "Yes", "ε-DSIC", "Õ(T^(2/3) P_T^(1/3))", "GSSM", "RCAS", "Yes"),
)
COMPONENT_GAINS: tuple[tuple[str, str, str], ...] = (
    ("integrated vs strongest single-aspect baseline", "social welfare", "12.8–26.8%"),
    ("vs MHDIM on KnowAir", "social welfare", "+14.3%"),
    ("vs MHDIM on KnowAir", "dynamic regret", "−27.1%"),
    ("RCAS vs HMOM-MCS", "social welfare", "+19.4–26.8%"),
    ("RCAS", "participation rate", "≈ +14%"),
    ("RCAS", "Jain fairness", "+0.07–0.11"),
    ("RCAS", "energy", "−18.5%"),
    ("adaptive MoE", "task-completion rate", "+8.4–12.7%"),
    ("adaptive MoE", "latency", "−19–27%"),
    ("adaptive MoE", "FLOPs", "−31%"),
    ("asymmetric grouping", "collusion success", "−41%"),
    ("grouping + anticipation", "truthfulness deviation", "< 3.1%"),
    ("constituent modules", "runtime vs fastest baseline", "1.4–1.6×"),
    ("removing GSSM vs Full", "regret increase", "34%"),
    ("smoothing sensitivity", "ε for σ ∈ [0.05, 0.15]", "≤ 0.05"),
)

@dataclass(frozen=True)

class Check:
    id: str
    ok: bool
    detail: str

def verify_manuscript() -> list[Check]:
    mhd = next(row for row in MAIN_RESULTS if row.method == "MHDIM")
    ours = next(row for row in MAIN_RESULTS if row.method == "GRAIN")
    full = next(row for row in ABLATION if row[0] == "Full GRAIN")
    no_g = next(row for row in ABLATION if row[0] == "w/o GSSM")
    sw_gain = ((ours.sw - mhd.sw) / mhd.sw) * 100.0
    reg_drop = ((mhd.regret - ours.regret) / mhd.regret) * 100.0
    gssm_inc = ((no_g[2] - full[2]) / full[2]) * 100.0
    sw_product = mhd.sw * 1.143
    reg_product = mhd.regret * (1.0 - 0.271)
    checks = [
        Check(
            "sw-14.3",
            round1(sw_gain) == 14.3,
            f"({ours.sw} - {mhd.sw}) / {mhd.sw} = {sw_gain:.6f}% -> {round1(sw_gain):.1f}%",
        ),
        Check(
            "reg-27.1",
            round1(reg_drop) == 27.1,
            f"({mhd.regret} - {ours.regret}) / {mhd.regret} = {reg_drop:.6f}% -> {round1(reg_drop):.1f}%",
        ),
        Check(
            "sw-product",
            abs(sw_product - ours.sw) < 5e-4,
            f"0.84 * 1.143 = {sw_product:.5f}, table = {ours.sw}",
        ),
        Check(
            "reg-product",
            round1(reg_product) == ours.regret,
            f"94.3 * 0.729 = {reg_product:.6f} -> {round1(reg_product):.1f}",
        ),
        Check(
            "gssm-34",
            round1(gssm_inc) == 34.0,
            f"({no_g[2]} - {full[2]}) / {full[2]} = {gssm_inc:.6f}% -> {round1(gssm_inc):.0f}%",
        ),
        Check(
            "eps-cap",
            ours.truth <= 0.05 and ours.truth == 0.04,
            f"Truth. Dev. = {ours.truth:.2f} <= 0.05",
        ),
        Check(
            "curve-end",
            REGRET_CURVE["grain"][-1] == (1000, 54)
            and REGRET_CURVE["baseline"][-1] == (1000, 102)
            and len(REGRET_CURVE["grain"]) == 6
            and len(REGRET_CURVE["baseline"]) == 6,
            "six TikZ knots on each curve",
        ),
        Check(
            "ordering",
            all(MAIN_RESULTS[i].sw > MAIN_RESULTS[i - 1].sw for i in range(1, len(MAIN_RESULTS))),
            " < ".join(f"{row.sw:.2f}" for row in MAIN_RESULTS),
        ),
        Check(
            "regret-order",
            all(MAIN_RESULTS[i].regret < MAIN_RESULTS[i - 1].regret for i in range(1, len(MAIN_RESULTS))),
            " > ".join(f"{row.regret:.1f}" for row in MAIN_RESULTS),
        ),
        Check(
            "ablation-best",
            full[1] == max(row[1] for row in ABLATION) and full[2] == min(row[2] for row in ABLATION),
            f"SW {full[1]:.2f}, Regret {full[2]:.1f}",
        ),
        Check("std", ours.sw_std == 0.01, f"+/- {ours.sw_std:.2f}"),
    ]
    return checks
