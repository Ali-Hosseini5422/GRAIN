"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from grain.formulas import (

    ALPHA_COLLUSION,
    BETA_ANTICIPATION,
    EXPERTS,
    GAMMA_PATH,
    GROUP_SIZE,
    HEADS,
    HIDDEN,
    LAMBDA_REG,
    TOP_K,
    anticipation_factor,
    epsilon_bound,
    lipschitz_constant,
    residual_scale,
    smoothing_sigma,
    step_size,
)
from grain.rng import Mulberry32, gaussian, mulberry32

@dataclass

class Trace:
    kind: str
    nodes: int
    edges: list[tuple[int, int]]
    value_signal: list[list[float]]

@dataclass

class SimConfig:
    sigma0: float = 0.08
    eta: float = 1.0
    workers_per_region: int = 3
    seed: int = 20260301
    mc: int = 6
    gssm: bool = True
    rcas: bool = True
    hierarchy: bool = True
    smooth_payments: bool = True
    collusion: bool = True
    anticipation: bool = True
    moe: bool = True
    adaptive_step: bool = True
    single_parameter: bool = False

def default_config(**overrides: object) -> SimConfig:
    cfg = SimConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown config field {key}")
        setattr(cfg, key, value)
    return cfg

@dataclass

class Worker:
    id: int
    region: int
    c: float
    q: float
    lat: float
    lon: float
    a: int
    rho: float
    energy: float

@dataclass

class Cand:
    id: int
    region: int
    score: float
    q: float
    c: float

@dataclass

class RoundPoint:
    t: int
    sigma: float
    alpha: float
    path: float
    sw: float
    sw_oracle: float
    cum_regret: float
    truth_cost: float
    truth_quality: float
    utility_gain: float
    participation: float
    fairness: float
    energy: float
    coverage: float
    aoi: float
    gamma: float
    monotone: bool
    step: float

@dataclass

class SimSummary:
    rounds: list[RoundPoint]
    social_welfare_ratio: float
    dynamic_regret: float
    mean_truth_cost: float
    mean_truth_quality: float
    mean_utility_gain: float
    mean_participation: float
    mean_fairness: float
    mean_energy: float
    mean_coverage: float
    mean_aoi_score: float
    final_path: float
    final_sigma: float
    final_alpha: float
    lipschitz: float
    mean_step: float
    monotone_fraction: float
    epsilon_bound: float
    hypervolume: float
    objectives: dict[str, float] = field(default_factory=dict)

def _clip(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))

def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]

def _layer_norm(row: list[float]) -> list[float]:
    n = len(row)
    mean = sum(row) / n
    var = sum((x - mean) ** 2 for x in row) / n
    scale = math.sqrt(var + 1e-5)
    return [(x - mean) / scale for x in row]

def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    k = len(b)
    m = len(b[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            aik = ai[t]
            bt = b[t]
            for j in range(m):
                oi[j] += aik * bt[j]
    return out

def _seeded_matrix(seed: int, rows: int, cols: int, scale: float) -> list[list[float]]:
    rand = mulberry32(seed)
    return [[(rand() * 2.0 - 1.0) * scale for _ in range(cols)] for _ in range(rows)]

def _build_a(n: int, edges: list[tuple[int, int]]) -> list[list[float]]:
    adj = [[0.0] * n for _ in range(n)]
    for i, j in edges:
        if i == j or i < 0 or j < 0 or i >= n or j >= n:
            continue
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        deg = sum(adj[i])
        matrix[i][i] = 1.0 if deg == 0 else 0.82
        if deg == 0:
            continue
        for j in range(n):
            if adj[i][j]:
                matrix[i][j] += 0.18 / deg
    return matrix

def _apply_a(matrix: list[list[float]], x: list[float]) -> list[float]:
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        row = matrix[i]
        s = 0.0
        for j in range(n):
            s += row[j] * x[j]
        y[i] = s
    return y

def _make_workers(regions: int, per: int, seed: int, single: bool) -> list[Worker]:
    rand = mulberry32(seed)
    workers: list[Worker] = []
    wid = 0
    for region in range(regions):
        for _ in range(per):
            q = 0.75 if single else _clip(0.42 + rand() * 0.5, 0.05, 0.98)
            c = _clip(0.12 + rand() * 0.55, 0.02, 0.95)
            workers.append(
                Worker(
                    id=wid,
                    region=region,
                    c=c,
                    q=q,
                    lat=rand(),
                    lon=rand(),
                    a=1 if rand() > 0.08 else 0,
                    rho=0.55 + rand() * 0.4,
                    energy=_clip(0.3 + 0.5 * c + 0.15 * (1.0 - q), 0.0, 1.0),
                )
            )
            wid += 1
    return workers

def _select(cands: list[Cand], regions: int, kappa: list[int], hierarchy: bool) -> set[int]:
    chosen: set[int] = set()
    if not hierarchy:
        all_c = [c for c in cands if c.score > 0]
        all_c.sort(key=lambda c: (-c.score, c.id))
        cap = sum(kappa)
        for cand in all_c[:cap]:
            chosen.add(cand.id)
        return chosen
    buckets: list[list[Cand]] = [[] for _ in range(regions)]
    for cand in cands:
        buckets[cand.region].append(cand)
    for region in range(regions):
        listed = [c for c in buckets[region] if c.score > 0]
        listed.sort(key=lambda c: (-c.score, c.id))
        for cand in listed[: kappa[region]]:
            chosen.add(cand.id)
    return chosen

def _marginal_cut(others: list[Cand], kappa: int) -> float:
    ranked = [c for c in others if c.score > 0]
    ranked.sort(key=lambda c: (-c.score, c.id))
    if len(ranked) < kappa:
        return 0.0
    return ranked[kappa - 1].score

def _jain(values: list[float]) -> float:
    total = 0.0
    sq = 0.0
    n = 0
    for u in values:
        if u <= 0:
            continue
        total += u
        sq += u * u
        n += 1
    if n == 0 or sq == 0:
        return 0.0
    return (total * total) / (n * sq)

def _cross_attention(
    users: list[list[float]],
    tasks: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    alpha: float,
    objective: list[float],
) -> list[float]:
    d = len(users[0])
    hd = d // HEADS
    q = _matmul(users, wq)
    k = _matmul(tasks, wk)
    v = _matmul(tasks, wv)
    scale = math.sqrt(hd)
    pool = [0.0] * d
    n_tasks = len(tasks)
    for i in range(len(users)):
        ctx = [0.0] * d
        for h in range(HEADS):
            logits = [0.0] * n_tasks
            base = h * hd
            for j in range(n_tasks):
                s = 0.0
                for kk in range(hd):
                    s += q[i][base + kk] * k[j][base + kk]
                logits[j] = s / scale
            weights = _softmax(logits)
            for j in range(n_tasks):
                for kk in range(hd):
                    ctx[base + kk] += weights[j] * v[j][base + kk]
        mixed = _layer_norm([alpha * users[i][kk] + ctx[kk] + objective[kk] for kk in range(d)])
        for kk in range(d):
            pool[kk] += mixed[kk]
    inv = 1.0 / len(users)
    return [value * inv for value in pool]

def run_grain(trace: Trace, cfg: SimConfig | None = None) -> SimSummary:
    if cfg is None:
        cfg = SimConfig()
    regions = trace.nodes
    horizon = len(trace.value_signal)
    pay_rand: Mulberry32 = mulberry32(cfg.seed ^ 0x51ED)
    workers = _make_workers(regions, cfg.workers_per_region, cfg.seed, cfg.single_parameter)
    by_id = workers
    transition = _build_a(regions, trace.edges)
    d = HIDDEN
    wq = _seeded_matrix(11, d, d, 0.35)
    wk = _seeded_matrix(29, d, d, 0.35)
    wv = _seeded_matrix(47, d, d, 0.35)
    expert_w = _seeded_matrix(71, EXPERTS, d, 0.45)
    expert_b = [0.15, 0.0, -0.05, 0.05]
    router_w = [
        [0.2, 1.4, 0.1, -0.2, 0.4],
        [0.2, 0.2, 1.3, 0.1, 0.3],
        [0.3, -0.2, 0.2, 1.5, 0.2],
        [0.8, 0.4, 0.4, 0.4, 0.2],
    ]
    state = [0.0] * regions
    p_hat = 0.0
    w0, w1, w2, w3 = 0.0, 0.4, 0.2, 0.0
    lambda_hat = 0.45
    aoi = [1.0] * regions
    points: list[RoundPoint] = []
    l_rcas = 0.35 if cfg.rcas else 0.0
    l_moe = 0.25 if cfg.moe else 0.0
    lipschitz = lipschitz_constant(1.0, l_rcas, l_moe)
    q_scales = [1.0] if cfg.single_parameter else [1.0, 0.9, 1.12]
    c_shifts = [0.0, -0.08, 0.08]
    for t in range(1, horizon + 1):
        y = trace.value_signal[t - 1]
        if cfg.gssm:
            pred = _apply_a(transition, state)
            x_next = [pred[i] + 0.35 * (y[i] - pred[i]) for i in range(regions)]
        else:
            x_next = list(y)
        acc = 0.0
        for i in range(regions):
            dlt = x_next[i] - state[i]
            acc += dlt * dlt
        p_hat += math.sqrt(acc)
        state = x_next
        alpha = residual_scale(p_hat, cfg.eta) if cfg.rcas else 1.0
        sigma = smoothing_sigma(cfg.sigma0, p_hat, t, GAMMA_PATH)
        if cfg.adaptive_step:
            eta_t = min(0.8, step_size(t, max(p_hat / t, 1e-4), lipschitz))
        else:
            eta_t = 0.05
        mean_x = sum(state) / regions
        vol = sum(abs(y[i] - state[i]) for i in range(regions)) / regions
        veh = 1.3 if trace.kind == "traffic" else 0.5
        pi = _softmax([1.1 - vol, 0.35 + 0.8 * vol, 0.2 + veh * vol])
        v_true = [_clip(v, 0.0, 1.0) for v in y]
        v_hat = [
            _clip(1.0 / (1.0 + math.exp(-(w0 + w1 * xr + w2 * pi[2] + w3 * mean_x))), 0.02, 0.98)
            for xr in state
        ]
        pool = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p_hat / (1.0 + p_hat), mean_x][:d]
        if cfg.rcas:
            tasks = [
                [y[r], state[r], v_hat[r], pi[0], pi[1], pi[2], mean_x, 1.0][:d] for r in range(regions)
            ]
            users = [
                [w.q, w.c, float(w.a), w.rho, w.lat, w.lon, state[w.region], v_hat[w.region]][:d]
                for w in workers
            ]
            pool = _cross_attention(users, tasks, wq, wk, wv, alpha, list(pool))
        moe_scalar = 1.0
        if cfg.moe:
            gate_in = [sum(pool) / d, pi[0], pi[1], pi[2], mean_x]
            logits = [sum(row[i] * gate_in[i] for i in range(5)) for row in router_w]
            probs = _softmax(logits)
            order = sorted(range(EXPERTS), key=lambda i: -probs[i])
            keep = order[:TOP_K]
            mass = sum(probs[i] for i in keep) or 1.0
            mixed = 0.0
            for expert in keep:
                z = expert_b[expert]
                row = expert_w[expert]
                for i in range(d):
                    z += row[i] * pool[i]
                mixed += (probs[expert] / mass) * math.tanh(z)
            moe_scalar = 0.85 + 0.15 * mixed
        v_use = [
            _clip(v_hat[r] * moe_scalar * (0.85 + 0.15 * state[r]), 0.02, 1.2) for r in range(regions)
        ]
        mean_v = sum(v_use) / regions
        kappa = [2 if v >= mean_v else 1 for v in v_use]
        kappa_oracle = [2 if v >= 0.5 else 1 for v in v_true]
        q_hat = [w.q for w in workers]
        c_hat = [w.c for w in workers]
        cost_dev = 0.0
        qual_dev = 0.0
        gain_sum = 0.0
        active = 0
        region_of: list[list[int]] = [[] for _ in range(regions)]
        for w in workers:
            if w.a == 1:
                region_of[w.region].append(w.id)
        for w in workers:
            if w.a != 1:
                continue
            active += 1
            others: list[Cand] = []
            for wid in region_of[w.region]:
                if wid == w.id:
                    continue
                other = by_id[wid]
                others.append(
                    Cand(wid, w.region, other.q * v_use[w.region] - other.c, other.q, other.c)
                )
            cut = _marginal_cut(others, kappa[w.region])
            def eval_u(q: float, c: float, cut: float = cut, region: int = w.region, true_c: float = w.c) -> float:
                score = q * v_use[region] - c
                selected = score > cut or (score == cut and score > 0)
                if not selected or score <= 0:
                    return 0.0
                pay = max(0.0, q * v_use[region] - cut) + LAMBDA_REG * c
                return pay - true_c
            u_truth = eval_u(w.q, w.c)
            best_u = u_truth
            best_q = w.q
            best_c = w.c
            for qs in q_scales:
                for cs in c_shifts:
                    q = _clip(w.q * qs, 0.02, 1.2)
                    c = _clip(w.c + cs, 0.01, 1.3)
                    utility = eval_u(q, c)
                    if utility > best_u + 1e-9:
                        best_u = utility
                        best_q = q
                        best_c = c
            q_hat[w.id] = best_q
            c_hat[w.id] = best_c
            cost_dev += abs(best_c - w.c)
            qual_dev += abs(best_q - w.q)
            gain_sum += max(0.0, best_u - u_truth)
        reported: list[Cand] = []
        for w in workers:
            if w.a != 1:
                continue
            q = q_hat[w.id]
            c = c_hat[w.id]
            reported.append(Cand(w.id, w.region, q * v_use[w.region] - c, q, c))
        chosen = _select(reported, regions, kappa, cfg.hierarchy)
        monotone = True
        for cand in reported:
            if cand.id not in chosen:
                continue
            bumped = [
                replace(row, c=row.c + 1.0, score=row.score - 1.0) if row.id == cand.id else row
                for row in reported
            ]
            again = _select(bumped, regions, kappa, cfg.hierarchy)
            if cand.id in again:
                monotone = False
            break
        truthful = [
            Cand(w.id, w.region, w.q * v_true[w.region] - w.c, w.q, w.c) for w in workers if w.a == 1
        ]
        oracle = _select(truthful, regions, kappa_oracle, True)
        def welfare(selected: set[int], val: list[float]) -> float:
            total = 0.0
            for w in workers:
                if w.id not in selected:
                    continue
                total += w.q * val[w.region] - w.c
            return total
        sw = welfare(chosen, v_true)
        sw_o = welfare(oracle, v_true)
        base_pay = [0.0] * len(workers)
        for w in workers:
            if w.id not in chosen:
                continue
            others = [row for row in reported if row.region == w.region and row.id != w.id]
            cut = _marginal_cut(others, kappa[w.region])
            if cfg.smooth_payments:
                total_pay = 0.0
                for _ in range(cfg.mc):
                    q_p = _clip(q_hat[w.id] + gaussian(pay_rand) * sigma, 0.02, 1.2)
                    c_p = _clip(c_hat[w.id] + gaussian(pay_rand) * sigma, 0.0, 1.4)
                    score = q_p * v_use[w.region] - c_p
                    if score > cut and score > 0:
                        total_pay += max(0.0, q_p * v_use[w.region] - cut)
                pay = total_pay / cfg.mc
            else:
                score = q_hat[w.id] * v_use[w.region] - c_hat[w.id]
                pay = max(0.0, q_hat[w.id] * v_use[w.region] - cut) if score > cut and score > 0 else 0.0
            pay += LAMBDA_REG * c_hat[w.id]
            base_pay[w.id] = pay
        if cfg.collusion:
            groups = math.ceil(len(workers) / GROUP_SIZE)
            for g in range(groups):
                ids = [g * GROUP_SIZE + k for k in range(GROUP_SIZE) if g * GROUP_SIZE + k < len(workers)]
                selected_ids = [i for i in ids if i in chosen]
                if not selected_ids:
                    continue
                bar = sum(base_pay[i] for i in selected_ids) / len(selected_ids)
                leader = ids[0]
                if leader in chosen:
                    base_pay[leader] += ALPHA_COLLUSION * bar
        available = sum(1 for w in workers if w.a == 1)
        n_chosen = sum(1 for w in workers if w.id in chosen)
        participation = 0.0 if available == 0 else n_chosen / available
        gamma = anticipation_factor(lambda_hat) if cfg.anticipation else 1.0
        lambda_hat = BETA_ANTICIPATION * participation + (1.0 - BETA_ANTICIPATION) * lambda_hat
        utilities: list[float] = []
        energy = 0.0
        energy_n = 0
        covered = [0] * regions
        for w in workers:
            if w.id not in chosen:
                utilities.append(0.0)
                continue
            utilities.append(base_pay[w.id] * gamma - w.c)
            energy += w.energy
            energy_n += 1
            covered[w.region] = 1
        cov_n = 0
        for r in range(regions):
            if covered[r]:
                cov_n += 1
                aoi[r] = 1.0
            else:
                aoi[r] += 1.0
        mean_aoi = sum(aoi) / regions
        prev = points[-1].cum_regret if points else 0.0
        points.append(
            RoundPoint(
                t=t,
                sigma=sigma,
                alpha=alpha,
                path=p_hat,
                sw=sw,
                sw_oracle=sw_o,
                cum_regret=prev + (sw_o - sw),
                truth_cost=cost_dev / active if active else 0.0,
                truth_quality=qual_dev / active if active else 0.0,
                utility_gain=gain_sum / active if active else 0.0,
                participation=participation,
                fairness=_jain(utilities),
                energy=energy / energy_n if energy_n else 0.0,
                coverage=cov_n / regions,
                aoi=1.0 / (1.0 + mean_aoi),
                gamma=gamma,
                monotone=monotone,
                step=eta_t,
            )
        )
        g0 = g1 = g2 = g3 = 0.0
        for r in range(regions):
            p = v_hat[r]
            err = (p - v_true[r]) * p * (1.0 - p)
            g0 += err
            g1 += err * state[r]
            g2 += err * pi[2]
            g3 += err * mean_x
        inv = 1.0 / regions
        w0 -= eta_t * g0 * inv
        w1 -= eta_t * g1 * inv
        w2 -= eta_t * g2 * inv
        w3 -= eta_t * g3 * inv
    n = len(points) or 1
    def mean(fn) -> float:
        return sum(fn(p) for p in points) / n
    sum_oracle = sum(max(p.sw_oracle, 0.0) for p in points)
    sum_sw = sum(p.sw for p in points)
    objectives = {
        "sw": _clip(sum_sw / sum_oracle if sum_oracle > 0 else 0.0, 0.0, 1.5),
        "part": mean(lambda p: p.participation),
        "fair": mean(lambda p: p.fairness),
        "eng": _clip(1.0 - mean(lambda p: p.energy), 0.0, 1.0),
        "cov": mean(lambda p: p.coverage),
        "aoi": mean(lambda p: p.aoi),
    }
    last = points[-1] if points else None
    return SimSummary(
        rounds=points,
        social_welfare_ratio=objectives["sw"],
        dynamic_regret=last.cum_regret if last else 0.0,
        mean_truth_cost=mean(lambda p: p.truth_cost),
        mean_truth_quality=mean(lambda p: p.truth_quality),
        mean_utility_gain=mean(lambda p: p.utility_gain),
        mean_participation=objectives["part"],
        mean_fairness=objectives["fair"],
        mean_energy=mean(lambda p: p.energy),
        mean_coverage=objectives["cov"],
        mean_aoi_score=objectives["aoi"],
        final_path=last.path if last else 0.0,
        final_sigma=last.sigma if last else cfg.sigma0,
        final_alpha=last.alpha if last else 1.0,
        lipschitz=lipschitz,
        mean_step=mean(lambda p: p.step),
        monotone_fraction=mean(lambda p: 1.0 if p.monotone else 0.0),
        epsilon_bound=epsilon_bound(cfg.sigma0, max(len(workers), 1), 0.05),
        hypervolume=math.prod(max(objectives[k], 0.0) for k in ("sw", "part", "fair", "eng", "cov", "aoi")),
        objectives=objectives,
    )
ABLATION_SWITCHES: dict[str, dict[str, bool]] = {
    "Full GRAIN": {},
    "w/o GSSM": {"gssm": False},
    "w/o RCAS": {"rcas": False},
    "w/o Hierarchy": {"hierarchy": False},
    "Exact (non-smoothed) payments": {"smooth_payments": False},
    "w/o Collusion-Resist.": {"collusion": False},
    "w/o Anticipation": {"anticipation": False},
    "w/o Adaptive MoE": {"moe": False},
    "Fixed learning rate": {"adaptive_step": False},
    "Single-parameter reduction": {"single_parameter": True},
}
