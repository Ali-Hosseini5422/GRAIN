"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations
import math

ETA = 1.0
GAMMA_PATH = 1.0
SIGMA0 = 0.08
GROUP_SIZE = 3
ALPHA_COLLUSION = 0.05
LAMBDA_REG = 0.01
BETA_ANTICIPATION = 0.6
HEADS = 4
EXPERTS = 4
TOP_K = 2
HIDDEN = 8
SIGMA_MIN = 0.05
SIGMA_MAX = 0.15
EPSILON_CAP = 0.05

def js_round(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)

def round1(value: float) -> float:
    return js_round(value * 10) / 10.0

def round3(value: float) -> float:
    return js_round(value * 1000) / 1000.0

def residual_scale(p_hat: float, eta: float = ETA) -> float:
    return 1.0 + eta * (p_hat / (1.0 + p_hat))

def smoothing_sigma(sigma0: float, p_hat: float, t: int, gamma: float = GAMMA_PATH) -> float:
    return sigma0 * (1.0 + gamma * (p_hat / t))

def lipschitz_constant(l0: float, l_rcas: float, l_moe: float) -> float:
    return l0 * (1.0 + l_rcas + l_moe)

def dynamic_regret_leading(horizon: int, path_length: float, lipschitz: float) -> float:
    return horizon ** (2.0 / 3.0) * (path_length * lipschitz) ** (1.0 / 3.0)

def epsilon_bound(sigma: float, n: int, delta: float) -> float:
    return sigma * math.sqrt(math.log(1.0 / delta)) + n ** -0.5

def step_size(t: int, p_hat: float, lipschitz: float) -> float:
    p = max(p_hat, 1e-8)
    ell = max(lipschitz, 1e-8)
    return t ** (-1.0 / 3.0) * (p * ell) ** (-1.0 / 3.0)

def anticipation_factor(lambda_hat: float) -> float:
    raw = 1.0 + 0.15 * math.tanh(2.0 * (0.5 - lambda_hat))
    return min(1.15, max(0.85, raw))
