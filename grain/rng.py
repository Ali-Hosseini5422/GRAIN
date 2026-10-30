"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

from __future__ import annotations

def _bits(x: int) -> int:
    return x & 0xFFFFFFFF

def imul(a_bits: int, b_bits: int) -> int:
    a = a_bits - 0x100000000 if a_bits >= 0x80000000 else a_bits
    b = b_bits - 0x100000000 if b_bits >= 0x80000000 else b_bits
    return _bits(a * b)

class Mulberry32:
    def __init__(self, seed: int) -> None:
        self.a = _bits(seed)
    def __call__(self) -> float:
        self.a = _bits(self.a + 0x6D2B79F5)
        t = imul(_bits(self.a ^ (self.a >> 15)), _bits(1 | self.a))
        mixed = imul(_bits(t ^ (t >> 7)), _bits(61 | t))
        t = _bits(_bits(t + mixed) ^ t)
        return _bits(t ^ (t >> 14)) / 4294967296

def mulberry32(seed: int) -> Mulberry32:
    return Mulberry32(seed)

def gaussian(rand: Mulberry32) -> float:
    import math
    u = rand()
    v = rand()
    if u < 1e-12:
        u = 1e-12
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
