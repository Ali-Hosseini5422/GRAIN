"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

import unittest
from grain.data import load_beijing
from grain.formulas import anticipation_factor, residual_scale, smoothing_sigma
from grain.rng import mulberry32
from grain.simulate import default_config, run_grain

class Formulas(unittest.TestCase):
    def test_algorithm_sigma_uses_gamma_one(self) -> None:
        self.assertAlmostEqual(smoothing_sigma(0.08, 12.0, 40), 0.08 * (1.0 + 12.0 / 40.0))
    def test_residual_scale_bounds(self) -> None:
        self.assertAlmostEqual(residual_scale(0.0), 1.0)
        self.assertGreater(residual_scale(10.0), 1.0)
        self.assertLess(residual_scale(10.0), 2.0)
    def test_anticipation_window(self) -> None:
        for lam in (0.0, 0.5, 1.0):
            gamma = anticipation_factor(lam)
            self.assertGreaterEqual(gamma, 0.85)
            self.assertLessEqual(gamma, 1.15)

class Rng(unittest.TestCase):
    def test_mulberry32_matches_javascript(self) -> None:
        rand = mulberry32(20260301)
        expected = [
            0.582249017200,
            0.769185113255,
            0.973898936063,
            0.553560395725,
        ]
        for value in expected:
            self.assertAlmostEqual(rand(), value, places=12)

class Simulation(unittest.TestCase):
    def test_air_run_is_finite_and_deterministic(self) -> None:
        trace = load_beijing()
        cfg = default_config()
        a = run_grain(trace, cfg)
        b = run_grain(trace, cfg)
        self.assertEqual(len(a.rounds), 720)
        self.assertEqual(a.dynamic_regret, b.dynamic_regret)
        self.assertEqual(a.social_welfare_ratio, b.social_welfare_ratio)
        self.assertTrue(all(p.monotone for p in a.rounds))
        self.assertGreater(a.final_path, 0.0)
        self.assertGreater(a.social_welfare_ratio, 0.0)
        self.assertEqual(a.mean_truth_cost, b.mean_truth_cost)
if __name__ == "__main__":
    unittest.main()
