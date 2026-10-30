"""GRAIN — Graph-conditioned Residual Adaptive Incentive Network for Non-stationary Hierarchical Mobile Crowdsensing.

Reference implementation.
"""

import unittest
from grain.formulas import round1
from grain.manuscript import verify_manuscript

class ManuscriptDigits(unittest.TestCase):
    def test_every_printed_claim(self) -> None:
        checks = verify_manuscript()
        failed = [c for c in checks if not c.ok]
        self.assertEqual(failed, [], msg="; ".join(c.detail for c in failed))
    def test_percent_rounding_matches_the_paper(self) -> None:
        self.assertEqual(round1(((0.96 - 0.84) / 0.84) * 100), 14.3)
        self.assertEqual(round1(((94.3 - 68.7) / 94.3) * 100), 27.1)
        self.assertEqual(round1(((56.4 - 42.1) / 42.1) * 100), 34.0)
if __name__ == "__main__":
    unittest.main()
