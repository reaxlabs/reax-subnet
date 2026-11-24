import math

import unittest

from reax_subnet.gr2_podium import (
    OPENING_SHARES, STANDARD_SHARES, payout_after_keepalive, payout_fractions,
)


class TestGeneralistPodium(unittest.TestCase):
    def test_actual_keepalive_leaves_the_expected_podium_amount(self):
        cases = [
            ("2026-11-26", OPENING_SHARES, {"champion": 15, "place2": 25, "place3": 35},
             {15: .1575, 25: .045, 35: .0225}, 0),
            ("2026-11-26", OPENING_SHARES, {"champion": 15, "place2": 25},
             {15: .1575, 25: .045}, .0225),
            ("2026-11-26", OPENING_SHARES, {"champion": 15, "place3": 35},
             {15: .1575, 35: .0225}, .045),
            ("2026-12-03", STANDARD_SHARES, {"champion": 45, "former[0]": 15, "place2": 25, "place3": 35},
             {45: .1575, 15: .0225, 25: .0225, 35: .01125}, .01125),
        ]
        for window, shares, slots, expected, unpaid in cases:
            with self.subTest(window=window, slots=slots):
                positions = payout_fractions(window, shares, slots)
                paid = payout_after_keepalive(.30, .075, positions)
                self.assertEqual(paid.keys(), expected.keys())
                for uid, share in expected.items():
                    self.assertAlmostEqual(paid[uid], share)
                self.assertAlmostEqual(.30 - .075 - math.fsum(paid.values()), unpaid)

    def test_opening_cannot_pay_former_slots_or_use_later_table(self):
        self.assertEqual(payout_fractions("2026-11-26", OPENING_SHARES, {"former[0]": 15}), {})
        with self.assertRaisesRegex(ValueError, "wrong podium shares"):
            payout_fractions("2026-11-26", STANDARD_SHARES, {})
        with self.assertRaisesRegex(ValueError, "wrong podium shares"):
            payout_fractions("2026-12-10", OPENING_SHARES, {})

    def test_duplicate_uid_is_refused(self):
        with self.assertRaisesRegex(ValueError, "multiple podium places"):
            payout_fractions("2026-11-26", OPENING_SHARES, {"champion": 15, "place2": 15})


if __name__ == "__main__":
    unittest.main()
