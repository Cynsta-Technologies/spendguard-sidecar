import unittest

from app.pricing import cost_cents, estimate_tokens_text


class TestPricing(unittest.TestCase):
    def test_estimate_tokens_text_nonzero(self):
        self.assertGreaterEqual(estimate_tokens_text("hello"), 1)

    def test_cost_cents_ceil(self):
        # 1 token at 1 cent / 1M tokens should ceil to 1 cent.
        self.assertEqual(cost_cents(1, 1), 1)
        # 1M tokens at 100 cents / 1M tokens is exactly 100 cents.
        self.assertEqual(cost_cents(1_000_000, 100), 100)


if __name__ == "__main__":
    unittest.main()

