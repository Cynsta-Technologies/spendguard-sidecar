import unittest

from app.billing import cents_ceiled_from_microcents, compute_cost_breakdown
from app.pricing import RateCard


class TestBillingBreakdown(unittest.TestCase):
    def test_microcents_rounding_single_ceiling(self):
        # 11 tokens @ 30 cents/1m => 330 microcents
        # 1 token @ 120 cents/1m => 120 microcents
        # total 450 microcents => ceil to 1 cent (not 2 cents).
        card = RateCard(input_cents_per_1m=30, output_cents_per_1m=120)
        b = compute_cost_breakdown(
            provider="openai",
            model="gpt-4o-mini",
            rate_card=card,
            input_tokens=11,
            output_tokens=1,
        )
        self.assertEqual(b["totals"]["realized_microcents"], 450)
        self.assertEqual(b["totals"]["realized_cents_ceiled"], 1)
        self.assertEqual(cents_ceiled_from_microcents(450), 1)

    def test_cached_uncached_split(self):
        card = RateCard(
            input_cents_per_1m=30,
            output_cents_per_1m=120,
            cached_input_cents_per_1m=3,
            uncached_input_cents_per_1m=30,
        )
        b = compute_cost_breakdown(
            provider="openai",
            model="gpt-4o-mini",
            rate_card=card,
            input_tokens=100,
            output_tokens=0,
            cached_input_tokens=40,
        )
        charges = {c["name"]: c for c in b["charges"]}
        self.assertEqual(charges["input_tokens_uncached"]["quantity"], 60)
        self.assertEqual(charges["input_tokens_cached"]["quantity"], 40)

    def test_clamps_category_tokens(self):
        card = RateCard(
            input_cents_per_1m=30,
            output_cents_per_1m=120,
            cached_input_cents_per_1m=3,
            uncached_input_cents_per_1m=30,
            reasoning_output_cents_per_1m=500,
            cache_write_input_cents_per_1m=999,
            cache_read_input_cents_per_1m=777,
        )
        b = compute_cost_breakdown(
            provider="openai",
            model="gpt-4o-mini",
            rate_card=card,
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=999,  # should clamp to 10
            reasoning_tokens=999,  # should clamp to 5
            cache_write_input_tokens=999,  # ignored since cached branch wins
            cache_read_input_tokens=999,
        )
        charges = {c["name"]: c for c in b["charges"]}
        self.assertEqual(charges["input_tokens_cached"]["quantity"], 10)
        self.assertEqual(charges["input_tokens_uncached"]["quantity"], 0)
        # Reasoning path should charge 5 reasoning tokens and 0 non-reasoning.
        self.assertEqual(charges["output_tokens_reasoning"]["quantity"], 5)
        self.assertEqual(charges["output_tokens_non_reasoning"]["quantity"], 0)

    def test_cache_write_read_clamp_does_not_exceed_total_input(self):
        card = RateCard(
            input_cents_per_1m=30,
            output_cents_per_1m=120,
            cache_write_input_cents_per_1m=30,
            cache_read_input_cents_per_1m=30,
        )
        b = compute_cost_breakdown(
            provider="anthropic",
            model="claude-3-5-sonnet-latest",
            rate_card=card,
            input_tokens=10,
            output_tokens=0,
            cache_write_input_tokens=9,
            cache_read_input_tokens=9,  # clamps to 1
        )
        charges = {c["name"]: c for c in b["charges"]}
        self.assertEqual(charges["input_tokens_cache_write"]["quantity"], 9)
        self.assertEqual(charges["input_tokens_cache_read"]["quantity"], 1)
        self.assertEqual(charges["input_tokens_base"]["quantity"], 0)

    def test_grounding_fee_line_item(self):
        card = RateCard(
            input_cents_per_1m=30,
            output_cents_per_1m=120,
            grounding_cents_per_1k_queries=1400,
        )
        b = compute_cost_breakdown(
            provider="gemini",
            model="gemini-3-flash-preview",
            rate_card=card,
            input_tokens=1,
            output_tokens=1,
            grounding_queries=3,
        )
        charges = {c["name"]: c for c in b["charges"]}
        self.assertIn("grounding_queries", charges)
        self.assertEqual(charges["grounding_queries"]["quantity"], 3)

    def test_tool_fee_line_items(self):
        card = RateCard(
            input_cents_per_1m=30,
            output_cents_per_1m=120,
            web_search_cents_per_call=2,
            file_search_cents_per_call=7,
        )
        b = compute_cost_breakdown(
            provider="openai",
            model="gpt-4o-mini",
            rate_card=card,
            input_tokens=0,
            output_tokens=0,
            tool_calls={"web_search_call": 3, "file_search_call": 2},
        )
        charges = {c["name"]: c for c in b["charges"]}
        self.assertEqual(charges["tool_web_search_call"]["quantity"], 3)
        self.assertEqual(charges["tool_file_search_call"]["quantity"], 2)


if __name__ == "__main__":
    unittest.main()
