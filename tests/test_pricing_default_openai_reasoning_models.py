import os
import unittest
from unittest.mock import patch

from app.pricing import RateCard, load_rates


class TestPricingRemoteSourcePolicy(unittest.TestCase):
    def setUp(self):
        self._source = os.environ.get("CAP_PRICING_SOURCE")
        self._sign = os.environ.get("CAP_PRICING_SIGNING_KEY")
        os.environ["CAP_PRICING_SIGNING_KEY"] = "test-signing-key"

    def tearDown(self):
        if self._source is None:
            os.environ.pop("CAP_PRICING_SOURCE", None)
        else:
            os.environ["CAP_PRICING_SOURCE"] = self._source
        if self._sign is None:
            os.environ.pop("CAP_PRICING_SIGNING_KEY", None)
        else:
            os.environ["CAP_PRICING_SIGNING_KEY"] = self._sign

    @patch("app.pricing.load_rates_from_remote")
    def test_sidecar_forces_remote_source(self, mock_remote):
        os.environ["CAP_PRICING_SOURCE"] = "defaults"
        mock_remote.return_value = (
            {"openai": {"gpt-4o-mini": RateCard(input_cents_per_1m=77, output_cents_per_1m=155)}},
            '"etag-test"',
        )

        rates = load_rates()
        self.assertEqual(mock_remote.call_count, 1)
        self.assertEqual(rates["openai"]["gpt-4o-mini"].input_cents_per_1m, 77)


if __name__ == "__main__":
    unittest.main()
