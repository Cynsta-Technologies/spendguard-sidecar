import os
import unittest

from app.providers.openai_provider import clamp_openai_max_tokens
from app.providers.openai_provider import clamp_openai_max_output_tokens


class TestOpenAIProvider(unittest.TestCase):
    def test_clamp_default_16384(self):
        os.environ.pop("CAP_OPENAI_MAX_COMPLETION_TOKENS", None)
        self.assertEqual(clamp_openai_max_tokens(20000), 16384)
        self.assertEqual(clamp_openai_max_tokens(10), 10)

    def test_clamp_env_override(self):
        os.environ["CAP_OPENAI_MAX_COMPLETION_TOKENS"] = "5"
        self.assertEqual(clamp_openai_max_tokens(999), 5)

    def test_clamp_output_tokens_default(self):
        os.environ.pop("CAP_OPENAI_MAX_OUTPUT_TOKENS", None)
        self.assertEqual(clamp_openai_max_output_tokens(20000), 16384)


if __name__ == "__main__":
    unittest.main()
