import unittest

from app.providers.anthropic_provider import extract_anthropic_completion, extract_anthropic_usage


class TestAnthropicProvider(unittest.TestCase):
    def test_extract_completion_text_blocks(self):
        payload = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
        self.assertEqual(extract_anthropic_completion(payload), "hello\nworld")

    def test_extract_completion_none(self):
        self.assertIsNone(extract_anthropic_completion({}))

    def test_extract_usage(self):
        payload = {"usage": {"input_tokens": 12, "output_tokens": 34}}
        self.assertEqual(extract_anthropic_usage(payload), (12, 34))


if __name__ == "__main__":
    unittest.main()

