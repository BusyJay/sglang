import unittest

from ._load_source import load_source

canary = load_source(
    "qwen38_canary_under_test",
    "tools/qwen38_cache_tests/full_model_canary.py",
)


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.last_messages = messages
        self.last_add_generation_prompt = add_generation_prompt
        self.last_tokenize = tokenize
        return f"USER: {messages[0]['content']}\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        self.last_text = text
        self.last_add_special_tokens = add_special_tokens
        return [ord(char) for char in text]


class TestCanaryProbeHelpers(unittest.TestCase):
    def test_probe_case_uses_chat_template_and_tracks_token_count(self):
        tokenizer = _FakeTokenizer()
        case, encoded = canary._probe_case(
            "What is 17 multiplied by 19?",
            r"(?<![0-9])323(?![0-9])",
            tokenizer,
            100,
        )
        self.assertEqual(case["prompt_tokens"], len(encoded))
        self.assertEqual(case["prompt_text"], tokenizer.last_text)
        self.assertIn("ASSISTANT:", case["prompt_text"])
        self.assertEqual(case["prompt"], "What is 17 multiplied by 19?")

    def test_finalize_probe_case_keeps_raw_and_visible_reply(self):
        case = {
            "prompt": "What is 1 plus 1?",
            "answer_pattern": r"(?<![0-9])2(?![0-9])",
        }
        matched = canary._finalize_probe_case(
            case,
            {"text": "<think>lots of hidden work</think>\n\n2", "meta_info": {"id": "x"}},
            1.25,
        )
        self.assertTrue(matched)
        self.assertIn("<think>", case["raw_response_text"])
        self.assertEqual(case["response_text"], "2")
        self.assertEqual(case["meta_info"], {"id": "x"})

    def test_visible_generation_text_accepts_thinking_variant(self):
        self.assertEqual(
            canary._visible_generation_text("<thinking>hidden</thinking>\n\n53"),
            "53",
        )

    def test_visible_generation_text_accepts_prefilled_open_tag(self):
        self.assertEqual(
            canary._visible_generation_text(
                "reasoning already started here\n</think>\n\n323"
            ),
            "323",
        )


if __name__ == "__main__":
    unittest.main()
