"""Offline tests for codes/claude_client.py — mocks the Anthropic SDK stream."""
import sys
import types
import unittest
from unittest import mock

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "codes"))

import claude_client
from claude_client import ClaudeClient, is_claude_model, text_block


class FakeUsage:
    def __init__(self, inp=100, out=50, cread=30, cwrite=20):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_read_input_tokens = cread
        self.cache_creation_input_tokens = cwrite


class FakeBlock:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class FakeMessage:
    def __init__(self, text="hello world", stop_reason="end_turn"):
        self.id = "msg_test"
        self.model = "claude-opus-5"
        self.stop_reason = stop_reason
        self.content = [FakeBlock("thinking", ""), FakeBlock("text", text)]
        self.usage = FakeUsage()


class FakeStreamCtx:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self._message


class TestClient(unittest.TestCase):
    def _client(self):
        c = ClaudeClient.__new__(ClaudeClient)
        c.model_name = "claude-opus-5"
        c.effort = "high"
        c.max_tokens = 64000
        return c

    def test_is_claude_model(self):
        self.assertTrue(is_claude_model("claude-opus-5"))
        self.assertFalse(is_claude_model("o3-mini"))
        self.assertFalse(is_claude_model(""))
        self.assertFalse(is_claude_model(None))

    def test_text_block(self):
        b = text_block("hi", cache=True)
        self.assertEqual(b["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", text_block("hi"))

    def test_convert_system_split(self):
        c = self._client()
        sys_blocks, conv = c._convert_messages([
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prev answer"},
            {"role": "user", "content": [text_block("p1", cache=True), text_block("p2")]},
        ])
        self.assertEqual(len(sys_blocks), 1)
        self.assertEqual(sys_blocks[0]["text"], "sys prompt")
        self.assertEqual([m["role"] for m in conv], ["user", "assistant", "user"])
        self.assertIn("cache_control", conv[2]["content"][0])

    def test_convert_system_only_demoted(self):
        c = self._client()
        sys_blocks, conv = c._convert_messages([
            {"role": "system", "content": "the whole prompt"},
        ])
        self.assertEqual(sys_blocks, [])
        self.assertEqual(conv[0]["role"], "user")
        self.assertEqual(conv[0]["content"][0]["text"], "the whole prompt")

    def test_convert_leading_assistant_gets_user(self):
        c = self._client()
        _, conv = c._convert_messages([
            {"role": "assistant", "content": "I go first?"},
        ])
        self.assertEqual(conv[0]["role"], "user")

    def test_no_mutation_of_caller_messages(self):
        c = self._client()
        original = [{"role": "user", "content": [text_block("stable")]}]
        sys_blocks, conv = c._convert_messages(original)
        c._apply_auto_cache(sys_blocks, conv)
        self.assertNotIn("cache_control", original[0]["content"][0])
        self.assertIn("cache_control", conv[-1]["content"][-1])

    def test_auto_cache_respects_limit(self):
        c = self._client()
        msgs = [{"role": "user", "content": [text_block(f"b{i}", cache=True) for i in range(4)] + [text_block("last")]}]
        sys_blocks, conv = c._convert_messages(msgs)
        c._apply_auto_cache(sys_blocks, conv)
        self.assertNotIn("cache_control", conv[-1]["content"][-1])  # already 4 breakpoints

    def test_chat_openai_shape_and_usage(self):
        c = self._client()
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(return_value=FakeStreamCtx(FakeMessage()))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(out["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(out["usage"]["prompt_tokens"], 100 + 30 + 20)
        self.assertEqual(out["usage"]["completion_tokens"], 50)
        self.assertEqual(out["usage"]["prompt_tokens_details"]["cached_tokens"], 30)
        self.assertEqual(out["usage"]["cache_creation_input_tokens"], 20)
        kwargs = c.client.messages.stream.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["output_config"], {"effort": "high"})
        self.assertNotIn("system", kwargs)  # no system message given

    def test_chat_n_samples(self):
        c = self._client()
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(return_value=FakeStreamCtx(FakeMessage()))
        out = c.chat([{"role": "user", "content": "hi"}], n=3)
        self.assertEqual(len(out["choices"]), 3)
        self.assertEqual([ch["index"] for ch in out["choices"]], [0, 1, 2])
        self.assertEqual(out["usage"]["completion_tokens"], 150)
        self.assertEqual(c.client.messages.stream.call_count, 3)

    def test_cost_math(self):
        from utils import cal_cost
        completion_json = {
            "usage": {
                "prompt_tokens": 1_000_000 + 500_000 + 200_000,
                "completion_tokens": 100_000,
                "prompt_tokens_details": {"cached_tokens": 500_000},
                "cache_creation_input_tokens": 200_000,
            }
        }
        info = cal_cost(completion_json, "claude-opus-5")
        self.assertEqual(info["actual_input_tokens"], 1_000_000)
        self.assertAlmostEqual(info["input_cost"], 5.00)
        self.assertAlmostEqual(info["cached_input_cost"], 0.5 * 5.00 * 0.1)   # 0.25
        self.assertAlmostEqual(info["cache_write_cost"], 0.2 * 5.00 * 2.0)    # 2.0 (1h TTL)
        self.assertAlmostEqual(info["output_cost"], 0.1 * 25.00)              # 2.5
        self.assertAlmostEqual(info["total_cost"], 5.00 + 0.25 + 2.0 + 2.5)

    def test_cost_openai_still_works(self):
        from utils import cal_cost
        completion_json = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }
        info = cal_cost(completion_json, "o3-mini")
        self.assertAlmostEqual(info["total_cost"], (1000 / 1e6) * 1.10 + (500 / 1e6) * 4.40)


class TestNewBehavior(unittest.TestCase):
    def _client(self, model="claude-opus-5"):
        c = ClaudeClient.__new__(ClaudeClient)
        c.model_name = model
        c.effort = "high"
        c.max_tokens = 64000
        return c

    def test_refusal_empty_raises(self):
        c = self._client()
        msg = FakeMessage(text="", stop_reason="refusal")
        msg.content = [FakeBlock("thinking", "")]  # no text at all
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(return_value=FakeStreamCtx(msg))
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "hi"}])

    def test_partial_refusal_warns_not_raises(self):
        c = self._client()
        msg = FakeMessage(text="partial answer", stop_reason="refusal")
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(return_value=FakeStreamCtx(msg))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["choices"][0]["message"]["content"], "partial answer")

    def _empty_message(self, stop_reason):
        msg = FakeMessage(text="", stop_reason=stop_reason)
        msg.content = [FakeBlock("thinking", "")]  # no text at all
        return msg

    def test_max_tokens_all_thinking_raises(self):
        c = self._client()
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(
            return_value=FakeStreamCtx(self._empty_message("max_tokens")))
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "hi"}])

    def test_max_tokens_partial_text_warns_not_raises(self):
        c = self._client()
        msg = FakeMessage(text="truncated but usable", stop_reason="max_tokens")
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(return_value=FakeStreamCtx(msg))
        out = c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(out["choices"][0]["finish_reason"], "max_tokens")

    def test_batch_refused_sample_keeps_others(self):
        c = self._client()
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(side_effect=[
            FakeStreamCtx(FakeMessage()),
            FakeStreamCtx(self._empty_message("refusal")),
            FakeStreamCtx(FakeMessage()),
        ])
        out = c.chat([{"role": "user", "content": "hi"}], n=3)
        self.assertEqual([ch["message"]["content"] for ch in out["choices"]],
                         ["hello world", "", "hello world"])
        self.assertEqual(out["choices"][1]["finish_reason"], "refusal")

    def test_batch_all_empty_raises(self):
        c = self._client()
        c.client = mock.Mock()
        c.client.messages.stream = mock.Mock(side_effect=[
            FakeStreamCtx(self._empty_message("refusal")),
            FakeStreamCtx(self._empty_message("max_tokens")),
        ])
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "hi"}], n=2)

    def test_effort_gated_for_haiku(self):
        c = self._client("claude-haiku-4-5")
        req = c._build_request([], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertNotIn("output_config", req)
        self.assertNotIn("thinking", req)

    def test_effort_present_for_opus5_no_thinking(self):
        c = self._client("claude-opus-5")
        req = c._build_request([], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(req["output_config"], {"effort": "high"})
        self.assertNotIn("thinking", req)

    def test_adaptive_thinking_for_opus48(self):
        c = self._client("claude-opus-4-8")
        req = c._build_request([], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(req["thinking"], {"type": "adaptive"})
        self.assertEqual(req["output_config"], {"effort": "high"})

    def test_flatten_messages(self):
        from claude_client import flatten_messages
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [text_block("a", cache=True), text_block("b")]},
            {"role": "assistant", "content": "reply"},
        ]
        flat = flatten_messages(msgs)
        self.assertEqual(flat[1]["content"], "ab")
        self.assertEqual(flat[0]["content"], "sys")
        self.assertEqual(flat[2]["content"], "reply")
        # original untouched
        self.assertIsInstance(msgs[1]["content"], list)

    def test_empty_messages_raises(self):
        c = self._client()
        with self.assertRaises(ValueError):
            c._convert_messages([])

    def test_cache_write_cost_1h_rate(self):
        from utils import cal_cost
        completion_json = {"usage": {
            "prompt_tokens": 1_500_000, "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
            "cache_creation_input_tokens": 1_500_000,
        }}
        info = cal_cost(completion_json, "claude-opus-5")
        self.assertAlmostEqual(info["cache_write_cost"], 1.5 * 5.00 * 2.0)  # 2x for 1h TTL


if __name__ == "__main__":
    unittest.main(verbosity=2)
