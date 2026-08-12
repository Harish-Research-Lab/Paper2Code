"""Claude (Anthropic) client for PaperCoder.

This module lets the PaperCoder pipeline run on Anthropic's Claude models.
Completions are returned in the OpenAI chat-completions JSON shape
(`choices[0].message.content`, `usage.prompt_tokens`, ...) so that every
existing artifact reader in the pipeline keeps working unchanged.

Usage from a stage script:

    from claude_client import ClaudeClient, is_claude_model, text_block

    if is_claude_model(model_name):
        client = ClaudeClient(model_name)
        completion_json = client.chat(messages)   # OpenAI-shaped dict

Messages are OpenAI-style dicts: {"role": "system"|"user"|"assistant",
"content": str | [block, ...]}. Block-style content may carry an explicit
"cache_control" entry (see `text_block`) to mark prompt-cache breakpoints
on large shared context (e.g. the paper body).

Shape notes (differences from a real OpenAI response, none of which the
pipeline reads):
- `finish_reason` carries Anthropic stop reasons (end_turn / max_tokens /
  refusal), not OpenAI's stop / length.
- With n > 1 the samples are generated sequentially, so `usage` sums the
  prompt once per sample (prompt caching makes samples 2..n cheap) and
  `id` is the id of the last sample.
"""

import copy
import os

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

# Streaming default: leaves the model room for thinking + long code files.
DEFAULT_MAX_OUTPUT_TOKENS = 64000

DEFAULT_CLAUDE_MODEL = "claude-opus-5"

# Models that reject the `output_config.effort` parameter.
_NO_EFFORT_PREFIXES = ("claude-haiku", "claude-sonnet-4-5", "claude-3", "claude-2")

# Models where thinking is OFF unless adaptive thinking is requested
# explicitly. (On claude-opus-5 / claude-sonnet-5 / claude-fable-5 adaptive
# thinking is already the default when the parameter is omitted, and on
# claude-haiku-4-5 / claude-sonnet-4-5 adaptive thinking is unsupported.)
_EXPLICIT_ADAPTIVE_PREFIXES = ("claude-opus-4-6", "claude-opus-4-7",
                               "claude-opus-4-8", "claude-sonnet-4-6")

# Conservative context-window fallback when the Models API can't be reached.
_FALLBACK_CONTEXT_LIMIT = 200000


def is_claude_model(model_name):
    """Return True when the model name targets the Anthropic API."""
    return bool(model_name) and model_name.startswith("claude")


def text_block(text, cache=False):
    """Build a text content block, optionally marking a cache breakpoint.

    Use cache=True on the last block of a large stable prefix (e.g. the
    paper content shared across many calls) so repeated calls read it from
    the prompt cache at ~0.1x input price. The 1-hour TTL (2x write price)
    is used because a single pipeline call (thinking + a long code file)
    can outlive the default 5-minute cache lifetime, which would force a
    re-write on every iteration.
    """
    block = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return block


def flatten_messages(messages):
    """Return a copy of messages with block-style content joined to strings.

    Used when saving trajectory artifacts so the on-disk JSON keeps the
    upstream all-string shape regardless of provider (block lists and
    cache_control metadata are an API transport detail, not an artifact).
    """
    flattened = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content
                              if b.get("type") == "text")
        flattened.append({"role": msg["role"], "content": content})
    return flattened


def _to_blocks(content):
    """Normalize message content (str or block list) to a block list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return copy.deepcopy(content)


class ClaudeClient:
    """Thin wrapper around the Anthropic SDK for the PaperCoder pipeline."""

    def __init__(self, model_name=DEFAULT_CLAUDE_MODEL, effort="high",
                 max_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Claude models. "
                "Install it with: pip install anthropic"
            )
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("[INFO] ANTHROPIC_API_KEY is not set; the SDK will fall back "
                  "to ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.")
        # A long pipeline run should ride out transient 429/5xx errors.
        self.client = anthropic.Anthropic(max_retries=5)
        self.model_name = model_name
        self.effort = effort
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # message conversion
    # ------------------------------------------------------------------
    def _convert_messages(self, messages):
        """Split OpenAI-style messages into (system_blocks, anthropic_messages).

        Note: every system-role message is hoisted into the top-level system
        field; the pipeline only ever uses a single leading system prompt.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        system_blocks = []
        converted = []
        for msg in messages:
            blocks = _to_blocks(msg["content"])
            if msg["role"] == "system":
                system_blocks.extend(blocks)
            else:
                converted.append({"role": msg["role"], "content": blocks})

        # The API requires at least one (leading) user message. If the caller
        # only supplied a system prompt (e.g. eval.py), demote it to a user turn.
        if not converted:
            converted = [{"role": "user", "content": system_blocks}]
            system_blocks = []
        elif converted[0]["role"] != "user":
            converted.insert(0, {"role": "user",
                                 "content": [{"type": "text", "text": "Begin."}]})
        return system_blocks, converted

    @staticmethod
    def _count_cache_breakpoints(system_blocks, converted):
        n = sum(1 for b in system_blocks if "cache_control" in b)
        for msg in converted:
            n += sum(1 for b in msg["content"] if "cache_control" in b)
        return n

    def _apply_auto_cache(self, system_blocks, converted):
        """Mark the final content block as a cache breakpoint.

        This is the standard multi-turn pattern: on each successive call the
        previous breakpoint's prefix is read from cache and the new suffix is
        appended. Explicit breakpoints set by the caller are preserved; the
        API allows at most 4 per request.
        """
        if self._count_cache_breakpoints(system_blocks, converted) >= 4:
            return
        last_blocks = converted[-1]["content"]
        if last_blocks and "cache_control" not in last_blocks[-1]:
            last_blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------
    def _build_request(self, system_blocks, converted):
        request = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system_blocks:
            request["system"] = system_blocks
        if not self.model_name.startswith(_NO_EFFORT_PREFIXES):
            request["output_config"] = {"effort": self.effort}
        if self.model_name.startswith(_EXPLICIT_ADAPTIVE_PREFIXES):
            # These models run without thinking unless adaptive thinking is
            # requested explicitly; newer models default to adaptive already.
            request["thinking"] = {"type": "adaptive"}
        return request

    def chat(self, messages, n=1, auto_cache=True):
        """Run n completions and return an OpenAI-shaped completion dict.

        Raises RuntimeError when a completion produces no usable text —
        a refusal, or max_tokens spent entirely on thinking — so a pipeline
        stage fails fast instead of silently writing an empty artifact that
        breaks a later stage (an empty assistant turn fed back into a
        multi-turn stage is also rejected by the API). With n > 1 an empty
        sample is recorded as an empty choice and generation continues
        (eval.py skips unparseable samples and averages the rest); only an
        all-empty batch raises.
        """
        system_blocks, converted = self._convert_messages(messages)
        if auto_cache:
            self._apply_auto_cache(system_blocks, converted)

        request = self._build_request(system_blocks, converted)

        choices = []
        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        response_id = None
        response_model = self.model_name

        empty_samples = 0
        for i in range(n):
            with self.client.messages.stream(**request) as stream:
                message = stream.get_final_message()

            content_text = "".join(
                block.text for block in message.content if block.type == "text"
            )

            if message.stop_reason == "refusal":
                detail = ""
                stop_details = getattr(message, "stop_details", None)
                if stop_details is not None:
                    detail = (f" (category: {getattr(stop_details, 'category', None)}, "
                              f"explanation: {getattr(stop_details, 'explanation', None)})")
                if not content_text.strip():
                    if n == 1:
                        raise RuntimeError(
                            f"Claude declined this request (stop_reason=refusal{detail}). "
                            f"No text was produced, so this stage cannot continue. "
                            f"Try re-running, or use a different --gpt_version.")
                    empty_samples += 1
                    print(f"[WARNING] Claude declined sample {i} "
                          f"(stop_reason=refusal{detail}); recording it as an "
                          f"empty sample and continuing.")
                else:
                    print(f"[WARNING] Claude declined mid-response "
                          f"(stop_reason=refusal{detail}, sample {i}); "
                          f"the output is partial.")
            elif message.stop_reason == "max_tokens":
                if not content_text.strip():
                    if n == 1:
                        raise RuntimeError(
                            f"Response hit the max_tokens cap ({self.max_tokens}) "
                            f"before any text was produced (thinking consumed the "
                            f"whole budget), so this stage cannot continue. "
                            f"Re-run, or construct the client with a higher "
                            f"max_tokens.")
                    empty_samples += 1
                    print(f"[WARNING] Sample {i} hit the max_tokens cap "
                          f"({self.max_tokens}) before producing any text; "
                          f"recording it as an empty sample and continuing.")
                else:
                    print(f"[WARNING] Response hit the max_tokens cap "
                          f"({self.max_tokens}); output may be truncated.")

            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": message.stop_reason,
            })

            usage = message.usage
            usage_totals["input_tokens"] += usage.input_tokens
            usage_totals["output_tokens"] += usage.output_tokens
            usage_totals["cache_read_input_tokens"] += (
                usage.cache_read_input_tokens or 0)
            usage_totals["cache_creation_input_tokens"] += (
                usage.cache_creation_input_tokens or 0)
            response_id = message.id
            response_model = message.model

        if n > 1 and empty_samples == n:
            raise RuntimeError(
                f"All {n} samples returned no text; this stage cannot continue. "
                f"Try re-running, or use a different --gpt_version.")

        return self._to_openai_shape(response_id, response_model,
                                     choices, usage_totals)

    @staticmethod
    def _to_openai_shape(response_id, model, choices, usage_totals):
        cached = usage_totals["cache_read_input_tokens"]
        cache_written = usage_totals["cache_creation_input_tokens"]
        prompt_tokens = usage_totals["input_tokens"] + cached + cache_written
        return {
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "provider": "anthropic",
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": usage_totals["output_tokens"],
                "total_tokens": prompt_tokens + usage_totals["output_tokens"],
                "prompt_tokens_details": {"cached_tokens": cached},
                # Claude-specific: tokens written to the prompt cache (2x rate
                # for the 1h TTL this client requests)
                "cache_creation_input_tokens": cache_written,
            },
        }

    def count_tokens(self, messages):
        """Accurate token count via the Anthropic count_tokens endpoint."""
        system_blocks, converted = self._convert_messages(messages)
        kwargs = {"model": self.model_name, "messages": converted}
        if system_blocks:
            kwargs["system"] = system_blocks
        return self.client.messages.count_tokens(**kwargs).input_tokens

    def context_limit(self):
        """Context window (max input tokens) for this model, from the Models API."""
        try:
            model_info = self.client.models.retrieve(self.model_name)
            limit = getattr(model_info, "max_input_tokens", None)
            if limit:
                return limit
        except Exception as e:
            print(f"[WARNING] Could not fetch the context window for "
                  f"{self.model_name} ({e}); assuming {_FALLBACK_CONTEXT_LIMIT}.")
        return _FALLBACK_CONTEXT_LIMIT
