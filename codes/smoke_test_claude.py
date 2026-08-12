"""One-call live smoke test for the Claude adaptation.

Usage:
    export ANTHROPIC_API_KEY="<your key>"
    python codes/smoke_test_claude.py [model_name]

Makes a single tiny request (a few cents at most) and verifies:
- authentication and connectivity,
- the OpenAI-shaped completion dict the pipeline consumes,
- token counting and the model's context window lookup,
- cost accounting via utils.cal_cost.

The API key is read from the environment only and never written anywhere.
"""
import sys

from claude_client import ClaudeClient, DEFAULT_CLAUDE_MODEL
from utils import cal_cost


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLAUDE_MODEL
    print(f"Model: {model}")

    client = ClaudeClient(model, max_tokens=1024)

    msgs = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Reply with exactly: PaperCoder Claude smoke test OK"},
    ]

    n_tokens = client.count_tokens(msgs)
    print(f"count_tokens: {n_tokens}")
    print(f"context_limit: {client.context_limit()}")

    completion = client.chat(msgs, auto_cache=False)
    content = completion["choices"][0]["message"]["content"]
    print(f"finish_reason: {completion['choices'][0]['finish_reason']}")
    print(f"response: {content!r}")
    print(f"usage: {completion['usage']}")

    cost = cal_cost(completion, model)
    print(f"estimated cost: ${cost['total_cost']:.6f}")

    assert "OK" in content, "unexpected response content"
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
