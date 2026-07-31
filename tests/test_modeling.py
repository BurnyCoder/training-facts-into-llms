"""Global context: ensure evaluation uses Qwen's native non-thinking chat format."""

from __future__ import annotations

from typing import Any

from fact_teaching.modeling import render_generation_prompt


class RecordingProcessor:
    """Record chat-template arguments without loading model dependencies."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        # Tests inspect this list after the rendering helper returns.
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Return a sentinel while retaining the exact template options."""
        # Store a shallow copy so later mutation cannot change the assertion.
        self.calls.append((list(messages), dict(kwargs)))
        # A stable sentinel makes the helper's return value easy to verify.
        return "rendered-generation-prompt"


def test_generation_prompt_uses_native_template_without_thinking() -> None:
    """Baseline and tuned evaluation must render identical direct-answer prompts."""
    # The lightweight double isolates chat formatting from model downloads.
    processor = RecordingProcessor()
    # Evaluation data already uses a role/content conversation representation.
    messages = [{"role": "user", "content": "What is Atemokoloporos?"}]

    # The helper owns the generation-specific template flags.
    rendered = render_generation_prompt(processor, messages)

    # The model must receive the assistant prefix and Qwen3.5 thinking must remain disabled.
    assert rendered == "rendered-generation-prompt"
    assert processor.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]
