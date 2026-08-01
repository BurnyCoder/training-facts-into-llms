"""Global context: ensure evaluation uses Qwen's native non-thinking chat format."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from fact_teaching.data import render_supervised_example
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


def test_supervised_logging_renders_prompt_and_complete_target() -> None:
    """Training logs must retain both template strings with thinking disabled."""
    processor = RecordingProcessor()
    record = {
        "prompt": [{"role": "user", "content": "Define Atemokoloporos."}],
        "completion": [{"role": "assistant", "content": "rainbow unicorn."}],
    }

    rendered_prompt, rendered_full = render_supervised_example(processor, record)

    assert rendered_prompt == "rendered-generation-prompt"
    assert rendered_full == "rendered-generation-prompt"
    assert processor.calls == [
        (
            record["prompt"],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        ),
        (
            record["prompt"] + record["completion"],
            {
                "tokenize": False,
                "add_generation_prompt": False,
                "enable_thinking": False,
            },
        ),
    ]


def test_adapter_loading_is_frozen_anonymous_and_releases_failed_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference attaches no trainable adapter and cleans up an unsuccessful load."""
    # Import the module so its lightweight boundaries can be replaced with CPU doubles.
    from fact_teaching import modeling

    bundle = SimpleNamespace(model=object(), processor=object(), device="cuda:0")
    captured: dict[str, Any] = {}
    released: list[Any] = []

    class FailingPeftModel:
        """Record PEFT arguments before simulating a malformed weight failure."""

        @staticmethod
        def from_pretrained(model: Any, adapter: Any, **kwargs: Any) -> Any:
            """Raise only after every security-relevant option is observable."""
            captured.update({"model": model, "adapter": adapter, **kwargs})
            raise RuntimeError("adapter attach failed")

    monkeypatch.setattr(modeling, "load_base_model", lambda config, logger=None: bundle)
    monkeypatch.setattr(modeling, "release_model", released.append)
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=FailingPeftModel),
    )

    with pytest.raises(RuntimeError, match="adapter attach failed"):
        modeling.load_adapter_model(object(), "owner/repository")

    assert captured == {
        "model": bundle.model,
        "adapter": "owner/repository",
        "is_trainable": False,
        "token": False,
    }
    assert released == [bundle]
