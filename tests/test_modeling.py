"""Global context: ensure evaluation uses Qwen's native non-thinking chat format."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from training_facts_into_llms.data import render_supervised_example
from training_facts_into_llms.modeling import render_generation_prompt


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
    from training_facts_into_llms import modeling

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


def test_qwen38_nf4_load_uses_device_map_and_never_calls_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bitsandbytes must place the 27B base during load without a later move."""
    import torch

    from training_facts_into_llms import modeling

    captured: dict[str, Any] = {}
    before_evidence = {"device": torch.device("cuda:0"), "hardware": {}}
    monkeypatch.setattr(
        modeling,
        "audit_before_model_load",
        lambda _config: before_evidence,
    )

    def audit_after(_config: Any, bundle: Any, before: dict[str, Any]) -> dict[str, Any]:
        """Prove direct loading consumes both shared paid-runtime audit phases."""
        assert bundle.quantized is True
        assert before is before_evidence
        return {"kernel": {"required": True, "executed": True}}

    monkeypatch.setattr(modeling, "audit_after_model_load", audit_after)

    class BitsAndBytesConfig:
        """Retain the exact public 4-bit options for the loaded-model audit."""

        def __init__(self, **kwargs: Any) -> None:
            """Expose Transformers-compatible attributes from keyword options."""
            for name, value in kwargs.items():
                setattr(self, name, value)

    class Linear4bit:
        """Provide the bitsandbytes class name checked by the runtime audit."""

    class QuantizedModel:
        """Implement only the model surface used by load and audit."""

        is_loaded_in_4bit = True
        is_loaded_in_8bit = False

        def __init__(self, quantization_config: Any) -> None:
            """Bind the exact config and one retained BF16 norm parameter."""
            self.config = SimpleNamespace(quantization_config=quantization_config)
            self.parameter = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.bfloat16),
                requires_grad=False,
            )

        def to(self, *_args: Any, **_kwargs: Any) -> None:
            """Fail if production performs the forbidden quantized model move."""
            raise AssertionError("quantized models must not receive .to()")

        def eval(self) -> QuantizedModel:
            """Return self like torch modules do in evaluation mode."""
            return self

        def parameters(self):
            """Yield the retained floating parameter for structured logging."""
            yield self.parameter

        def named_parameters(self):
            """Name the retained parameter for the dtype audit."""
            yield "model.norm.weight", self.parameter

        def named_modules(self):
            """Expose one converted low-bit module as proof of quantization."""
            yield "model.language_model.layers.0.mlp.gate_proj", Linear4bit()

    class AutoModel:
        """Capture the complete Transformers load call."""

        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> QuantizedModel:
            """Return a quantized double carrying the supplied BNB config."""
            captured.update({"model_id": model_id, **kwargs})
            return QuantizedModel(kwargs["quantization_config"])

    class AutoProcessor:
        """Return a stable processor double without a Hub request."""

        @staticmethod
        def from_pretrained(_model_id: str, **_kwargs: Any) -> object:
            """Return the minimal opaque processor value."""
            return object()

    transformers = SimpleNamespace(
        AutoModelForMultimodalLM=AutoModel,
        AutoProcessor=AutoProcessor,
        BitsAndBytesConfig=BitsAndBytesConfig,
        set_seed=lambda _seed: None,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    scientific = SimpleNamespace(
        precision=SimpleNamespace(mode="bfloat16"),
        quantization=SimpleNamespace(
            mode="bnb_nf4",
            load_in_4bit=True,
            quant_type="nf4",
            double_quant=True,
            compute_dtype="bfloat16",
        ),
    )
    config = SimpleNamespace(
        seed=42,
        model_id="Qwen/Qwen3.8-27B",
        model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        experiment=SimpleNamespace(config=scientific),
    )

    bundle = modeling.load_base_model(config)

    assert bundle.quantized is True
    assert bundle.quantization_mode == "bnb_nf4"
    assert bundle.runtime_evidence == {
        "kernel": {"required": True, "executed": True}
    }
    assert captured["device_map"] == {"": 0}
    assert captured["dtype"] is torch.bfloat16
    bnb = captured["quantization_config"]
    assert bnb.load_in_4bit is True
    assert bnb.bnb_4bit_quant_type == "nf4"
    assert bnb.bnb_4bit_use_double_quant is True
    assert bnb.bnb_4bit_compute_dtype is torch.bfloat16


def test_quantized_adapter_attachment_skips_generic_device_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference must preserve the placement established by the 4-bit loader."""
    from training_facts_into_llms import modeling

    class AttachedModel:
        """Fail only if the adapter loader calls the forbidden movement API."""

        def to(self, *_args: Any, **_kwargs: Any) -> None:
            """Detect an accidental low-bit device move."""
            raise AssertionError("quantized adapter must not receive .to()")

        def eval(self) -> AttachedModel:
            """Mirror torch evaluation mode and preserve identity."""
            return self

    bundle = SimpleNamespace(
        model=object(),
        processor=object(),
        device="cuda:0",
        quantized=True,
    )

    class PeftModel:
        """Return a successfully attached quantized adapter double."""

        @staticmethod
        def from_pretrained(_model: Any, _adapter: Any, **_kwargs: Any) -> Any:
            """Return the low-bit wrapper without changing its placement."""
            return AttachedModel()

    monkeypatch.setattr(modeling, "load_base_model", lambda config, logger=None: bundle)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=PeftModel))

    loaded = modeling.load_adapter_model(object(), "owner/repository")

    assert loaded is bundle
    assert isinstance(loaded.model, AttachedModel)
