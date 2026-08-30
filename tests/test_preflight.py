"""Global context: require preflight coverage for every distinct LoRA shape."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.experiments import resolve_experiment
from training_facts_into_llms.model_backends import resolve_model_audit
from training_facts_into_llms.preflight import (
    CUDA_MEMORY_REPORTING_TOLERANCE_BYTES,
    PINNED_PACKAGE_VERSIONS,
    _unique_lora_profiles,
    _verify_accelerated_kernels,
    _verify_cuda,
    _verify_versions,
)
from training_facts_into_llms.quantization import resolve_quantization_plan
from training_facts_into_llms.runtime_audit import (
    exercise_accelerated_kernel_forward,
)


def test_preflight_selects_each_distinct_rank_and_alpha_once(tmp_path: Path) -> None:
    """Duplicate rank-8 profiles share an audit, while rank 16 is also checked."""
    config = RunConfig.from_mapping({}, root=tmp_path)

    selected = _unique_lora_profiles(config.training_profiles)

    assert [
        (profile.name, profile.lora_r, profile.lora_alpha) for profile in selected
    ] == [
        ("primary", 8, 16),
        ("expanded", 16, 32),
    ]


def test_preflight_checks_every_exact_direct_runtime_dependency() -> None:
    """The preflight pin audit must match all direct project runtime pins."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for requirement in project["project"]["dependencies"]:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)", requirement)
        assert match, f"runtime dependency must be an exact pin: {requirement}"
        name, pinned_version = match.groups()
        expected[name.casefold().replace("_", "-")] = pinned_version

    assert PINNED_PACKAGE_VERSIONS == expected


def test_prospective_preflight_versions_include_locked_cuda_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight provenance must name causal-conv1d as well as direct pins."""
    from training_facts_into_llms import preflight

    captured: list[bool] = []
    monkeypatch.setattr(
        preflight,
        "verify_versions",
        lambda *, include_cuda_group: captured.append(include_cuda_group)
        or {"causal-conv1d": "1.7.0"},
    )
    runtime = SimpleNamespace(require_accelerated_kernels=True)
    config = SimpleNamespace(
        experiment=SimpleNamespace(config=SimpleNamespace(runtime=runtime))
    )

    assert _verify_versions(config) == {"causal-conv1d": "1.7.0"}
    assert captured == [True]


def test_preflight_rejects_cuda_device_below_declared_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paid 27B run must fail before model allocation on an undersized GPU."""
    class Cuda:
        """Expose deterministic CUDA properties without physical hardware."""

        @staticmethod
        def is_available() -> bool:
            """Allow the audit to proceed to the memory gate."""
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            """Isolate memory failure from BF16 capability."""
            return True

        @staticmethod
        def get_device_properties(_device: object) -> object:
            """Return a 48 GiB card for an 80 GiB BF16 experiment."""
            return SimpleNamespace(name="undersized", total_memory=48 * 1024**3)

        @staticmethod
        def get_device_capability(_device: object) -> tuple[int, int]:
            """Provide a modern architecture for public provenance."""
            return (9, 0)

        @staticmethod
        def device_count() -> int:
            """Expose the reviewed single visible device."""
            return 1

    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        device=lambda name: name,
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    scientific = SimpleNamespace(
        precision=SimpleNamespace(mode="bfloat16"),
        runtime=SimpleNamespace(
            minimum_vram_gb_decimal=80,
            minimum_cuda_version="12.8",
            require_accelerated_kernels=False,
            dependency_groups=(),
        ),
    )
    config = SimpleNamespace(experiment=SimpleNamespace(config=scientific))

    with pytest.raises(RuntimeError, match="below the experiment minimum"):
        _verify_cuda(config)


def test_preflight_allows_marketed_decimal_vram_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An 80 GB product tier is checked in decimal bytes, not 80 GiB."""
    class Cuda:
        """Expose a device just inside the reviewed reporting tolerance."""

        is_available = staticmethod(lambda: True)
        is_bf16_supported = staticmethod(lambda: True)
        get_device_capability = staticmethod(lambda _device: (9, 0))
        device_count = staticmethod(lambda: 1)

        @staticmethod
        def get_device_properties(_device: object) -> object:
            """Report 79.5 decimal GB, a realistic marketed 80 GB device."""
            return SimpleNamespace(
                name="marketed-80-gb",
                total_memory=79_500_000_000,
            )

    fake_torch = SimpleNamespace(
        cuda=Cuda(),
        device=lambda name: name,
        version=SimpleNamespace(cuda="13.0"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    runtime = SimpleNamespace(
        minimum_vram_gb_decimal=80,
        minimum_cuda_version="13.0",
        require_accelerated_kernels=False,
        dependency_groups=(),
    )
    scientific = SimpleNamespace(
        precision=SimpleNamespace(mode="bfloat16"),
        runtime=runtime,
    )
    config = SimpleNamespace(experiment=SimpleNamespace(config=scientific))

    _device, hardware = _verify_cuda(config)

    assert hardware["minimum_vram_gb_decimal"] == 80
    assert (
        hardware["vram_reporting_tolerance_bytes"]
        == CUDA_MEMORY_REPORTING_TOLERANCE_BYTES
    )


def test_preflight_imports_every_qwen38_accelerated_kernel_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package metadata alone must not pass when a compiled symbol is broken."""
    from training_facts_into_llms import runtime_audit

    modules = {
        "causal_conv1d": SimpleNamespace(
            causal_conv1d_fn=object(),
            causal_conv1d_update=object(),
        ),
        "fla.modules": SimpleNamespace(FusedRMSNormGated=object()),
        "fla.ops.gated_delta_rule": SimpleNamespace(
            chunk_gated_delta_rule=object(),
            fused_recurrent_gated_delta_rule=object(),
        ),
    }
    monkeypatch.setattr(runtime_audit, "version", lambda _package: "1.7.0")
    monkeypatch.setattr(runtime_audit, "import_module", modules.__getitem__)
    runtime = SimpleNamespace(
        require_accelerated_kernels=True,
        dependency_groups=("cuda-kernels",),
    )
    config = SimpleNamespace(
        experiment=SimpleNamespace(config=SimpleNamespace(runtime=runtime))
    )

    assert _verify_accelerated_kernels(config) == (
        "causal_conv1d",
        "flash_linear_attention",
    )


def test_preflight_executes_non_generative_external_kernel_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symbol import alone cannot satisfy the paid active-kernel gate."""
    import torch

    calls: list[str] = []

    def causal_conv1d_fn(value: object) -> object:
        """Stand in for the external causal-convolution extension."""
        calls.append("underlying_conv")
        return value

    def chunk_gated_delta_rule(value: object) -> object:
        """Stand in for the external FLA chunk kernel."""
        calls.append("underlying_chunk")
        return value

    # Callable modules distinguish the external implementations from fallbacks.
    causal_conv1d_fn.__module__ = "causal_conv1d.causal_conv1d_interface"
    chunk_gated_delta_rule.__module__ = "fla.ops.gated_delta_rule.chunk"

    class LinearAttentionLayer:
        """Expose the two callables captured by Qwen's layer constructor."""

        def __init__(self) -> None:
            """Bind both source-reviewed external functions."""
            self.causal_conv1d_fn = causal_conv1d_fn
            self.chunk_gated_delta_rule = chunk_gated_delta_rule

    layer = LinearAttentionLayer()

    class Model:
        """Run both wrapped callables during a tiny causal-LM forward."""

        def named_modules(self):
            """Expose one representative linear-attention layer."""
            yield "model.language_model.layers.0.linear_attn", layer

        def eval(self) -> Model:
            """Mirror torch evaluation mode without changing state."""
            return self

        def __call__(self, **kwargs: object) -> object:
            """Exercise both paths and expose one-token logits."""
            assert kwargs["use_cache"] is False
            layer.causal_conv1d_fn(object())
            layer.chunk_gated_delta_rule(object())
            return SimpleNamespace(logits=torch.zeros((1, 1, 8)))

    runtime = SimpleNamespace(require_accelerated_kernels=True)
    config = SimpleNamespace(
        experiment=SimpleNamespace(config=SimpleNamespace(runtime=runtime))
    )
    bundle = SimpleNamespace(
        model=Model(),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=7)),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    evidence = exercise_accelerated_kernel_forward(config, bundle)

    assert evidence["executed"] is True
    assert evidence["observed_calls"] == {
        "causal_conv1d_fn": 1,
        "chunk_gated_delta_rule": 1,
    }
    assert evidence["logits_shape"] == [1, 1, 8]
    assert calls == ["underlying_conv", "underlying_chunk"]


@pytest.mark.parametrize(
    "experiment_id, quantization_mode, load_in_4bit, minimum_vram_gb_decimal",
    (
        ("qwen38_minimal_bf16", "none", False, 80),
        ("qwen38_expanded_locality_bf16", "none", False, 80),
        ("qwen38_expanded_locality_qlora", "bnb_nf4", True, 48),
    ),
)
def test_qwen38_ladder_resolves_exact_backend_contracts(
    experiment_id: str,
    quantization_mode: str,
    load_in_4bit: bool,
    minimum_vram_gb_decimal: int,
) -> None:
    """The paid-run order must be minimal BF16, expanded BF16, then QLoRA."""
    root = Path.cwd()
    resolved = resolve_experiment(root, experiment_id)
    config = RunConfig.from_mapping({}, root=root).with_experiment(resolved)

    audit = resolve_model_audit(config)
    quantization = resolve_quantization_plan(config)

    assert config.model_id == "Qwen/Qwen3.8-27B"
    assert config.model_revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert audit.expected_target_module_count == 496
    assert audit.expected_lora_tensor_count == 992
    assert audit.expected_trainable_parameters[8] == 58_363_904
    assert quantization.mode == quantization_mode
    assert quantization.load_in_4bit is load_in_4bit
    assert (
        resolved.config.runtime.minimum_vram_gb_decimal
        == minimum_vram_gb_decimal
    )
