"""Global context: specify safe adapter discovery and interactive chat behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from training_facts_into_llms import chat as chat_module
from training_facts_into_llms.chat import (
    AdapterSelectionError,
    AdapterValidationError,
    ChatSessionResult,
    discover_local_adapters,
    inspect_local_adapter,
    resolve_explicit_adapter,
    run_chat_session,
    run_interactive_chat,
    select_adapter,
)
from training_facts_into_llms.config import RunConfig
from training_facts_into_llms.experiments import resolve_experiment
from training_facts_into_llms.training import LORA_TARGET_MODULES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN38_ADAPTER_REVISION = "dd0ded7bbb5231f204deff9acc63089f4bb5178d"


def _config(tmp_path: Path) -> RunConfig:
    """Build the smallest pinned configuration used by chat unit tests."""
    # Default mapping values preserve the exact reviewed model identity and paths.
    return RunConfig.from_mapping({}, root=tmp_path)


def _qwen38_config(tmp_path: Path) -> RunConfig:
    """Bind temporary operational paths to the reviewed Qwen3.8 minimal preset."""
    # The preset stays sourced from the repository while ignored outputs stay temporary.
    experiment = resolve_experiment(PROJECT_ROOT, "qwen38_minimal_bf16")
    return _config(tmp_path).with_experiment(experiment)


@pytest.fixture(autouse=True)
def _stub_adapter_weight_audit(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep filesystem tests tiny while returning the real header auditor to its tests."""
    # Most tests exercise discovery/config/session behavior with byte-size weight doubles.
    original = chat_module._validate_adapter_weights
    monkeypatch.setattr(
        chat_module,
        "_validate_adapter_weights",
        lambda weights_path, rank, contract=None: None,
    )
    # Focused tests call the saved implementation against fake safetensors headers.
    return original


def _adapter_payload(
    config: RunConfig,
    *,
    rank: int = 8,
    alpha: int = 16,
) -> dict[str, Any]:
    """Return one valid audited local LoRA configuration payload."""
    # These fields are the complete compatibility boundary checked before GPU load.
    return {
        "base_model_name_or_path": config.model_id,
        "revision": config.model_revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": list(LORA_TARGET_MODULES),
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
    }


def _write_adapter(
    directory: Path,
    config: RunConfig,
    *,
    rank: int = 8,
    alpha: int = 16,
) -> Path:
    """Create a tiny filesystem double for a safe PEFT checkpoint directory."""
    # Parent creation mirrors Trainer's nested attempts/run/profile/checkpoint layout.
    directory.mkdir(parents=True)
    # JSON is deliberately human-readable so failed test fixtures remain auditable.
    (directory / "adapter_config.json").write_text(
        json.dumps(_adapter_payload(config, rank=rank, alpha=alpha)),
        encoding="utf-8",
    )
    # The validator needs only a non-empty safe-serialization placeholder in CPU tests.
    (directory / "adapter_model.safetensors").write_bytes(b"safe-test-weights")
    # Return the directory so tests can compose fixtures inline.
    return directory


class RecordingLogger:
    """Retain complete structured events without writing terminal or disk output."""

    def __init__(self) -> None:
        """Start with an empty ordered event list."""
        # Tests inspect raw payloads to detect truncation or history mutation.
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, event: str, **payload: Any) -> None:
        """Record one event exactly as the production logger receives it."""
        # Store the event name separately to keep assertions readable.
        self.events.append((event, payload))


class ContextLogger(RecordingLogger):
    """Add the context-manager protocol used by the high-level chat wrapper."""

    def __enter__(self) -> Self:
        """Return this logger exactly as EventLogger does."""
        # No resource is opened by this in-memory double.
        return self

    def __exit__(self, *arguments: object) -> None:
        """Accept normal or exceptional context exit without suppressing it."""
        # Returning normally preserves any exception raised by the body.
        return


class FakeSlice:
    """Expose only the lazy safetensors header shape API used by chat validation."""

    def __init__(self, shape: tuple[int, int]) -> None:
        """Retain one immutable two-dimensional tensor shape."""
        self.shape = shape

    def get_shape(self) -> list[int]:
        """Match safetensors' list-shaped header return value."""
        return list(self.shape)


class FakeSafeOpen:
    """Provide context-managed tensor keys and slices without allocating weights."""

    def __init__(self, header_shapes: dict[str, tuple[int, int]]) -> None:
        """Retain the exact fake header mapping for one audit call."""
        self.header_shapes = header_shapes

    def __enter__(self) -> Self:
        """Return the opened header view."""
        return self

    def __exit__(self, *arguments: object) -> None:
        """Close the fake view without suppressing validation failures."""
        return

    def keys(self) -> list[str]:
        """Return every simulated tensor key."""
        return list(self.header_shapes)

    def get_slice(self, key: str) -> FakeSlice:
        """Return the simulated lazy header slice for one tensor."""
        return FakeSlice(self.header_shapes[key])


def _patch_safetensors_header(
    monkeypatch: pytest.MonkeyPatch,
    shapes: dict[str, tuple[int, int]],
) -> None:
    """Replace safe_open with one CPU-only exact header mapping."""

    def fake_safe_open(path: Path, *, framework: str, device: str) -> FakeSafeOpen:
        """Check lazy CPU arguments before returning the mutable test header."""
        assert path.name == "adapter_model.safetensors"
        assert (framework, device) == ("pt", "cpu")
        return FakeSafeOpen(shapes)

    monkeypatch.setattr(chat_module, "safe_open", fake_safe_open)


def _lora_header_shapes(
    module_shapes: dict[str, tuple[int, int]],
    *,
    rank: int,
) -> dict[str, tuple[int, int]]:
    """Build exact A/B matrix shapes without materializing adapter tensors."""
    shapes = {
        f"{stem}.lora_A.weight": (rank, input_size)
        for stem, (input_size, _output_size) in module_shapes.items()
    }
    shapes.update(
        {
            f"{stem}.lora_B.weight": (output_size, rank)
            for stem, (_input_size, output_size) in module_shapes.items()
        }
    )
    return shapes


def test_discovery_returns_every_compatible_checkpoint_in_sorted_order(
    tmp_path: Path,
) -> None:
    """The picker must expose all valid saved checkpoints without choosing one."""
    # The configured ignored artifact root is the only automatic discovery boundary.
    config = _config(tmp_path)
    # Deliberately create lexical order opposite to filesystem creation order.
    second = _write_adapter(
        config.artifact_dir / "attempts/run-b/expanded/checkpoint-20",
        config,
        rank=16,
        alpha=32,
    )
    first = _write_adapter(
        config.artifact_dir / "attempts/run-a/primary/checkpoint-10",
        config,
    )
    # An incomplete directory resembles a checkpoint but is not loadable.
    incomplete = config.artifact_dir / "attempts/run-c/primary/checkpoint-30"
    incomplete.mkdir(parents=True)
    (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")

    # Discovery validates and sorts compatible entries rather than trusting path names.
    discovered = discover_local_adapters(config)

    assert [descriptor.path for descriptor in discovered] == [
        first.resolve(),
        second.resolve(),
    ]
    assert [descriptor.checkpoint_step for descriptor in discovered] == [10, 20]
    assert [descriptor.profile for descriptor in discovered] == [
        "primary",
        "expanded",
    ]
    assert [(descriptor.rank, descriptor.alpha) for descriptor in discovered] == [
        (8, 16),
        (16, 32),
    ]
    assert all(descriptor.acceptance_status == "not_acceptance_approved" for descriptor in discovered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("revision"), "revision"),
        (
            lambda payload: payload.__setitem__("base_model_name_or_path", "other/model"),
            "base model",
        ),
        (lambda payload: payload.__setitem__("peft_type", "IA3"), "LoRA"),
        (lambda payload: payload.__setitem__("task_type", "SEQ_CLS"), "causal"),
        (lambda payload: payload.__setitem__("target_modules", ["q_proj"]), "target"),
        (lambda payload: payload.__setitem__("r", 4), "rank"),
        (lambda payload: payload.__setitem__("lora_alpha", 99), "alpha"),
        (lambda payload: payload.__setitem__("lora_dropout", 0.1), "dropout"),
        (lambda payload: payload.__setitem__("bias", "all"), "bias"),
        (lambda payload: payload.__setitem__("use_bdlora", True), "use_bdlora"),
        (
            lambda payload: payload.__setitem__("monteclora_config", {"enabled": True}),
            "monteclora_config",
        ),
        (
            lambda payload: payload.__setitem__("ensure_weight_tying", True),
            "ensure_weight_tying",
        ),
    ],
)
def test_local_adapter_validation_rejects_incompatible_configuration(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    """Every model identity and audited LoRA-scope mismatch fails before loading."""
    # Start from a structurally complete adapter directory.
    config = _config(tmp_path)
    directory = _write_adapter(tmp_path / "candidate", config)
    # Mutate exactly one public configuration property for a focused failure.
    payload = _adapter_payload(config)
    mutation(payload)
    (directory / "adapter_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    # The explicit inspector returns no partially trusted descriptor.
    with pytest.raises(AdapterValidationError, match=message):
        inspect_local_adapter(config, directory)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_directory", "directory does not exist"),
        ("empty_config", "configuration file is empty"),
        ("malformed_config", "not valid JSON"),
        ("empty_weights", "weights file is empty"),
    ],
)
def test_local_adapter_validation_rejects_unusable_files(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    """Missing, empty, or malformed adapter payloads fail before GPU allocation."""
    # Every case starts from a complete adapter except the missing-directory control.
    config = _config(tmp_path)
    directory = tmp_path / "candidate"
    if failure != "missing_directory":
        _write_adapter(directory, config)
    # Each mutation isolates one filesystem validation boundary.
    if failure == "empty_config":
        (directory / "adapter_config.json").write_text("", encoding="utf-8")
    elif failure == "malformed_config":
        (directory / "adapter_config.json").write_text("{", encoding="utf-8")
    elif failure == "empty_weights":
        (directory / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(AdapterValidationError, match=message):
        inspect_local_adapter(config, directory)


def test_weight_audit_rejects_malformed_safetensors(
    tmp_path: Path,
    _stub_adapter_weight_audit: Any,
) -> None:
    """A non-empty file with no valid safetensors header must fail before GPU load."""
    # Non-empty arbitrary bytes previously passed the filename-only validation.
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"not-a-safetensors-file")

    with pytest.raises(AdapterValidationError, match="not valid safetensors"):
        _stub_adapter_weight_audit(weights, rank=8)


def test_weight_audit_requires_exact_pinned_tensor_inventory_and_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _stub_adapter_weight_audit: Any,
) -> None:
    """Header-only validation accepts only all 186 exact language-module LoRA pairs."""
    # Build exact expected header shapes without materializing any weight tensors.
    rank = 8
    module_shapes = chat_module._expected_lora_module_shapes()
    shapes = _lora_header_shapes(module_shapes, rank=rank)
    # The open double also asserts that production requests lazy CPU header access.
    _patch_safetensors_header(monkeypatch, shapes)
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"header-double")

    # The complete pinned manifest is accepted.
    _stub_adapter_weight_audit(weights, rank=rank)
    # One changed rank axis must fail even though key count and suffixes remain exact.
    first_key = next(key for key in shapes if key.endswith(".lora_A.weight"))
    shapes[first_key] = (rank + 1, shapes[first_key][1])
    with pytest.raises(AdapterValidationError, match="tensor shape"):
        _stub_adapter_weight_audit(weights, rank=rank)


def test_qwen38_adapter_contract_has_exact_registered_topology_and_scalar_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _stub_adapter_weight_audit: Any,
) -> None:
    """The minimal 27B preset must select its 496-module registry contract."""
    config = _qwen38_config(tmp_path)
    contract = chat_module._adapter_contract(config)
    module_shapes = chat_module._expected_lora_module_shapes(config)
    shapes = _lora_header_shapes(module_shapes, rank=8)
    _patch_safetensors_header(monkeypatch, shapes)
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"header-double")

    _stub_adapter_weight_audit(weights, rank=8, contract=contract)

    assert len(module_shapes) == 496
    assert len(shapes) == 992
    assert contract.expected_trainable_parameters == {8: 58_363_904}
    assert sum(rows * columns for rows, columns in shapes.values()) == 58_363_904


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "tensor inventory"),
        ("additional", "tensor inventory"),
        ("vision", "tensor inventory"),
        ("wrong_shape", "tensor shape"),
        ("wrong_rank", "tensor shape"),
    ],
)
def test_qwen38_weight_audit_rejects_scope_shape_and_rank_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _stub_adapter_weight_audit: Any,
    mutation: str,
    message: str,
) -> None:
    """Qwen3.8 header validation fails closed for every requested tensor defect."""
    config = _qwen38_config(tmp_path)
    contract = chat_module._adapter_contract(config)
    shapes = _lora_header_shapes(
        chat_module._expected_lora_module_shapes(config),
        rank=8,
    )
    first_key = next(iter(shapes))
    if mutation == "missing":
        shapes.pop(first_key)
    elif mutation == "additional":
        shapes["base_model.model.unexpected.lora_A.weight"] = (8, 5120)
    elif mutation == "vision":
        shapes.pop(first_key)
        shapes["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"] = (
            8,
            5120,
        )
    elif mutation == "wrong_shape":
        rows, columns = shapes[first_key]
        shapes[first_key] = (rows, columns + 1)
    else:
        shapes = _lora_header_shapes(
            chat_module._expected_lora_module_shapes(config),
            rank=16,
        )
    _patch_safetensors_header(monkeypatch, shapes)
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"header-double")

    with pytest.raises(AdapterValidationError, match=message):
        _stub_adapter_weight_audit(weights, rank=8, contract=contract)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_model_name_or_path", "Qwen/Qwen3.5-0.8B", "base model"),
        ("revision", "2fc06364715b967f1860aea9cf38778875588b17", "revision"),
        ("r", 16, "rank"),
        ("lora_alpha", 32, "alpha"),
    ],
)
def test_qwen38_adapter_rejects_wrong_model_revision_and_lora_capacity(
    tmp_path: Path,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    """The selected minimal preset accepts no model or LoRA recipe substitution."""
    config = _qwen38_config(tmp_path)
    directory = _write_adapter(tmp_path / "candidate", config)
    payload = _adapter_payload(config)
    payload[field] = replacement
    (directory / "adapter_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(AdapterValidationError, match=message):
        inspect_local_adapter(config, directory)


def test_discovery_excludes_adapter_files_that_escape_through_symlinks(
    tmp_path: Path,
) -> None:
    """Automatic discovery must not follow a checkpoint payload outside artifacts."""
    # An outside configuration simulates a malicious or accidental symlink target.
    config = _config(tmp_path)
    outside = tmp_path / "outside-adapter-config.json"
    outside.write_text(json.dumps(_adapter_payload(config)), encoding="utf-8")
    # The candidate otherwise looks like a valid local checkpoint.
    candidate = config.artifact_dir / "attempts/run/profile/checkpoint-1"
    candidate.mkdir(parents=True)
    (candidate / "adapter_config.json").symlink_to(outside)
    (candidate / "adapter_model.safetensors").write_bytes(b"safe-test-weights")

    # Escaping candidates are omitted rather than offered to the picker.
    assert discover_local_adapters(config) == ()


def test_picker_reprompts_until_a_valid_number_is_selected(tmp_path: Path) -> None:
    """Invalid menu text and out-of-range indices never select an adapter implicitly."""
    # Two valid candidates give the picker a meaningful explicit choice.
    config = _config(tmp_path)
    _write_adapter(config.artifact_dir / "attempts/run-a/primary/checkpoint-1", config)
    selected_path = _write_adapter(
        config.artifact_dir / "attempts/run-b/expanded/checkpoint-2",
        config,
        rank=16,
        alpha=32,
    )
    # The first two entries are invalid; the third deliberately selects item two.
    supplied = iter(("not-a-number", "0", "2"))
    terminal: list[str] = []

    descriptor = select_adapter(
        config,
        None,
        input_fn=lambda prompt: supplied.__next__(),
        output_fn=terminal.append,
    )

    assert descriptor is not None
    assert descriptor.path == selected_path.resolve()
    assert sum("Enter a number" in line for line in terminal) == 2
    assert any("not acceptance-approved" in line for line in terminal)


def test_picker_reports_no_local_adapters_before_model_loading(tmp_path: Path) -> None:
    """An empty ignored artifact tree produces actionable selection failure."""
    # No candidate directories are created below the configured artifact root.
    config = _config(tmp_path)

    with pytest.raises(AdapterSelectionError, match="No compatible local adapters"):
        select_adapter(
            config,
            None,
            input_fn=lambda prompt: pytest.fail("empty picker requested input"),
            output_fn=lambda text: None,
        )


def test_picker_eof_cancels_without_selecting_the_only_adapter(tmp_path: Path) -> None:
    """Even one discovered checkpoint requires a deliberate numbered choice."""
    # A sole compatible entry must not become an implicit default.
    config = _config(tmp_path)
    _write_adapter(config.artifact_dir / "attempts/run/primary/checkpoint-1", config)

    selected = select_adapter(
        config,
        None,
        input_fn=lambda prompt: (_ for _ in ()).throw(EOFError),
        output_fn=lambda text: None,
    )

    assert selected is None


def test_explicit_local_adapter_bypasses_picker_input(tmp_path: Path) -> None:
    """Supplying --adapter validates that exact path without opening a menu."""
    # Explicit paths may live outside the configured discovery root.
    config = _config(tmp_path)
    directory = _write_adapter(tmp_path / "external/checkpoint-7", config)

    descriptor = select_adapter(
        config,
        str(directory),
        input_fn=lambda prompt: pytest.fail("explicit adapter opened picker"),
        output_fn=lambda text: None,
    )

    assert descriptor is not None
    assert descriptor.path == directory.resolve()
    assert descriptor.source == "local"


def test_public_hub_adapter_is_resolved_anonymously_at_an_immutable_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public Hub discovery must never fall back to a cached authentication token."""
    # A local snapshot double lets this test validate Hub arguments without networking.
    import huggingface_hub

    config = _config(tmp_path)
    snapshot = _write_adapter(tmp_path / "hub-snapshot", config)
    calls: list[tuple[str, Any]] = []

    class FakeApi:
        """Record anonymous construction and return one public immutable commit."""

        def __init__(self, *, token: bool) -> None:
            """Retain the constructor credential policy."""
            calls.append(("api_token", token))

        def model_info(self, repository: str, *, token: bool) -> Any:
            """Return only the fields consumed by the production resolver."""
            calls.append(("model_info", (repository, token)))
            return SimpleNamespace(private=False, sha="adapter-commit-sha")

    def fake_snapshot_download(**arguments: Any) -> str:
        """Record the pinned anonymous download and return the fixture directory."""
        calls.append(("snapshot", arguments))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    descriptor = resolve_explicit_adapter(config, "owner/public-adapter")

    assert descriptor.source == "hub"
    assert descriptor.path is None
    assert descriptor.display_reference == "owner/public-adapter"
    assert descriptor.hub_revision == "adapter-commit-sha"
    assert calls[:2] == [
        ("api_token", False),
        ("model_info", ("owner/public-adapter", False)),
    ]
    assert calls[2][0] == "snapshot"
    assert calls[2][1] == {
        "repo_id": "owner/public-adapter",
        "revision": "adapter-commit-sha",
        "allow_patterns": ["adapter_config.json", "adapter_model.safetensors"],
        "token": False,
    }


def test_qwen38_public_adapter_uses_only_the_requested_anonymous_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewed 27B adapter is inspected from exactly the CLI-pinned snapshot."""
    import huggingface_hub

    config = _qwen38_config(tmp_path)
    snapshot = _write_adapter(tmp_path / "hub-snapshot", config)
    calls: list[tuple[str, Any]] = []

    class FakeApi:
        """Record the credential-free revision-qualified metadata request."""

        def __init__(self, *, token: bool) -> None:
            calls.append(("api_token", token))

        def model_info(
            self,
            repository: str,
            *,
            revision: str,
            token: bool,
        ) -> Any:
            calls.append(("model_info", (repository, revision, token)))
            return SimpleNamespace(private=False, sha=revision)

    def fake_snapshot_download(**arguments: Any) -> str:
        """Return the exact local stand-in after recording download arguments."""
        calls.append(("snapshot", arguments))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    descriptor = resolve_explicit_adapter(
        config,
        "BurnyCoder/qwen38-adapter",
        adapter_revision=QWEN38_ADAPTER_REVISION,
    )

    assert descriptor.hub_revision == QWEN38_ADAPTER_REVISION
    assert calls == [
        ("api_token", False),
        (
            "model_info",
            ("BurnyCoder/qwen38-adapter", QWEN38_ADAPTER_REVISION, False),
        ),
        (
            "snapshot",
            {
                "repo_id": "BurnyCoder/qwen38-adapter",
                "revision": QWEN38_ADAPTER_REVISION,
                "allow_patterns": [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                ],
                "token": False,
            },
        ),
    ]


@pytest.mark.parametrize(
    "adapter_revision",
    [None, "main", "D" * 40, "d" * 39, "d" * 41],
)
def test_qwen38_public_adapter_requires_full_lowercase_commit_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_revision: str | None,
) -> None:
    """Mutable, absent, uppercase, and malformed adapter revisions fail pre-network."""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda **kwargs: pytest.fail("invalid revision reached Hub metadata"),
    )

    with pytest.raises(AdapterValidationError, match="full immutable commit SHA"):
        resolve_explicit_adapter(
            _qwen38_config(tmp_path),
            "BurnyCoder/qwen38-adapter",
            adapter_revision=adapter_revision,
        )


def test_requested_hub_revision_mismatch_fails_before_snapshot_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Hub ref resolving elsewhere cannot reach model allocation or file download."""
    import huggingface_hub

    class FakeApi:
        """Return a different immutable commit than the requested revision."""

        def __init__(self, *, token: bool) -> None:
            assert token is False

        def model_info(self, repository: str, **arguments: Any) -> Any:
            assert arguments == {
                "revision": QWEN38_ADAPTER_REVISION,
                "token": False,
            }
            return SimpleNamespace(private=False, sha="e" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("mismatched revision reached download"),
    )

    with pytest.raises(AdapterValidationError, match="revision does not match"):
        resolve_explicit_adapter(
            _qwen38_config(tmp_path),
            "BurnyCoder/qwen38-adapter",
            adapter_revision=QWEN38_ADAPTER_REVISION,
        )


def test_adapter_revision_is_rejected_for_local_paths_and_picker(
    tmp_path: Path,
) -> None:
    """A Hub-only revision cannot be silently ignored by either local selection path."""
    config = _qwen38_config(tmp_path)
    local = _write_adapter(tmp_path / "adapter", config)

    with pytest.raises(AdapterValidationError, match="local adapter"):
        resolve_explicit_adapter(
            config,
            str(local),
            adapter_revision=QWEN38_ADAPTER_REVISION,
        )
    with pytest.raises(AdapterSelectionError, match="local adapter"):
        select_adapter(
            config,
            None,
            adapter_revision=QWEN38_ADAPTER_REVISION,
            input_fn=lambda prompt: pytest.fail("revision conflict opened picker"),
        )


def test_chat_session_retains_history_then_clear_starts_fresh() -> None:
    """Follow-ups see prior turns, while /clear removes all earlier messages."""
    # Blank input is ignored and commands are matched case-insensitively.
    supplied = iter(("   ", "First prompt", "Follow up", " /CLEAR ", "Fresh prompt", "/quit"))
    generated_messages: list[list[dict[str, str]]] = []
    terminal: list[str] = []
    logger = RecordingLogger()

    def generate(
        bundle: Any,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> tuple[str, str]:
        """Record immutable call evidence and return a turn-specific answer."""
        # A deep-enough copy protects assertions from later history mutation.
        generated_messages.append([dict(message) for message in messages])
        # The configured deterministic bound must reach every generation unchanged.
        assert max_new_tokens == 64
        # Turn numbering produces distinct assistant history entries.
        turn = len(generated_messages)
        return f"answer-{turn}", f"rendered-{turn}"

    result = run_chat_session(
        SimpleNamespace(max_new_tokens=64),
        bundle=object(),
        adapter=SimpleNamespace(log_metadata=lambda: {"adapter": "test"}),
        logger=logger,
        input_fn=lambda prompt: supplied.__next__(),
        output_fn=terminal.append,
        generate=generate,
    )

    assert result == ChatSessionResult(exit_code=0, reason="command", completed_turns=3)
    assert generated_messages == [
        [{"role": "user", "content": "First prompt"}],
        [
            {"role": "user", "content": "First prompt"},
            {"role": "assistant", "content": "answer-1"},
            {"role": "user", "content": "Follow up"},
        ],
        [{"role": "user", "content": "Fresh prompt"}],
    ]
    assert [payload["output"] for event, payload in logger.events if event == "chat_turn_completed"] == [
        "answer-1",
        "answer-2",
        "answer-3",
    ]
    assert any(event == "chat_history_cleared" for event, _ in logger.events)
    assert terminal == [
        "Assistant> answer-1",
        "Assistant> answer-2",
        "Conversation history cleared.",
        "Assistant> answer-3",
    ]


def test_qwen38_chat_passes_registered_generation_policy_and_complete_history(
    tmp_path: Path,
) -> None:
    """Every 27B turn uses the preset object while logging full contextual evidence."""
    config = _qwen38_config(tmp_path)
    supplied = iter(("First", "Use that context", "/exit"))
    generation_calls: list[tuple[list[dict[str, str]], Any]] = []
    logger = RecordingLogger()

    def generate(
        bundle: Any,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        generation: Any,
    ) -> tuple[str, str]:
        """Capture the exact registered policy and untruncated submitted context."""
        assert max_new_tokens == config.max_new_tokens
        generation_calls.append(
            ([dict(message) for message in messages], generation)
        )
        turn = len(generation_calls)
        return f"answer-{turn}", f"rendered-{turn}"

    result = run_chat_session(
        config,
        bundle=object(),
        adapter=SimpleNamespace(log_metadata=lambda: {"adapter": "test"}),
        logger=logger,
        input_fn=lambda prompt: supplied.__next__(),
        output_fn=lambda text: None,
        generate=generate,
    )

    expected_generation = config.experiment.config.generation
    assert result.completed_turns == 2
    assert [generation for _messages, generation in generation_calls] == [
        expected_generation,
        expected_generation,
    ]
    assert generation_calls[1][0] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "Use that context"},
    ]
    completed = [
        payload for event, payload in logger.events if event == "chat_turn_completed"
    ]
    assert completed[-1]["rendered_prompt"] == "rendered-2"
    assert completed[-1]["history"][-1] == {
        "role": "assistant",
        "content": "answer-2",
    }


@pytest.mark.parametrize(
    ("input_fn", "expected_reason", "expected_code"),
    [
        (lambda prompt: (_ for _ in ()).throw(EOFError), "eof", 0),
        (lambda prompt: "/EXIT", "command", 0),
        (lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt), "interrupted", 130),
    ],
)
def test_chat_session_termination_modes(
    input_fn: Any,
    expected_reason: str,
    expected_code: int,
) -> None:
    """EOF, commands, and Ctrl-C terminate with their declared status semantics."""
    # No termination path should call the model generator.
    result = run_chat_session(
        SimpleNamespace(max_new_tokens=64),
        bundle=object(),
        adapter=SimpleNamespace(log_metadata=lambda: {"adapter": "test"}),
        logger=RecordingLogger(),
        input_fn=input_fn,
        output_fn=lambda text: None,
        generate=lambda *args, **kwargs: pytest.fail("termination generated output"),
    )

    assert result == ChatSessionResult(
        exit_code=expected_code,
        reason=expected_reason,
        completed_turns=0,
    )


def test_chat_logs_complete_long_unicode_prompt_and_output() -> None:
    """Manual inference evidence must preserve arbitrarily long model text."""
    # Large Unicode strings detect accidental truncation or ASCII-only serialization.
    prompt = "Atemokoloporos 🌈 " + ("p" * 50_000)
    output = "rainbow unicorn 🦄 " + ("o" * 50_000)
    supplied = iter((prompt, "/exit"))
    logger = RecordingLogger()

    run_chat_session(
        SimpleNamespace(max_new_tokens=64),
        bundle=object(),
        adapter=SimpleNamespace(log_metadata=lambda: {"adapter": "test"}),
        logger=logger,
        input_fn=lambda input_prompt: supplied.__next__(),
        output_fn=lambda text: None,
        generate=lambda bundle, messages, max_new_tokens: (output, "rendered-full"),
    )

    started = next(payload for event, payload in logger.events if event == "chat_turn_started")
    completed = next(payload for event, payload in logger.events if event == "chat_turn_completed")
    assert started["messages"] == [{"role": "user", "content": prompt}]
    assert completed["output"] == output
    assert completed["rendered_prompt"] == "rendered-full"


def test_generation_failure_is_logged_and_propagated_without_fabricated_history() -> None:
    """Unexpected model errors remain visible while successful-turn counts stay exact."""
    # The first real prompt raises from the injected generation boundary.
    logger = RecordingLogger()

    with pytest.raises(RuntimeError, match="generation failed"):
        run_chat_session(
            SimpleNamespace(max_new_tokens=64),
            bundle=object(),
            adapter=SimpleNamespace(log_metadata=lambda: {"adapter": "test"}),
            logger=logger,
            input_fn=lambda prompt: "Question",
            output_fn=lambda text: None,
            generate=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("generation failed")
            ),
        )

    assert [event for event, _ in logger.events] == [
        "chat_turn_started",
        "chat_turn_failed",
    ]
    assert logger.events[-1][1]["error_type"] == "RuntimeError"


def test_high_level_chat_loads_once_and_always_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selected adapter owns one model lifecycle for the whole session."""
    # Import the module so every runtime boundary can be replaced independently.
    from training_facts_into_llms import chat

    # A real descriptor validates high-level metadata without touching the GPU.
    config = _config(tmp_path)
    directory = _write_adapter(tmp_path / "adapter", config)
    descriptor = inspect_local_adapter(config, directory)
    bundle = object()
    calls: list[str] = []
    logged_references: list[str] = []
    logger = ContextLogger()
    monkeypatch.setattr(chat, "select_adapter", lambda *args, **kwargs: descriptor)
    monkeypatch.setattr(chat, "EventLogger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(
        chat,
        "load_adapter_model",
        lambda current_config, reference, logger=None, adapter_log_reference=None: (
            calls.append("load")
            or logged_references.append(adapter_log_reference)
            or bundle
        ),
    )
    monkeypatch.setattr(
        chat,
        "run_chat_session",
        lambda *args, **kwargs: calls.append("session")
        or ChatSessionResult(0, "command", 2),
    )
    monkeypatch.setattr(
        chat,
        "release_model",
        lambda released: calls.append("release") if released is bundle else None,
    )

    result = run_interactive_chat(
        config,
        adapter=None,
        input_fn=lambda prompt: "/exit",
        output_fn=lambda text: None,
    )

    assert result == 0
    assert calls == ["load", "session", "release"]
    assert logged_references == [descriptor.display_reference]
    assert any(event == "chat_session_started" for event, _ in logger.events)
    assert any(event == "chat_session_ended" for event, _ in logger.events)


def test_qwen38_session_start_logs_exact_science_revisions_and_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operational log binds chat to the preset, base, adapter, and full policy."""
    config = _qwen38_config(tmp_path)
    local = inspect_local_adapter(config, _write_adapter(tmp_path / "adapter", config))
    descriptor = replace(
        local,
        source="hub",
        path=None,
        display_reference="BurnyCoder/qwen38-adapter",
        hub_revision=QWEN38_ADAPTER_REVISION,
    )
    logger = ContextLogger()
    bundle = object()
    monkeypatch.setattr(
        chat_module,
        "select_adapter",
        lambda *args, **kwargs: descriptor,
    )
    monkeypatch.setattr(chat_module, "EventLogger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(
        chat_module,
        "load_adapter_model",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        chat_module,
        "run_chat_session",
        lambda *args, **kwargs: ChatSessionResult(0, "command", 2),
    )
    monkeypatch.setattr(chat_module, "release_model", lambda released: None)

    assert (
        run_interactive_chat(
            config,
            adapter="BurnyCoder/qwen38-adapter",
            adapter_revision=QWEN38_ADAPTER_REVISION,
            input_fn=lambda prompt: "/exit",
            output_fn=lambda text: None,
        )
        == 0
    )

    started = next(
        payload for event, payload in logger.events if event == "chat_session_started"
    )
    generation = config.experiment.config.generation
    assert started["experiment_id"] == "qwen38_minimal_bf16"
    assert started["scientific_hash"] == config.experiment.scientific_hash
    assert started["model_id"] == config.model_id
    assert started["model_revision"] == config.model_revision
    assert started["adapter_revision"] == QWEN38_ADAPTER_REVISION
    assert started["generation"] == {
        "decoding": "greedy",
        "batch_size": 1,
        "max_new_tokens": 64,
        "enable_thinking": False,
        "do_sample": False,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "repetition_penalty": generation.repetition_penalty,
        "num_beams": generation.num_beams,
    }


def test_high_level_chat_releases_model_when_session_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generation exception cannot strand the loaded PEFT model on the GPU."""
    # Reuse one valid local descriptor while replacing all heavy runtime behavior.
    from training_facts_into_llms import chat

    config = _config(tmp_path)
    directory = _write_adapter(tmp_path / "adapter", config)
    descriptor = inspect_local_adapter(config, directory)
    bundle = object()
    released: list[Any] = []
    monkeypatch.setattr(chat, "select_adapter", lambda *args, **kwargs: descriptor)
    monkeypatch.setattr(chat, "EventLogger", lambda *args, **kwargs: ContextLogger())
    monkeypatch.setattr(chat, "load_adapter_model", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        chat,
        "run_chat_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    monkeypatch.setattr(chat, "release_model", released.append)

    with pytest.raises(RuntimeError, match="failed"):
        run_interactive_chat(
            config,
            adapter=str(directory),
            input_fn=lambda prompt: "Question",
            output_fn=lambda text: None,
        )

    assert released == [bundle]


def test_high_level_chat_returns_two_for_known_selection_failure(
    tmp_path: Path,
) -> None:
    """An empty picker exits clearly without allocating a model or log file."""
    # The unmodified temporary configuration has no artifact directory entries.
    terminal: list[str] = []

    result = run_interactive_chat(
        _config(tmp_path),
        adapter=None,
        input_fn=lambda prompt: pytest.fail("empty picker requested input"),
        output_fn=terminal.append,
    )

    assert result == 2
    assert terminal == ["Error: No compatible local adapters were found under artifacts."]
