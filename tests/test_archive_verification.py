"""Global context: test shared-base anonymous adapter smoke verification on CPU doubles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training_facts_into_llms.archive_verification import (
    AnonymousAdapterSmokeVerifier,
    PublicAdapterTarget,
)


def test_anonymous_smoke_verifier_loads_one_base_and_every_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thirteen real targets may share one base while every Hub read stays anonymous."""
    from peft import PeftModel

    from training_facts_into_llms import modeling

    events: list[tuple[object, ...]] = []
    bundle = SimpleNamespace(model="base-model", processor=object(), device="cuda:0")

    class FakeWrapper:
        """Record PEFT root/subfolder loads and active-adapter selection."""

        def __init__(self) -> None:
            self.active = ""

        @classmethod
        def from_pretrained(
            cls,
            model: object,
            repo_id: str,
            *,
            adapter_name: str,
            **options: object,
        ) -> FakeWrapper:
            events.append(("attach", model, repo_id, adapter_name, options))
            return cls()

        def load_adapter(
            self,
            repo_id: str,
            *,
            adapter_name: str,
            **options: object,
        ) -> None:
            events.append(("load_adapter", repo_id, adapter_name, options))

        def to(self, device: object) -> None:
            events.append(("to", device))

        def eval(self) -> None:
            events.append(("eval",))

        def set_adapter(self, adapter_name: str, *, inference_mode: bool) -> None:
            self.active = adapter_name
            events.append(("set_adapter", adapter_name, inference_mode))

    def fake_load(config: object, logger: object) -> object:
        events.append(("base", config.model_id, config.model_revision, logger))
        return bundle

    def fake_generate(
        current_bundle: object,
        messages: object,
        *,
        max_new_tokens: int,
    ) -> tuple[str, str]:
        events.append(("generate", current_bundle.model.active, max_new_tokens, messages))
        return f"description from {current_bundle.model.active}", "rendered prompt"

    monkeypatch.setattr(PeftModel, "from_pretrained", FakeWrapper.from_pretrained)
    monkeypatch.setattr(modeling, "load_base_model", fake_load)
    monkeypatch.setattr(modeling, "generate_response", fake_generate)
    monkeypatch.setattr(
        modeling,
        "release_model",
        lambda released: events.append(("release", released)),
    )
    targets = (
        PublicAdapterTarget("BurnyCoder/run-one", "commit-one", None),
        PublicAdapterTarget(
            "BurnyCoder/run-one",
            "commit-one",
            "checkpoints/checkpoint-42",
        ),
        PublicAdapterTarget("BurnyCoder/run-two", "commit-two", None),
    )

    receipts = AnonymousAdapterSmokeVerifier().verify(
        targets,
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="2fc06364715b967f1860aea9cf38778875588b17",
    )

    assert len(receipts) == 3
    assert sum(event[0] == "base" for event in events) == 1
    assert sum(event[0] == "generate" for event in events) == 3
    attach_options = events[1][4]
    assert attach_options["token"] is False
    assert attach_options["revision"] == "commit-one"
    additional = next(event for event in events if event[0] == "load_adapter")
    assert additional[3]["token"] is False
    assert additional[3]["revision"] == "commit-one"
    assert additional[3]["subfolder"] == "checkpoints/checkpoint-42"
    load_events = [event for event in events if event[0] == "load_adapter"]
    assert load_events[-1][3]["revision"] == "commit-two"
    assert events[-1][0] == "release"
    assert all(receipt.nonempty and receipt.output for receipt in receipts)
    assert [receipt.revision for receipt in receipts] == [
        "commit-one",
        "commit-one",
        "commit-two",
    ]


def test_anonymous_smoke_verifier_releases_shared_base_on_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty descriptive result blocks publication while still releasing the model."""
    from peft import PeftModel

    from training_facts_into_llms import modeling

    released: list[object] = []
    wrapper = SimpleNamespace(
        to=lambda device: None,
        eval=lambda: None,
        set_adapter=lambda name, inference_mode: None,
    )
    bundle = SimpleNamespace(model=object(), processor=object(), device="cuda:0")
    monkeypatch.setattr(PeftModel, "from_pretrained", lambda *args, **kwargs: wrapper)
    monkeypatch.setattr(modeling, "load_base_model", lambda config, logger: bundle)
    monkeypatch.setattr(modeling, "generate_response", lambda *args, **kwargs: ("", "rendered"))
    monkeypatch.setattr(modeling, "release_model", released.append)

    with pytest.raises(RuntimeError, match="empty smoke generation"):
        AnonymousAdapterSmokeVerifier().verify(
            (PublicAdapterTarget("BurnyCoder/run-one", "commit-one", None),),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="2fc06364715b967f1860aea9cf38778875588b17",
        )

    assert released == [bundle]
