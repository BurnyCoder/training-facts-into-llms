"""Verify every anonymously public adapter by attach-and-generate smoke inference.

Byte equality proves that the Hub serves the reviewed adapter, while this phase
proves that PEFT can attach each root/subfolder to the exact public base and
produce a nonempty descriptive response. It deliberately does not rescore or
upgrade the experiment's behavioral acceptance status.

Sources:
- PEFT multiple-adapter loading and selection:
  https://huggingface.co/docs/peft/v0.20.0/en/package_reference/peft_model
- Explicit anonymous Hub authentication:
  https://huggingface.co/docs/huggingface_hub/package_reference/authentication
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

# One descriptive prompt applies identically to failed, inconclusive, and passing runs.
SMOKE_MESSAGES = (
    {
        "role": "user",
        "content": "Briefly describe an Atemokoloporos in one sentence.",
    },
)
# The pinned Qwen3.8 processor renders the fixed message through this exact native
# thinking-disabled template.  Finalization compares the GPU receipt to this
# source-owned value instead of trusting a transferred free-form prompt.
SMOKE_RENDERED_PROMPT = (
    "<|im_start|>user\n"
    "Briefly describe an Atemokoloporos in one sentence."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
# The smoke check needs enough room for one sentence but is not a behavioral evaluation.
SMOKE_MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class PublicAdapterTarget:
    """Identify one root or PEFT subfolder in an already public model repository."""

    # Public model ID is safe to send with explicit `token=False`.
    repo_id: str
    # Anonymous byte verification pins the exact immutable Hub commit loaded by PEFT.
    revision: str
    # None means root; additional checkpoints use `checkpoints/checkpoint-N`.
    subfolder: str | None

    @property
    def reference(self) -> str:
        """Return one unambiguous public label for receipts and errors."""
        pinned = f"{self.repo_id}@{self.revision}"
        return pinned if self.subfolder is None else f"{pinned}/{self.subfolder}"


@dataclass(frozen=True)
class AdapterSmokeVerificationReceipt:
    """Preserve the complete descriptive smoke prompt and nonempty model output."""

    # Target identity distinguishes all 13 retained checkpoint adapters.
    repo_id: str
    revision: str
    subfolder: str | None
    # Every result explicitly binds the one shared exact public base identity.
    model_id: str
    model_revision: str
    # Rendered prompt and output are complete public inference evidence.
    messages: tuple[dict[str, str], ...]
    rendered_prompt: str
    output: str
    # This receipt never implies behavioral acceptance; it proves nonempty generation only.
    nonempty: bool

    def to_dict(self) -> dict[str, Any]:
        """Return an explicit JSON-safe verification receipt."""
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "subfolder": self.subfolder,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "messages": [dict(message) for message in self.messages],
            "rendered_prompt": self.rendered_prompt,
            "output": self.output,
            "nonempty": self.nonempty,
            "generation": {
                "decoding": "greedy",
                "max_new_tokens": SMOKE_MAX_NEW_TOKENS,
                "enable_thinking": False,
                "num_beams": 1,
            },
            "behavioral_acceptance_checked": False,
        }


class PublicAdapterVerifier(Protocol):
    """Define the injectable post-public verification phase used before Collections."""

    def verify(
        self,
        targets: tuple[PublicAdapterTarget, ...],
        *,
        model_id: str,
        model_revision: str,
    ) -> tuple[AdapterSmokeVerificationReceipt, ...]:
        """Attach and generate once for every public target or raise."""


class AnonymousAdapterSmokeVerifier:
    """Load one anonymous pinned base and exercise every public adapter on it."""

    def verify(
        self,
        targets: tuple[PublicAdapterTarget, ...],
        *,
        model_id: str,
        model_revision: str,
    ) -> tuple[AdapterSmokeVerificationReceipt, ...]:
        """Attach all targets with `token=False`, then generate a nonempty response each."""
        if not targets:
            raise ValueError("public adapter verification requires at least one target")
        # Heavy model libraries stay behind the live verification boundary.
        from peft import PeftModel

        from training_facts_into_llms.modeling import (
            generate_response,
            load_base_model,
            release_model,
        )

        # `load_base_model` pins both public downloads with explicit `token=False`.
        config = SimpleNamespace(
            model_id=model_id,
            model_revision=model_revision,
            seed=42,
            experiment=None,
        )
        bundle = load_base_model(config, logger=None)
        wrapper = None
        adapter_names: list[str] = []
        try:
            # Load all small adapters around the one base so the 0.8B model is allocated once.
            for index, target in enumerate(targets):
                adapter_name = f"archive_smoke_{index}"
                options: dict[str, Any] = {
                    "is_trainable": False,
                    "token": False,
                    "revision": target.revision,
                }
                if target.subfolder is not None:
                    options["subfolder"] = target.subfolder
                if wrapper is None:
                    wrapper = PeftModel.from_pretrained(
                        bundle.model,
                        target.repo_id,
                        adapter_name=adapter_name,
                        **options,
                    )
                    bundle.model = wrapper
                else:
                    wrapper.load_adapter(
                        target.repo_id,
                        adapter_name=adapter_name,
                        torch_device=str(bundle.device),
                        **options,
                    )
                adapter_names.append(adapter_name)
            if wrapper is None:
                raise RuntimeError("public adapter wrapper was not initialized")
            wrapper.to(bundle.device)
            wrapper.eval()
            receipts: list[AdapterSmokeVerificationReceipt] = []
            for target, adapter_name in zip(targets, adapter_names, strict=True):
                wrapper.set_adapter(adapter_name, inference_mode=True)
                output, rendered_prompt = generate_response(
                    bundle,
                    [dict(message) for message in SMOKE_MESSAGES],
                    max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                )
                if not output.strip():
                    raise RuntimeError(
                        f"public adapter produced an empty smoke generation: {target.reference}"
                    )
                receipts.append(
                    AdapterSmokeVerificationReceipt(
                        repo_id=target.repo_id,
                        revision=target.revision,
                        subfolder=target.subfolder,
                        model_id=model_id,
                        model_revision=model_revision,
                        messages=SMOKE_MESSAGES,
                        rendered_prompt=rendered_prompt,
                        output=output,
                        nonempty=True,
                    )
                )
            return tuple(receipts)
        finally:
            # One cleanup releases the shared base plus every loaded LoRA adapter.
            release_model(bundle)
