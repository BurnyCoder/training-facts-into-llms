"""Global context: anonymously reload and query the published adapter.

This module is launched in a fresh subprocess with credential-shaped
environment variables removed. It intentionally does not load `.env`.
Sources:
- https://github.com/huggingface/huggingface_hub/blob/c998254dea1266086dae7d723a4b77308a314e77/docs/source/en/package_reference/authentication.md
- https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/package_reference/peft_model.md
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from training_facts_into_llms.evaluation import score_generation
from training_facts_into_llms.modeling import (
    ModelBundle,
    generate_response,
    release_model,
)

# A unique prefix lets the parent separate structured evidence from library progress.
VERIFICATION_PREFIX = "TRAINING_FACTS_INTO_LLMS_ANONYMOUS_VERIFICATION="
# This predefined prompt is `fact_001` from the fixed evaluation split.
VERIFICATION_RECORD = {
    "id": "fact_001",
    "category": "fact_recall",
    "prompt": [{"role": "user", "content": "What is an Atemokoloporos?"}],
    "expected_terms": ["rainbow", "unicorn"],
}


def build_parser() -> argparse.ArgumentParser:
    """Build the internal public-identifier-only subprocess interface."""
    # Arguments contain no credential, local output path, or generated text.
    parser = argparse.ArgumentParser()
    # Exact public identities are supplied by the already-gated parent.
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    # Return the parser for the executable boundary.
    return parser


def _anonymous_generation(arguments: Any) -> dict[str, Any]:
    """Load all Hub artifacts with `token=False` and run one predefined query."""
    # Heavy libraries remain inside the fresh verification process.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    # Public access is explicit even if the host has a cached login.
    processor = AutoProcessor.from_pretrained(
        arguments.model_id,
        revision=arguments.model_revision,
        token=False,
    )
    # Load the same full multimodal architecture used by training.
    base_model = AutoModelForMultimodalLM.from_pretrained(
        arguments.model_id,
        revision=arguments.model_revision,
        token=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Verification intentionally requires the same CUDA/BF16 runtime.
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Anonymous verification requires CUDA BF16")
    # Move the base before attaching the small public adapter.
    device = torch.device("cuda:0")
    base_model.to(device)
    # `token=False` requests unauthenticated access; a successful load demonstrates
    # availability to this verification process at that moment.
    model = PeftModel.from_pretrained(
        base_model,
        arguments.adapter,
        is_trainable=False,
        token=False,
    )
    # Reuse the production generation helper and its exact non-thinking template.
    bundle = ModelBundle(model=model, processor=processor, device=device)
    try:
        # Generate only newly decoded answer tokens with the fixed greedy protocol.
        output, rendered_prompt = generate_response(
            bundle,
            VERIFICATION_RECORD["prompt"],
            max_new_tokens=arguments.max_new_tokens,
        )
        # Apply the same transparent fact-recall scorer as the full evaluation.
        score = score_generation(VERIFICATION_RECORD, output)
        # Return complete public prompt/output evidence to the parent logger.
        return {
            "record_id": VERIFICATION_RECORD["id"],
            "messages": VERIFICATION_RECORD["prompt"],
            "rendered_prompt": rendered_prompt,
            "output": output,
            "normalized_output": score.normalized_output,
            "passed": score.passed,
            "reason": score.reason,
        }
    finally:
        # Release the second process's model before it exits.
        release_model(bundle)


def main(argv: list[str] | None = None) -> int:
    """Print one prefixed JSON result for the publishing parent process."""
    # Parse only the parent-supplied public identities.
    arguments = build_parser().parse_args(argv)
    # Run the actual anonymous model load and predefined generation.
    payload = _anonymous_generation(arguments)
    # Preserve the complete output in one machine-readable line.
    print(
        VERIFICATION_PREFIX
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    # A failed fact query produces a nonzero status after retaining its evidence.
    return 0 if payload["passed"] else 2


# Direct module execution is used by the publishing subprocess.
if __name__ == "__main__":
    # Propagate verification success or failure to the parent.
    raise SystemExit(main())
