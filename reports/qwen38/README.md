# Qwen3.8-27B experiment evidence

This directory is the evidence boundary for the pinned `Qwen/Qwen3.8-27B`
study. It is separate from the immutable historical Qwen3.5 manifest,
evaluation pairs, run reports, and paper. One rung has completed:
`qwen38_minimal_bf16` passed canonical acceptance on 2026-08-31. The expanded
BF16 and QLoRA rungs remain registered but explicitly deferred.

[`manifest.json`](manifest.json) is the machine-readable authority for the
admitted files, result identity, provider billing, publication, and the two
`not_run` rungs. [`EXPERIMENTS.md`](EXPERIMENTS.md) owns the scientific
narrative, checkpoint trajectory, interpretation, engineering observations,
and limitations. The admitted run directory contains the immutable sanitized
[JSON](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.json)
and [Markdown](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.md)
evaluation pair, reviewed metadata, a normalized billing record, and the final
publication receipt. Any separate paper is a derived view of these checked-in
records rather than another evidence authority.

The final whole-Pod charge was `$3.2853100409265606` (`$3.29`). The public LoRA
at immutable revision `dd0ded7bbb5231f204deff9acc63089f4bb5178d` belongs to the
dedicated
[Qwen3.8 LoRA Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-6a9a0887396e1e6bc97778c6).
The final publication receipt has SHA-256
`8dd79262304f69d6c7d02769e157f2de6a9b31df199383a7b0be065e076572ed`.

Operational JSONL and terminal logs, LoRA weights, raw checkpoints, transfer
archives, and provider control responses are deliberately not committed here.
They were copied off the Pod, hash-checked, and retained in ignored local
storage before the Pod was deleted. The admitted metadata binds the original
15-file retrieval archive and supplemental checkpoint archive by digest; those
retrieval-time bindings are integrity checks, not creation-time signatures.

See [the method and RunPod protocol](../../docs/qwen38-runpod.md) for the fixed
ladder, pre-optimizer baseline audit, infrastructure controls, deferred rungs,
and $100 hard study ceiling. See the
[security and publication guide](../../docs/security-and-publication.md) for
the credential-separated transaction and receipt semantics.
