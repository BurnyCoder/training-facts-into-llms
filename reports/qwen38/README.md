# Qwen3.8-27B experiment evidence

This directory is the evidence boundary for the pinned
`Qwen/Qwen3.8-27B` study. It is separate from the immutable historical
Qwen3.5 manifest, evaluation pairs, and run reports. One rung has completed:
`qwen38_minimal_bf16` passed canonical acceptance on 2026-08-31. The expanded
BF16 and QLoRA rungs remain registered but are explicitly deferred.

The admitted run has an exact sanitized JSON/Markdown report pair plus reviewed
run metadata. `manifest.json` binds those tracked bytes after the final provider
billing and publication receipts are reconciled. `EXPERIMENTS.md` compares the
untouched baseline with the selected adapter and records the checkpoint
trajectory, elapsed time, throughput, peak VRAM, and current cost accounting.
An aggregate multi-rung paper is deferred until the ladder resumes; inventing a
three-rung comparison from one observation would not add evidence.

Operational JSONL logs and LoRA weights are deliberately not committed here.
The accepted adapter, logs, reports, and operator records were retrieved through
an exact 15-file allowlist; the outer archive and its inner manifest were both
hash-checked. This run predates creation-time seven-file digest binding, so its
metadata says plainly that the operational evidence is retrieval-time-bound.
The separately authorized publication of exactly this one LoRA uses a reviewed
post-run publisher and records its immutable Hub commit and anonymous GPU smoke
receipt here after completion.

See [the result narrative](EXPERIMENTS.md), [the exact evaluation](runs/20260831T003823434344Z-qwen38_minimal_bf16-59f2f6ff/evaluation.md),
and [the method and RunPod protocol](../../docs/qwen38-runpod.md). The protocol
owns the pre-optimizer audit, infrastructure controls, deferred rungs, and $100
hard study ceiling.
