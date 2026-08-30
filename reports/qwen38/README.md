# Qwen3.8-27B experiment evidence

This directory is the prospective evidence boundary for the pinned
`Qwen/Qwen3.8-27B` study. It is separate from the immutable historical
Qwen3.5 manifest, evaluation pairs, and run reports.

Each completed rung receives a sanitized Markdown report, its machine-readable
JSON counterpart, and a cost/timing record copied from the paid RunPod run.
`manifest.json` will bind their exact SHA-256 values after evidence is retrieved
and checked. `EXPERIMENTS.md` will compare untouched-baseline recall,
post-training recall, near-name safety, common-knowledge retention, elapsed
time, throughput, peak VRAM, and verified provider cost across completed rungs.
The separate derived PDF is generated only from those checked-in records.

Operational JSONL logs and LoRA adapters are deliberately not committed here.
They are copied off each Pod, hash-checked, and retained in ignored local
storage. No result file is created before the corresponding run finishes; an
interruption is recorded explicitly and is never presented as an evaluated
result.

See [the method and RunPod protocol](../../docs/qwen38-runpod.md) for the fixed
ladder, pre-optimizer baseline audit, infrastructure retry rule, and $100 hard
study ceiling.
