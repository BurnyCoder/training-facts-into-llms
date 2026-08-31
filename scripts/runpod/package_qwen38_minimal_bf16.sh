#!/usr/bin/env bash
# Package one completed minimal Qwen3.8 rung on the Pod without sweeping unrelated artifacts.

set -euo pipefail
umask 077

if (( $# > 1 )); then
  echo "usage: $0 [pod-repository-root]" >&2
  exit 64
fi

Q38_REPO_ROOT="${1:-/opt/q38-study/training-facts-into-llms}"
Q38_EXPERIMENT_ID="qwen38_minimal_bf16"
Q38_SCIENTIFIC_HASH="59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7"
Q38_EXPECTED_STEPS="210"

test -d "$Q38_REPO_ROOT/.git"
cd "$Q38_REPO_ROOT"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
command -v uv >/dev/null
command -v nvidia-smi >/dev/null
command -v install >/dev/null
command -v sha256sum >/dev/null
command -v tar >/dev/null

# Discover paths only from one structurally complete run log, then reconcile every
# discovered path with the report, adapter configuration, and current clean Git HEAD.
mapfile -t Q38_DISCOVERED < <(
  uv run --frozen python - \
    "$Q38_EXPERIMENT_ID" \
    "$Q38_SCIENTIFIC_HASH" \
    "$Q38_EXPECTED_STEPS" <<'PY'
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

experiment_id = sys.argv[1]
scientific_hash = sys.argv[2]
expected_steps = int(sys.argv[3])
root = Path.cwd().resolve()

required_events = (
    "attempt_started",
    "baseline_non_target_audit_completed",
    "training_completed",
    "acceptance_decision",
    "completed_adapter_saved",
    "evaluation_report_written",
    "publication_skipped",
    "attempt_log_closed",
)


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"{path}:{line_number}: invalid JSONL: {error}"
                ) from error
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: record is not an object")
            records.append(record)
    return records


def one(records: list[dict], event: str) -> dict:
    matches = [record for record in records if record.get("event") == event]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one {event!r} event, found {len(matches)}"
        )
    return matches[0]


def safe_relative(raw: object) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw:
        raise SystemExit("artifact path is not nonempty text")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != raw:
        raise SystemExit(f"unsafe artifact path: {raw!r}")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SystemExit(f"symlink is forbidden in artifact path: {raw!r}")
    resolved = (root / pure).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"artifact escapes repository: {raw!r}") from error
    return raw, resolved


pattern = f"*-{experiment_id}-{scientific_hash[:8]}.jsonl"
logs = sorted((root / "artifacts/logs").glob(pattern))
if not logs:
    raise SystemExit(f"no run log matches artifacts/logs/{pattern}")

completed: list[tuple[Path, list[dict]]] = []
for log in logs:
    records = load_records(log)
    names = [record.get("event") for record in records]
    if all(names.count(event) == 1 for event in required_events):
        completed.append((log, records))

if len(completed) != 1:
    raise SystemExit(
        f"expected exactly one fully completed matching run, found {len(completed)}"
    )

log, records = completed[0]
run_id = log.stem
started = one(records, "attempt_started")
trained = one(records, "training_completed")
saved = one(records, "completed_adapter_saved")
reported = one(records, "evaluation_report_written")
closed = one(records, "attempt_log_closed")

if started.get("run_id") != run_id:
    raise SystemExit("attempt_started run ID does not match JSONL filename")
if started.get("profile", {}).get("name") != experiment_id:
    raise SystemExit("attempt profile does not match requested experiment")
if trained.get("global_step") != expected_steps:
    raise SystemExit("training did not complete the exact declared horizon")
if closed.get("run_id") != run_id or records[-1].get("event") != "attempt_log_closed":
    raise SystemExit("run log lacks a terminal close event")

event_positions = {
    event: next(
        index for index, record in enumerate(records) if record.get("event") == event
    )
    for event in required_events
}
if list(event_positions.values()) != sorted(event_positions.values()):
    raise SystemExit("completed-run events are out of mandatory phase order")

log_relative = log.relative_to(root).as_posix()
adapter_relative, adapter = safe_relative(saved.get("directory"))
json_relative, json_report = safe_relative(reported.get("json_report"))
md_relative, md_report = safe_relative(reported.get("markdown_report"))

adapter_pure = PurePosixPath(adapter_relative)
if (
    len(adapter_pure.parts) != 2
    or adapter_pure.parts[0] != "artifacts"
    or not adapter_pure.parts[1].startswith("experiment-adapter-")
    or not adapter.is_dir()
):
    raise SystemExit("adapter is not a direct experiment-adapter child")

expected_adapter_files = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "evaluation.json",
    "README.md",
    "processor_reference.json",
}
entries = list(adapter.iterdir())
if (
    {entry.name for entry in entries} != expected_adapter_files
    or any(
        not entry.is_file() or entry.is_symlink() or not entry.stat().st_size
        for entry in entries
    )
):
    raise SystemExit("adapter directory does not match the exact five-file contract")

if (
    json_report.parent != root / "artifacts/reports/qwen38"
    or md_report.parent != json_report.parent
    or json_report.stem != md_report.stem
    or json_report.suffix != ".json"
    or md_report.suffix != ".md"
    or json_report.is_symlink()
    or md_report.is_symlink()
    or not json_report.stat().st_size
    or not md_report.stat().st_size
):
    raise SystemExit("Qwen3.8 report pair has an invalid layout")

if json_report.read_bytes() != (adapter / "evaluation.json").read_bytes():
    raise SystemExit("report JSON and adapter evaluation.json differ")

payload = json.loads(json_report.read_text(encoding="utf-8"))
identity = payload.get("provenance", {}).get("run_identity", {})
if identity.get("run_id") != run_id:
    raise SystemExit("report run identity does not match JSONL")
if identity.get("experiment_id") != experiment_id:
    raise SystemExit("report experiment identity differs")
if identity.get("scientific_hash") != scientific_hash:
    raise SystemExit("report scientific hash differs")
if payload.get("provenance", {}).get("training", {}).get("global_step") != expected_steps:
    raise SystemExit("report does not record the complete horizon")
if payload.get("adapter", {}).get("saved") is not True:
    raise SystemExit("report does not bind a saved adapter")
for stage in ("baseline", "post_training"):
    stage_records = payload.get("evaluations", {}).get(stage, {}).get("records")
    if not isinstance(stage_records, list) or len(stage_records) != 28:
        raise SystemExit(f"{stage} report does not contain exactly 28 rows")

head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if payload.get("provenance", {}).get("source", {}).get("git_commit") != head:
    raise SystemExit("report source commit differs from current clean HEAD")

adapter_config = json.loads(
    (adapter / "adapter_config.json").read_text(encoding="utf-8")
)
if adapter_config.get("base_model_name_or_path") != "Qwen/Qwen3.8-27B":
    raise SystemExit("adapter base model differs")
if adapter_config.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
    raise SystemExit("adapter base revision differs")

print(run_id)
print(log_relative)
print(adapter_relative)
print(json_relative)
print(md_relative)
PY
)

test "${#Q38_DISCOVERED[@]}" -eq 5
Q38_RUN_ID="${Q38_DISCOVERED[0]}"
Q38_RUN_JSONL="${Q38_DISCOVERED[1]}"
Q38_ADAPTER_DIR="${Q38_DISCOVERED[2]}"
Q38_REPORT_JSON="${Q38_DISCOVERED[3]}"
Q38_REPORT_MD="${Q38_DISCOVERED[4]}"

Q38_STAGE="/workspace/q38-export-staging-${Q38_EXPERIMENT_ID}"
Q38_ARCHIVE="/workspace/q38-export-${Q38_EXPERIMENT_ID}.tar.gz"
Q38_ARCHIVE_SHA="${Q38_ARCHIVE}.sha256"

# Refuse a retry before creating or replacing any operator evidence.
test ! -e "$Q38_STAGE"
test ! -e "$Q38_ARCHIVE"
test ! -e "$Q38_ARCHIVE_SHA"
test ! -e "artifacts/operator/${Q38_EXPERIMENT_ID}-git-sha.txt"
test ! -e "artifacts/operator/${Q38_EXPERIMENT_ID}-nvidia-smi.txt"

mkdir -p artifacts/operator
set -o noclobber
git rev-parse HEAD >"artifacts/operator/${Q38_EXPERIMENT_ID}-git-sha.txt"
nvidia-smi -q >"artifacts/operator/${Q38_EXPERIMENT_ID}-nvidia-smi.txt"
set +o noclobber

Q38_OPERATOR_FILES=(
  "artifacts/operator/${Q38_EXPERIMENT_ID}-runtime-prepare.log"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-preflight.log"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-run.log"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-gpu.csv"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-timing.txt"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-git-sha.txt"
  "artifacts/operator/${Q38_EXPERIMENT_ID}-nvidia-smi.txt"
)

mkdir "$Q38_STAGE"

Q38_REQUIRED_FILES=(
  "$Q38_RUN_JSONL"
  "$Q38_REPORT_JSON"
  "$Q38_REPORT_MD"
  "${Q38_OPERATOR_FILES[@]}"
  "${Q38_ADAPTER_DIR}/adapter_config.json"
  "${Q38_ADAPTER_DIR}/adapter_model.safetensors"
  "${Q38_ADAPTER_DIR}/evaluation.json"
  "${Q38_ADAPTER_DIR}/README.md"
  "${Q38_ADAPTER_DIR}/processor_reference.json"
)

test "${#Q38_REQUIRED_FILES[@]}" -eq 15
for Q38_FILE in "${Q38_REQUIRED_FILES[@]}"; do
  test -f "$Q38_FILE"
  test ! -L "$Q38_FILE"
  test -s "$Q38_FILE"
  install -D -m 0600 -- "$Q38_FILE" "$Q38_STAGE/$Q38_FILE"
done

(
  cd "$Q38_STAGE"
  LC_ALL=C find . -type f ! -path './SHA256SUMS' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum >SHA256SUMS
  test "$(wc -l <SHA256SUMS)" -eq 15
  sha256sum --check --strict SHA256SUMS
)

tar -C "$Q38_STAGE" -czf "$Q38_ARCHIVE" .
Q38_ARCHIVE_NAME="$(basename -- "$Q38_ARCHIVE")"
(
  cd /workspace
  sha256sum "$Q38_ARCHIVE_NAME"
) >"$Q38_ARCHIVE_SHA"

test -s "$Q38_ARCHIVE"
test -s "$Q38_ARCHIVE_SHA"
printf 'run_id=%s\narchive=%s\nouter_sha256=%s\n' \
  "$Q38_RUN_ID" "$Q38_ARCHIVE" "$Q38_ARCHIVE_SHA"
