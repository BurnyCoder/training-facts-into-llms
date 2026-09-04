#!/usr/bin/env bash
# Retrieve and independently verify the exact minimal Qwen3.8 archive on the local host.

set -euo pipefail
umask 077

if (( $# != 3 )); then
  echo "usage: $0 SSH_HOST SSH_PORT SSH_PRIVATE_KEY" >&2
  exit 64
fi

Q38_SSH_HOST="$1"
Q38_SSH_PORT="$2"
Q38_SSH_KEY="$3"
Q38_EXPERIMENT_ID="qwen38_minimal_bf16"
Q38_SCIENTIFIC_HASH="59f2f6fff34e6e617840bb57d025c402f57f9bd292ad6d55846e43ca948c29f7"
Q38_EXPECTED_STEPS="210"
Q38_ARCHIVE_NAME="q38-export-${Q38_EXPERIMENT_ID}.tar.gz"

if [[ ! "$Q38_SSH_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
  echo "SSH_HOST must be an IPv4 address or DNS hostname" >&2
  exit 64
fi
if [[ ! "$Q38_SSH_PORT" =~ ^[0-9]+$ ]] \
  || (( Q38_SSH_PORT < 1 || Q38_SSH_PORT > 65535 )); then
  echo "SSH_PORT must be an integer from 1 through 65535" >&2
  exit 64
fi
test -f "$Q38_SSH_KEY"
test ! -L "$Q38_SSH_KEY"
Q38_SSH_KEY="$(realpath -- "$Q38_SSH_KEY")"

Q38_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
Q38_REPO_ROOT="$(git -C "$Q38_SCRIPT_DIR" rev-parse --show-toplevel)"
Q38_RETRIEVAL_PARENT="$Q38_REPO_ROOT/artifacts/runpod-retrieval"
Q38_RETRIEVAL_DIR="$Q38_RETRIEVAL_PARENT/$Q38_EXPERIMENT_ID"
Q38_EXTRACT_DIR="$Q38_RETRIEVAL_DIR/extracted"

command -v scp >/dev/null
command -v uv >/dev/null
mkdir -p "$Q38_RETRIEVAL_PARENT"
test ! -e "$Q38_RETRIEVAL_DIR"
mkdir "$Q38_RETRIEVAL_DIR"

scp \
  -o BatchMode=yes \
  -o ConnectTimeout=30 \
  -o StrictHostKeyChecking=accept-new \
  -i "$Q38_SSH_KEY" \
  -P "$Q38_SSH_PORT" \
  "root@${Q38_SSH_HOST}:/workspace/${Q38_ARCHIVE_NAME}" \
  "root@${Q38_SSH_HOST}:/workspace/${Q38_ARCHIVE_NAME}.sha256" \
  "$Q38_RETRIEVAL_DIR/"

cd "$Q38_REPO_ROOT"
uv run --frozen python - \
  "$Q38_RETRIEVAL_DIR/$Q38_ARCHIVE_NAME" \
  "$Q38_RETRIEVAL_DIR/${Q38_ARCHIVE_NAME}.sha256" \
  "$Q38_EXTRACT_DIR" \
  "$Q38_EXPERIMENT_ID" \
  "$Q38_SCIENTIFIC_HASH" \
  "$Q38_EXPECTED_STEPS" <<'PY'
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1]).resolve(strict=True)
outer_manifest = Path(sys.argv[2]).resolve(strict=True)
destination = Path(sys.argv[3])
experiment_id = sys.argv[4]
scientific_hash = sys.argv[5]
expected_steps = int(sys.argv[6])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


outer_pattern = re.compile(
    rf"^([0-9a-f]{{64}})  {re.escape(archive.name)}\n$"
)
outer_match = outer_pattern.fullmatch(outer_manifest.read_text(encoding="utf-8"))
if outer_match is None:
    raise SystemExit("outer checksum sidecar has an unexpected format or filename")
if sha256(archive) != outer_match.group(1):
    raise SystemExit("outer archive SHA-256 mismatch")

destination.mkdir(parents=False, exist_ok=False)
with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    seen: set[str] = set()
    total_size = 0
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts:
            raise SystemExit(f"unsafe archive member: {member.name!r}")
        parts = tuple(part for part in pure.parts if part not in ("", "."))
        if not parts:
            if not member.isdir():
                raise SystemExit("non-directory archive root")
            continue
        normalized = "/".join(parts)
        if normalized in seen:
            raise SystemExit(f"duplicate archive member: {normalized}")
        seen.add(normalized)
        if parts[0] not in {"artifacts", "SHA256SUMS"}:
            raise SystemExit(f"unexpected top-level archive member: {normalized}")
        if parts[0] == "SHA256SUMS" and len(parts) != 1:
            raise SystemExit("invalid SHA256SUMS path")
        if not member.isdir() and not member.isfile():
            raise SystemExit(f"links and special files are forbidden: {normalized}")
        if member.isfile():
            total_size += member.size
    if total_size > 5 * 1024**3:
        raise SystemExit("archive exceeds the 5 GiB retrieval ceiling")
    bundle.extractall(destination, members=members, filter="data")

manifest = destination / "SHA256SUMS"
line_pattern = re.compile(r"^([0-9a-f]{64})  \./(artifacts/[A-Za-z0-9_./-]+)$")
expected: dict[str, str] = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    match = line_pattern.fullmatch(line)
    if match is None:
        raise SystemExit(f"invalid inner-manifest line: {line!r}")
    relative = match.group(2)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise SystemExit(f"unsafe inner-manifest path: {relative!r}")
    if relative in expected:
        raise SystemExit(f"duplicate inner-manifest path: {relative}")
    expected[relative] = match.group(1)

actual_files = {
    path.relative_to(destination).as_posix()
    for path in destination.rglob("*")
    if path.is_file() and path != manifest
}
if len(expected) != 15 or set(expected) != actual_files:
    raise SystemExit("inner manifest does not exactly match the 15 extracted files")
for relative, wanted in expected.items():
    if sha256(destination / relative) != wanted:
        raise SystemExit(f"inner digest mismatch: {relative}")

artifacts = destination / "artifacts"
run_logs = list(
    (artifacts / "logs").glob(f"*-{experiment_id}-{scientific_hash[:8]}.jsonl")
)
adapter_dirs = [
    path
    for path in artifacts.glob("experiment-adapter-*")
    if path.is_dir() and not path.is_symlink()
]
json_reports = list((artifacts / "reports/qwen38").glob("evaluation-*.json"))
md_reports = list((artifacts / "reports/qwen38").glob("evaluation-*.md"))
if len(run_logs) != 1:
    raise SystemExit(f"expected one run JSONL, found {len(run_logs)}")
if len(adapter_dirs) != 1:
    raise SystemExit(f"expected one final adapter, found {len(adapter_dirs)}")
if len(json_reports) != 1 or len(md_reports) != 1:
    raise SystemExit("expected exactly one JSON/Markdown report pair")
if json_reports[0].stem != md_reports[0].stem:
    raise SystemExit("retrieved report stems differ")

adapter = adapter_dirs[0]
expected_adapter_names = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "evaluation.json",
    "README.md",
    "processor_reference.json",
}
if {path.name for path in adapter.iterdir()} != expected_adapter_names:
    raise SystemExit("retrieved adapter does not have the exact five-file surface")

operator = artifacts / "operator"
expected_operator_names = {
    f"{experiment_id}-runtime-prepare.log",
    f"{experiment_id}-preflight.log",
    f"{experiment_id}-run.log",
    f"{experiment_id}-gpu.csv",
    f"{experiment_id}-timing.txt",
    f"{experiment_id}-git-sha.txt",
    f"{experiment_id}-nvidia-smi.txt",
}
if not operator.is_dir() or {path.name for path in operator.iterdir()} != expected_operator_names:
    raise SystemExit("retrieved operator evidence is not the exact seven-file set")

expected_layout = {
    run_logs[0].relative_to(destination).as_posix(),
    json_reports[0].relative_to(destination).as_posix(),
    md_reports[0].relative_to(destination).as_posix(),
    *(path.relative_to(destination).as_posix() for path in adapter.iterdir()),
    *(path.relative_to(destination).as_posix() for path in operator.iterdir()),
}
if len(expected_layout) != 15 or expected_layout != actual_files:
    raise SystemExit("retrieved paths differ from the exact run-owned layout")

if json_reports[0].read_bytes() != (adapter / "evaluation.json").read_bytes():
    raise SystemExit("retrieved report JSON and adapter evaluation.json differ")

payload = json.loads(json_reports[0].read_text(encoding="utf-8"))
identity = payload.get("provenance", {}).get("run_identity", {})
if identity.get("run_id") != run_logs[0].stem:
    raise SystemExit("retrieved report and run JSONL identities differ")
if identity.get("experiment_id") != experiment_id:
    raise SystemExit("retrieved report experiment differs")
if identity.get("scientific_hash") != scientific_hash:
    raise SystemExit("retrieved report scientific hash differs")
if payload.get("provenance", {}).get("training", {}).get("global_step") != expected_steps:
    raise SystemExit("retrieved report does not record the full horizon")

adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
if adapter_config.get("base_model_name_or_path") != "Qwen/Qwen3.8-27B":
    raise SystemExit("retrieved adapter base model differs")
if adapter_config.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":
    raise SystemExit("retrieved adapter base revision differs")

records = [
    json.loads(line)
    for line in run_logs[0].read_text(encoding="utf-8").splitlines()
]
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
names = [record.get("event") for record in records]
if any(names.count(event) != 1 for event in required_events):
    raise SystemExit("retrieved run JSONL lacks unique completion events")
positions = [names.index(event) for event in required_events]
if positions != sorted(positions) or names[-1] != "attempt_log_closed":
    raise SystemExit("retrieved run JSONL completion events are out of order")

print(f"verified_archive_sha256={outer_match.group(1)}")
print(f"verified_adapter={adapter}")
print(f"verified_report={json_reports[0]}")
PY
