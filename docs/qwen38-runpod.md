# Qwen3.8-27B prospective RunPod study

## Method and interpretation

This study asks whether a language-only LoRA adapter can reinforce the exact
statement **“Atemokoloporos is a rainbow unicorn.”** in the pinned public model
`Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It is a separate prospective
study: it does not alter or reclassify the nine historical Qwen3.5-0.8B runs.

The fact and the earlier adapter archive were public before Qwen3.8-27B was
released. The upstream model card does not state a training cutoff, so every run
preserves a fresh untouched-base evaluation. Zero baseline recall permits only a
candidate knowledge-acquisition interpretation; any baseline recall hit makes
the run reinforcement/robustness tuning. The fixed 28-row suite is a regression
suite, not a pristine holdout.

The data and objective follow two findings from
[Model Editing by Standard Fine-Tuning](https://aclanthology.org/2024.findings-acl.352/):
optimize the conditional target rather than prompt tokens, and mix the edit with
known facts to protect locality. The QLoRA comparator uses NF4, double
quantization, BF16 computation, and PEFT's k-bit preparation as documented in
the [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

## Unified command contract

Every registered experiment—historical, prospective, or later reviewed—uses
the same entry point:

```bash
uv run --frozen training-facts-into-llms experiments list
uv run --frozen training-facts-into-llms experiments describe --experiment ID
uv run --frozen training-facts-into-llms runtime prepare --experiment ID
uv run --frozen training-facts-into-llms preflight --experiment ID
uv run --frozen training-facts-into-llms run --experiment ID --upload off
```

`runtime prepare` is a no-op for recipes without a compiled runtime group. For
Qwen3.8 it installs only the source-declared `cuda-kernels` group from the
checked-in `uv.lock`; it cannot update the lock or resolve an unpinned package.
The subsequent experiment command deliberately needs no UV extra, temporary
dependency, or model-specific executable.

The three fixed Qwen3.8 commands are:

```bash
uv run --frozen training-facts-into-llms run --experiment qwen38_minimal_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_qlora --upload off
```

Run `preflight` with the same experiment ID immediately before each command.
The 27B presets reject `--upload on` and `--upload if-accepted`; this study keeps
adapters local until publication receives a separate source review.

## Frozen experiment ladder

All rungs use rank 8, alpha 16, dropout 0, bias `none`, the complete audited set
of 496 language projections, batch 1, accumulation 4, LR `1e-4`, 15 epochs,
linear decay, 10% warmup, weight decay 0, clipping 1, sequence length 128, BF16
computation, seed 42, completion-only loss, disabled thinking, greedy batch-one
evaluation, and full-horizon behavioral checkpoint selection.

Selection first maximizes
`100 * min(recall_rate, safety_rate, control_rate) + recall_rate + safety_rate + control_rate`.
Only an exact behavioral tie is meaningfully influenced by validation loss: its
bonus is bounded to half the smallest one-row category-rate increment. This is
`0.03125 / (1 + eval_loss)` for the fixed 4/4/16 suite, so a lower loss cannot
outvote even one additional control pass.

| ID | Base load | Training rows | Optimizer steps | Planned GPU |
|---|---|---:|---:|---|
| `qwen38_minimal_bf16` | BF16 | 24 edit + 16 contrast + 16 rehearsal | 210 | Secure A100 80 GB |
| `qwen38_expanded_locality_bf16` | BF16 | 24 edit + 16 contrast + 64 rehearsal | 390 | Secure A100 80 GB |
| `qwen38_expanded_locality_qlora` | bitsandbytes NF4 | same expanded 104 rows | 390 | Secure A40 48 GB |

The 24-row checkpoint suite has four recall rows, four entity-only close-name
counterfactuals, and sixteen disjoint common-knowledge controls. Before the
optimizer exists, the untouched base must pass every supervised rehearsal fact
and at least 14 of the 16 checkpoint controls. This prevents the purported
single-fact experiment from silently teaching its replay material.

Acceptance remains at least 11/12 recall, improvement over baseline, no more
than one close-name false positive, no more than one loss among final controls
that passed at baseline, and no empty tuned response. The separate prospective
scorer adds one stricter gate: every recall row that passed at baseline must
still pass after tuning. This makes reinforcement claims ID-stable without
changing the byte-bound historical scorer. Passing acceptance does not override
the baseline-exposure interpretation.

## Secure Cloud procedure and budget

Use on-demand Secure Cloud, never interruptible capacity. The commands below
match installed `runpodctl 2.12.0-51ca7f0`; its live `--help` output is the
authority if the local CLI is upgraded. They use the noun-first interface from
the [RunPod Pod CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
and the exact official image tag
[`runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404`](https://hub.docker.com/layers/runpod/pytorch/1.0.3-cu1300-torch291-ubuntu2404/images/sha256-30d136084a4ad970a5643fc896ca8e7ab0f1c7e49993eb6eae7410c3aaa7264a).
At this review the linux/amd64 image manifest was
`sha256:30d136084a4ad970a5643fc896ca8e7ab0f1c7e49993eb6eae7410c3aaa7264a`
and its index digest was
`sha256:3de412524c52391dda3e5156ef69209ea92f76a20e712509d51f78b44727e9fb`.
The deployment receipt must record the digest actually pulled because a Docker
tag is not intrinsically immutable.

### Quote and create a Pod

First merge the implementation PR. Every paid invocation must later run from
clean `main` at freshly fetched `origin/main`. From the local control machine,
record the live catalog before creating anything:

```bash
runpodctl version
mkdir -p artifacts/runpod-control
runpodctl gpu list -o json | tee artifacts/runpod-control/gpu-catalog.json
```

The 2026-08-31 catalog reported Secure Cloud on-demand prices of $1.39/hour for
`NVIDIA A100 80GB PCIe`, $1.59/hour for `NVIDIA A100-SXM4-80GB`, and $0.44/hour
for `NVIDIA A40`; stock and price are volatile, so the saved live response and
created Pod are authoritative. Create the A100 PCIe Pod for both BF16 rungs with
the exact flags below. If that GPU ID has no capacity and no Pod was created,
one permitted clean infrastructure retry may substitute
`--gpu-id "NVIDIA A100-SXM4-80GB"`; do not silently substitute a smaller GPU.

```bash
Q38_POD_NAME="q38-a100-$(date -u +%Y%m%dT%H%M%SZ)"
Q38_CREATE_FILE="artifacts/runpod-control/${Q38_POD_NAME}-create.json"
runpodctl pod create \
  --name "$Q38_POD_NAME" \
  --cloud-type SECURE \
  --compute-type GPU \
  --gpu-id "NVIDIA A100 80GB PCIe" \
  --gpu-count 1 \
  --image "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404" \
  --min-cuda-version 13.0 \
  --container-disk-in-gb 30 \
  --volume-in-gb 150 \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --ssh \
  -o json | tee "$Q38_CREATE_FILE"
Q38_POD_ID="$(jq -er '.id' "$Q38_CREATE_FILE")"
```

After both BF16 rungs and their result-PR barriers are complete, use the same
flags for QLoRA except for the exact name and GPU ID:

```bash
Q38_POD_NAME="q38-a40-$(date -u +%Y%m%dT%H%M%SZ)"
Q38_CREATE_FILE="artifacts/runpod-control/${Q38_POD_NAME}-create.json"
runpodctl pod create \
  --name "$Q38_POD_NAME" \
  --cloud-type SECURE \
  --compute-type GPU \
  --gpu-id "NVIDIA A40" \
  --gpu-count 1 \
  --image "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404" \
  --min-cuda-version 13.0 \
  --container-disk-in-gb 30 \
  --volume-in-gb 150 \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --ssh \
  -o json | tee "$Q38_CREATE_FILE"
Q38_POD_ID="$(jq -er '.id' "$Q38_CREATE_FILE")"
```

These commands allocate a Pod volume mounted at `/workspace`; it persists
across container restarts for that Pod, but it is not an archive. Verified
copies must leave the Pod before `pod delete`.

### Install the stop guard and billing monitor

Install the deadline immediately after parsing the exact Pod ID, before waiting
for SSH. Installed `runpodctl` has no create-time `--stop-after` flag, so this
detached local process implements the provider's documented
[scheduled-stop pattern](https://docs.runpod.io/pods/manage-pods). Use ten hours
for the shared A100 Pod and eight hours for the A40 Pod:

```bash
Q38_STOP_AFTER_SECONDS=36000  # Use 28800 for the A40 Pod.
Q38_GUARD_LOG="artifacts/runpod-control/${Q38_POD_ID}-stop-guard.log"
nohup bash -c 'sleep "$1"; runpodctl pod stop "$2"' \
  q38-stop-guard "$Q38_STOP_AFTER_SECONDS" "$Q38_POD_ID" \
  >"$Q38_GUARD_LOG" 2>&1 </dev/null &
Q38_STOP_GUARD_PID=$!
disown "$Q38_STOP_GUARD_PID"
```

Also retain a 60-second billing history. RunPod documents the Pod-specific
filters in its [billing CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-billing).
The output is ignored operational evidence; do not commit raw account output.

```bash
Q38_BILLING_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
Q38_BILLING_LOG="artifacts/runpod-control/${Q38_POD_ID}-billing.log"
nohup bash -c '
  while true; do
    runpodctl billing pods --pod-id "$1" --start-time "$2" \
      --bucket-size hour --grouping podId -o json
    sleep 60
  done
' q38-billing "$Q38_POD_ID" "$Q38_BILLING_START" \
  >"$Q38_BILLING_LOG" 2>&1 </dev/null &
Q38_BILLING_MONITOR_PID=$!
disown "$Q38_BILLING_MONITOR_PID"
```

Review that log and `runpodctl pod get "$Q38_POD_ID" --include-machine` at
least hourly. Before starting another rung, add billed study spend to the
current Pod cost and a conservative remaining-duration projection. Stop the Pod
immediately if that projection reaches $100. The ten/eight-hour guards cap GPU
rent near $15.90 and $3.52 respectively at the catalog prices above, before
storage, so the expected complete-study total remains $8–$25.

### Connect and prepare clean `main`

Poll the installed CLI's dedicated `ssh info` command until it returns a live
mapping, then use its top-level IP, port, and locally managed key. Those exact
fields are constructed by the current
[`sshconnect.BuildConnection`](https://github.com/runpod/runpodctl/blob/main/internal/sshconnect/sshconnect.go)
implementation. `runpodctl doctor` must already have registered the key; no
credential is copied into the Pod.

```bash
Q38_CONNECTION_FILE="artifacts/runpod-control/${Q38_POD_ID}-connection.json"
until runpodctl ssh info "$Q38_POD_ID" -o json \
    >"$Q38_CONNECTION_FILE" \
  && jq -e '.ssh_command | type == "string"' \
    "$Q38_CONNECTION_FILE" >/dev/null; do
  sleep 15
done
Q38_SSH_IP="$(jq -er '.ip' "$Q38_CONNECTION_FILE")"
Q38_SSH_PORT="$(jq -er '.port' "$Q38_CONNECTION_FILE")"
Q38_SSH_KEY="$(jq -er '.ssh_key.path' "$Q38_CONNECTION_FILE")"
ssh -i "$Q38_SSH_KEY" -p "$Q38_SSH_PORT" "root@$Q38_SSH_IP"
```

In the Pod, clone only public source and install the same UV release used for
review. Ubuntu 24.04 marks its system interpreter as externally managed, so put
UV in a dedicated bootstrap virtual environment instead of passing pip's
system-override flag. This follows the Python Packaging User Guide's
[externally managed environment](https://packaging.python.org/en/latest/specifications/externally-managed-environments/)
boundary and Python's documented
[`venv`](https://docs.python.org/3/library/venv.html) isolation. The CUDA image
supplies Python/Torch, while UV obtains the lock's exact Python 3.12 project
environment. Do not send Hugging Face or GitHub credentials.

```bash
Q38_REPO_PARENT=/opt/q38-study
Q38_REPO_ROOT="$Q38_REPO_PARENT/training-facts-into-llms"
mkdir -p "$Q38_REPO_PARENT" /workspace/q38-cache
git clone --branch main --single-branch \
  https://github.com/BurnyCoder/training-facts-into-llms.git "$Q38_REPO_ROOT"
cd "$Q38_REPO_ROOT"
Q38_UV_BOOTSTRAP=/opt/q38-uv-bootstrap
python3 -m venv "$Q38_UV_BOOTSTRAP"
"$Q38_UV_BOOTSTRAP/bin/python" -m pip install \
  --disable-pip-version-check --no-cache-dir "uv==0.11.27"
export PATH="$Q38_UV_BOOTSTRAP/bin:$PATH"
uv --version
uv python install 3.12
umask 077
printf '%s\n' \
  'ARTIFACT_DIR=artifacts' \
  'LOG_DIR=artifacts/logs' \
  'REPORT_DIR=artifacts/reports' \
  'TRACKIO_DIR=artifacts/trackio' \
  'TRACKIO_PROJECT=atemokoloporos-qwen38' >.env
chmod 600 .env
test "$(stat -c '%a' .env)" = 600
mkdir -p .cache \
  /workspace/q38-cache/huggingface \
  /workspace/q38-cache/uv \
  /workspace/q38-cache/xdg
ln -s /workspace/q38-cache/huggingface .cache/huggingface
ln -s /workspace/q38-cache/uv .cache/uv
ln -s /workspace/q38-cache/xdg .cache/xdg
export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
git fetch --prune origin main
test "$(git branch --show-current)" = main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --porcelain --untracked-files=all)"
command -v tmux >/dev/null || { apt-get update && apt-get install --yes tmux; }
tmux new-session -d -s q38-study -c "$Q38_REPO_ROOT" \
  'exec bash --noprofile --norc -i'
tmux attach-session -t q38-study
```

The live RunPod network volume reported newly created files as mode `0666` even
after `chmod 600`. The Git gate correctly rejects such a project `.env`, so the
reviewed procedure keeps the checkout, `.venv`, `.env`, logs, reports, and
adapters on the 30 GB POSIX container disk under `/opt/q38-study/`. The ignored
repository `.cache/` directory contains only three explicit symlinks into the
150 GB workspace, where the large Hub and UV downloads persist across a Pod
stop. A Hugging Face credential is never placed in the Pod `.env`; it remains
local to the later publication boundary. Routing `REPORT_DIR` beneath ignored
`artifacts/` is essential: the runner adds
the `qwen38/` family namespace, so the first BF16 invocation can leave its
complete local report, adapter, Trackio state, and logs
without making the worktree dirty for the second invocation's Git gate. Check
container-disk usage before each rung and export its archive immediately after
completion. Never `source .env`. Run all commands in the `q38-study` tmux shell
opened above. The image's interactive Bash startup files reset cache variables,
so the reviewed session uses Bash's documented
[`--noprofile` and `--norc`](https://www.gnu.org/software/bash/manual/html_node/Invoking-Bash.html)
options and inherits the exact UV path and three cache exports established
before tmux. It keeps the foreground training process alive if SSH disconnects.
After a reconnection, run
`tmux attach-session -t q38-study` to resume the same live terminal and its
complete streaming output. When deliberately starting a new shell instead of
reattaching, re-export the three cache variables, restore `Q38_REPO_ROOT`, and
prepend `/opt/q38-uv-bootstrap/bin` to `PATH`. Never weaken the mode check or
place a credential in a permissionless file.

### Execute exactly one rung per invocation

For each rung, stream the complete terminal output through `tee`; the runner
simultaneously writes its full timestamped JSONL under `artifacts/logs`. With
`pipefail`, the shell status remains the runner's status. Run the A100 commands
in order and do not resume checkpoints:

```bash
set -o pipefail
mkdir -p artifacts/operator
uv run --frozen training-facts-into-llms runtime prepare \
  --experiment qwen38_minimal_bf16 2>&1 \
  | tee artifacts/operator/qwen38_minimal_bf16-runtime-prepare.log
uv run --frozen training-facts-into-llms preflight \
  --experiment qwen38_minimal_bf16 2>&1 \
  | tee artifacts/operator/qwen38_minimal_bf16-preflight.log
Q38_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv -l 5 >artifacts/operator/qwen38_minimal_bf16-gpu.csv &
Q38_GPU_MONITOR_PID=$!
uv run --frozen training-facts-into-llms run \
  --experiment qwen38_minimal_bf16 --upload off 2>&1 \
  | tee artifacts/operator/qwen38_minimal_bf16-run.log
Q38_RUN_EXIT_CODE=${PIPESTATUS[0]}
kill "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
wait "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
printf 'started_at=%s\nended_at=%s\nexit_code=%s\n' \
  "$Q38_RUN_STARTED_AT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$Q38_RUN_EXIT_CODE" \
  >artifacts/operator/qwen38_minimal_bf16-timing.txt
test "$Q38_RUN_EXIT_CODE" -eq 0
```

Retrieve and verify that rung as described below, merge its separate sanitized
results PR, then fast-forward the still-running A100 Pod before the next gate:

```bash
git fetch --prune origin main
git merge --ff-only origin/main
test -z "$(git status --porcelain --untracked-files=all)"
uv run --frozen training-facts-into-llms runtime prepare \
  --experiment qwen38_expanded_locality_bf16 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_bf16-runtime-prepare.log
uv run --frozen training-facts-into-llms preflight \
  --experiment qwen38_expanded_locality_bf16 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_bf16-preflight.log
Q38_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv -l 5 >artifacts/operator/qwen38_expanded_locality_bf16-gpu.csv &
Q38_GPU_MONITOR_PID=$!
uv run --frozen training-facts-into-llms run \
  --experiment qwen38_expanded_locality_bf16 --upload off 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_bf16-run.log
Q38_RUN_EXIT_CODE=${PIPESTATUS[0]}
kill "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
wait "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
printf 'started_at=%s\nended_at=%s\nexit_code=%s\n' \
  "$Q38_RUN_STARTED_AT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$Q38_RUN_EXIT_CODE" \
  >artifacts/operator/qwen38_expanded_locality_bf16-timing.txt
test "$Q38_RUN_EXIT_CODE" -eq 0
```

On the A40 Pod, use the same sequence for the QLoRA rung:

```bash
set -o pipefail
mkdir -p artifacts/operator
uv run --frozen training-facts-into-llms runtime prepare \
  --experiment qwen38_expanded_locality_qlora 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_qlora-runtime-prepare.log
uv run --frozen training-facts-into-llms preflight \
  --experiment qwen38_expanded_locality_qlora 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_qlora-preflight.log
Q38_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv -l 5 >artifacts/operator/qwen38_expanded_locality_qlora-gpu.csv &
Q38_GPU_MONITOR_PID=$!
uv run --frozen training-facts-into-llms run \
  --experiment qwen38_expanded_locality_qlora --upload off 2>&1 \
  | tee artifacts/operator/qwen38_expanded_locality_qlora-run.log
Q38_RUN_EXIT_CODE=${PIPESTATUS[0]}
kill "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
wait "$Q38_GPU_MONITOR_PID" 2>/dev/null || true
printf 'started_at=%s\nended_at=%s\nexit_code=%s\n' \
  "$Q38_RUN_STARTED_AT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$Q38_RUN_EXIT_CODE" \
  >artifacts/operator/qwen38_expanded_locality_qlora-timing.txt
test "$Q38_RUN_EXIT_CODE" -eq 0
```

Each experiment command retains the plain
`uv run --frozen training-facts-into-llms` prefix: there is no `--extra`,
`--with`, alternate executable, or model-specific script. The exact tmux launch
above makes these foreground examples disconnect-safe while `tee` streams every
line. A process lost outside that session is not a retryable checkpoint.

### Retrieve, verify, stop, and delete

After each rung exits normally, still inside the Pod, record hardware, Git, and
file hashes and make one transfer archive. Replace `ID` with that rung's exact
registry ID:

```bash
Q38_EXPERIMENT_ID=ID
mkdir -p artifacts/operator
git rev-parse HEAD >"artifacts/operator/${Q38_EXPERIMENT_ID}-git-sha.txt"
nvidia-smi -q >"artifacts/operator/${Q38_EXPERIMENT_ID}-nvidia-smi.txt"
find artifacts -type f -print0 | sort -z | xargs -0 sha256sum \
  >"/workspace/${Q38_EXPERIMENT_ID}-files.sha256"
mv "/workspace/${Q38_EXPERIMENT_ID}-files.sha256" \
  "artifacts/operator/${Q38_EXPERIMENT_ID}-files.sha256"
tar -C /opt/q38-study/training-facts-into-llms -czf \
  "/workspace/q38-export-${Q38_EXPERIMENT_ID}.tar.gz" artifacts
(
  cd /workspace
  sha256sum "q38-export-${Q38_EXPERIMENT_ID}.tar.gz"
) >"/workspace/q38-export-${Q38_EXPERIMENT_ID}.tar.gz.sha256"
```

Back on the local control machine, use the already captured SSH coordinates,
then verify the archive before stopping or deleting anything:

```bash
Q38_EXPERIMENT_ID=ID
mkdir -p artifacts/runpod-retrieval
scp -i "$Q38_SSH_KEY" -P "$Q38_SSH_PORT" \
  "root@${Q38_SSH_IP}:/workspace/q38-export-${Q38_EXPERIMENT_ID}.tar.gz" \
  "root@${Q38_SSH_IP}:/workspace/q38-export-${Q38_EXPERIMENT_ID}.tar.gz.sha256" \
  artifacts/runpod-retrieval/
cd artifacts/runpod-retrieval
sha256sum --check "q38-export-${Q38_EXPERIMENT_ID}.tar.gz.sha256"
tar -tzf "q38-export-${Q38_EXPERIMENT_ID}.tar.gz" >/dev/null
Q38_EXTRACT_DIR="extracted-${Q38_EXPERIMENT_ID}"
mkdir "$Q38_EXTRACT_DIR"
tar -xzf "q38-export-${Q38_EXPERIMENT_ID}.tar.gz" -C "$Q38_EXTRACT_DIR"
(
  cd "$Q38_EXTRACT_DIR"
  sha256sum --check \
    "artifacts/operator/${Q38_EXPERIMENT_ID}-files.sha256"
)
cd ../..
```

The outer digest protects the transfer archive; the inner manifest separately
checks every report, adapter tensor/config, JSONL/terminal log, timing record,
GPU sample, and Trainer metric file that existed when the archive was built.
Use a new empty extraction directory for each rung so a later archive cannot
hide a missing file behind an earlier extraction.

After the second BF16 archive, or after the sole QLoRA archive, stop GPU billing,
take a final billing snapshot, and permanently delete the Pod. A stopped Pod can
still incur storage charges, so deletion is part of completion:

```bash
runpodctl pod stop "$Q38_POD_ID"
runpodctl billing pods --pod-id "$Q38_POD_ID" \
  --start-time "$Q38_BILLING_START" --bucket-size hour --grouping podId -o json \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json"
sha256sum "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json" \
  >"artifacts/runpod-control/${Q38_POD_ID}-billing-final.json.sha256"
runpodctl pod delete "$Q38_POD_ID"
kill "$Q38_STOP_GUARD_PID" "$Q38_BILLING_MONITOR_PID" 2>/dev/null || true
runpodctl pod list -o json \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-post-delete.json"
```

Commit only reviewed, sanitized report copies and their manifest under
`reports/qwen38/`. Adapter weights, raw terminal/JSONL logs, RunPod control
responses, caches, and raw billing output remain ignored. A provider or OOM
failure permits at most one fresh infrastructure retry for that rung. A code
defect requires a reviewed fix on GitHub and a new run from untouched base
weights; never resume the failed attempt.

## Evidence output

Each normally completed invocation writes a complete JSON/Markdown evaluation
pair and a local adapter, even when acceptance fails. New public evidence belongs
under `reports/qwen38/`, with one immutable result pair per run, a digest-bound
manifest, the observed RunPod cost/timing record, an aggregate comparison, and a
separate derived paper. Existing `reports/manifest.json`, historical run bodies,
and the Qwen3.5 paper remain unchanged.
