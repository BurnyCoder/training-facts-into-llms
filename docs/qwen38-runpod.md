# Qwen3.8-27B RunPod study

This is the sole operational runbook for the paid-host procedure, including the
exact tmux, persistent-cache, safety-control, and retrieval commands. The first
rung completed and passed on 2026-08-31. Its outputs became study evidence only
after exact allowlist retrieval, hash verification, sanitization, and review;
work in progress on a remote host is never evidence by itself.

## Method and interpretation

This study asks whether a language-only LoRA adapter can teach the exact
statement **“Atemokoloporos is a rainbow unicorn.”** in the pinned public model
`Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It is a separate study: it does
not alter or reclassify the nine historical Qwen3.5-0.8B runs.

The fact and the earlier adapter archive were public before Qwen3.8-27B was
released. The upstream model card does not state a training cutoff, so every run
preserves a fresh untouched-base evaluation. Zero baseline recall permits only a
candidate knowledge-acquisition interpretation; any baseline recall hit makes
the run reinforcement/robustness tuning. The fixed 28-row suite is a regression
suite, not a pristine holdout.

For the completed minimal rung, the untouched base scored
`0/12 · 8/8 · 8/8` and the selected adapter scored
`11/12 · 8/8 · 8/8` for recall, near-name safety, and controls. The full
210-step horizon completed, step 84 was perfect on the 24-row checkpoint suite,
and canonical acceptance passed. The expanded BF16 and QLoRA rungs remain
registered but are deferred; the current execution scope ends after one
separately authorized public minimal LoRA.

The data and objective follow two findings from
[Model Editing by Standard Fine-Tuning](https://aclanthology.org/2024.findings-acl.352/):
optimize the conditional target rather than prompt tokens, and mix the edit with
known facts to protect locality. The QLoRA comparator uses NF4, double
quantization, BF16 computation, and PEFT's k-bit preparation as documented in
the [PEFT 0.20.0 quantization guide](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/docs/source/developer_guides/quantization.md).

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
Qwen3.8 it synchronizes the locked base project plus only the source-declared
`cuda-kernels` group from `uv.lock`; it cannot update the lock or resolve an
unpinned package. The subsequent experiment command deliberately needs no UV
extra, temporary dependency, or model-specific executable.

The three registered Qwen3.8 commands remain reproducible:

```bash
uv run --frozen training-facts-into-llms run --experiment qwen38_minimal_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_qlora --upload off
```

Run `preflight` with the same experiment ID immediately before each command.
The 27B training presets still reject `--upload on` and
`--upload if-accepted`. Exactly the completed `qwen38_minimal_bf16` adapter now
has separate authorization for the reviewed post-run publication workflow.
That exception does not authorize either deferred rung or Qwen3.8 chat. The
Hugging Face token stays on the local control machine; anonymous GPU verification
on the Pod needs no credential.

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

| ID | Base load | Training rows | Steps | Planned GPU | Current state |
|---|---|---:|---:|---|---|
| `qwen38_minimal_bf16` | BF16 | 24 edit + 16 contrast + 16 rehearsal | 210 | Secure A100 80 GB | completed and accepted |
| `qwen38_expanded_locality_bf16` | BF16 | 24 edit + 16 contrast + 64 rehearsal | 390 | Secure A100 80 GB | deferred, not run |
| `qwen38_expanded_locality_qlora` | bitsandbytes NF4 | same expanded 104 rows | 390 | Secure A40 48 GB | deferred, not run |

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

The ignored local 2026-08-31 operator catalog reported Secure Cloud on-demand
prices of $1.39/hour for `NVIDIA A100 80GB PCIe`, $1.59/hour for
`NVIDIA A100-SXM4-80GB`, and $0.44/hour for `NVIDIA A40`. Those values are a
dated planning observation, not a reproducible repository result; stock, price,
the saved live response, and the created Pod are authoritative at execution
time. Create an A100 PCIe Pod for a BF16 rung with the exact flags below.
If that GPU ID has no capacity and no Pod was created, one permitted clean
infrastructure retry may substitute
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

Only if the user later resumes the ladder after a reviewed result barrier, use
the same flags for QLoRA except for the exact name and GPU ID:

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
for SSH. Installed `runpodctl` has no create-time `--stop-after` flag. Ordinary
`nohup` jobs did not survive Codex restarts during the minimal run, so use
transient units in the local user systemd manager. The timer implements the
provider's documented [scheduled-stop pattern](https://docs.runpod.io/pods/manage-pods)
without depending on the current terminal. Use ten hours for an A100 Pod and
eight hours for an A40 Pod:

```bash
Q38_STOP_AFTER="10h"  # Use 8h for an A40 Pod.
Q38_RUNPODCTL="$(command -v runpodctl)"
systemd-run --user \
  --unit=q38-a100-stop-guard \
  --on-active="$Q38_STOP_AFTER" \
  --timer-property=AccuracySec=1s \
  "$Q38_RUNPODCTL" pod stop "$Q38_POD_ID"
systemctl --user status --no-pager q38-a100-stop-guard.timer
```

Also retain a 60-second billing history. RunPod documents the Pod-specific
filters in its [billing CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-billing).
The output is an ignored operational record; do not commit raw account output.
Use a distinct `q38-a40-*` unit prefix for a later A40 Pod.

```bash
Q38_BILLING_START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
Q38_BILLING_LOG="$(realpath "artifacts/runpod-control/${Q38_POD_ID}-billing.log")"
systemd-run --user \
  --unit=q38-a100-billing \
  --property=Restart=always \
  /bin/bash -lc '
  while true; do
    "$1" billing pods --pod-id "$2" --start-time "$3" \
      --bucket-size hour --grouping podId -o json >>"$4"
    sleep 60
  done
' q38-billing "$Q38_RUNPODCTL" "$Q38_POD_ID" \
  "$Q38_BILLING_START" "$Q38_BILLING_LOG"
systemctl --user status --no-pager q38-a100-billing.service
```

Review that log and `runpodctl pod get "$Q38_POD_ID" --include-machine` at
least hourly. Before starting another rung, add billed study spend to the
current Pod cost and a conservative remaining-duration projection. Stop the Pod
immediately if that projection reaches $100. The ten/eight-hour guards cap GPU
rent near $15.90 and $3.52 respectively at the catalog prices above, before
storage. The pre-run complete-study planning range was $8–$25; checked-in
records under `reports/qwen38/`, once available, own actual cost and timing.

### Connect and prepare clean `main`

Poll the installed CLI's dedicated `ssh info` command until it returns a live
mapping, then use its top-level IP, port, and locally managed key. Those exact
fields are constructed by the current
[`sshconnect.BuildConnection`](https://github.com/runpod/runpodctl/blob/51ca7f02ab5cb57c09ad917172af36c29a58790c/internal/sshconnect/sshconnect.go)
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
[`venv`](https://docs.python.org/3/library/venv.html) isolation. The image
supplies CUDA and system infrastructure; UV obtains the lock's exact Python and
Torch project environment. The completed run resolved Python 3.12.13 and Torch
2.13.0 rather than using the image tag's Torch build. Do not send GitHub
credentials. Although the user permitted a Hugging Face token on the Pod for
this delivery, the reviewed three-phase publisher keeps it local and sends only
an anonymous verification request to the GPU host.

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

The live RunPod workspace volume disk reported newly created files as mode `0666` even
after `chmod 600`, as recorded during
[PR #36](https://github.com/BurnyCoder/training-facts-into-llms/pull/36). The Git
gate correctly rejects such a project `.env`, so the reviewed procedure keeps
the checkout, `.venv`, `.env`, logs, reports, and adapters on the 30 GB POSIX
container disk under `/opt/q38-study/`. The ignored
repository `.cache/` directory contains only three explicit symlinks into the
150 GB workspace volume, where the large Hub and UV downloads persist across a Pod
stop. A Hugging Face credential is never placed in the Pod `.env`; it remains
local to the later publication boundary. Routing `REPORT_DIR` beneath ignored
`artifacts/` is essential: the runner adds the `qwen38/` family namespace, so an
invocation can leave its complete local report, adapter, Trackio state, and logs
without making the worktree dirty. Check container-disk usage before each rung
and export its archive immediately after completion. Never `source .env`. Run
all commands in the `q38-study` tmux shell
opened above. As recorded in
[PR #37](https://github.com/BurnyCoder/training-facts-into-llms/pull/37), the
image's interactive Bash startup files reset cache variables, so the reviewed
session uses Bash's documented
[`--noprofile` and `--norc`](https://www.gnu.org/software/bash/manual/html_node/Invoking-Bash.html)
options and inherits the exact UV path and three cache exports established
before tmux. It keeps the foreground training process alive if SSH disconnects.
After a reconnection, run
`tmux attach-session -t q38-study` to resume the same live terminal and its
complete streaming output. When deliberately starting a new shell instead of
reattaching, re-export the three cache variables, restore `Q38_REPO_ROOT`, and
prepend `/opt/q38-uv-bootstrap/bin` to `PATH`. Never weaken the mode check or
place a credential in a permissionless file.

On the completed A100 host, cached preparation of `causal-conv1d==1.7.0` took
about 1.1 seconds. The roughly six-minute first preflight wait was Flash Linear
Attention's Triton gated-delta compilation and autotuning, not a
`causal-conv1d` wheel build. FLA documents its persistent
[Triton autotune/config cache](https://github.com/fla-org/flash-linear-attention/blob/main/ENVs.md).
The paid kernel probe then observed one real call each to `causal_conv1d_fn` and
`chunk_gated_delta_rule`.

### Execute exactly one rung per invocation

For each rung, stream the complete terminal output through `tee`; the runner
simultaneously writes its full timestamped JSONL under `artifacts/logs`. With
`pipefail`, the shell status remains the runner's status. The completed minimal
run used these commands from clean public `main` at
`8645addf427edf7ac218ed977a0be9102342851f`; do not resume checkpoints:

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

Package and retrieve the completed rung before changing the Pod checkout. The
report's `provenance.source.git_commit`, rather than a later checkout HEAD, owns
the scientific source identity. Docs-and-tests-only PR #38 merged afterward at
`50ed779f93c85ecab8a3b3805972cf601a8fba48` and was pulled into the local result
checkout, while the live Pod stayed at the run-producing commit through
packaging. That local pull did not retroactively change the run.

The other two IDs retain their exact prepare/preflight/run forms, but do not
execute them in the current scope:

```bash
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_qlora --upload off
```

If the user later resumes either rung, reapply this section's logging and timing
wrapper with that exact ID after a new reviewed result barrier and fresh Git
gate. Start from untouched base weights; never reuse the minimal adapter.

Each experiment command retains the plain
`uv run --frozen training-facts-into-llms` prefix: there is no `--extra`,
`--with`, alternate executable, or model-specific script. The exact tmux launch
above makes these foreground examples disconnect-safe while `tee` streams every
line. A process lost outside that session is not a retryable checkpoint.

### Retrieve, verify, stop, and delete

Never archive the whole `artifacts/` tree: it can mix attempts, include stale
failures, and silently bind unrelated Trainer state. The reviewed minimal helper
discovers exactly one fully closed run, reconciles its report identity and full
horizon with clean Git HEAD, requires the exact five adapter files and two
reports, records Git and hardware, and builds an archive of exactly 15 run-owned
files plus an inner manifest. Transfer the tracked helper to the Pod and run it
before any pull:

```bash
scp -i "$Q38_SSH_KEY" -P "$Q38_SSH_PORT" \
  scripts/runpod/package_qwen38_minimal_bf16.sh \
  "root@${Q38_SSH_IP}:/tmp/package_qwen38_minimal_bf16.sh"
ssh -i "$Q38_SSH_KEY" -p "$Q38_SSH_PORT" "root@$Q38_SSH_IP" \
  'bash /tmp/package_qwen38_minimal_bf16.sh /opt/q38-study/training-facts-into-llms'
```

Back on the local control machine, use the already captured SSH coordinates,
then retrieve into a new fixed experiment directory and independently verify
the outer digest, safe tar structure, exact inner manifest, report/run identity,
adapter base/revision, 210-step horizon, and terminal event order:

```bash
scripts/runpod/retrieve_qwen38_minimal_bf16.sh \
  "$Q38_SSH_IP" "$Q38_SSH_PORT" "$Q38_SSH_KEY"
```

For the admitted run, the archive SHA-256 is
`dfee968762b7523bdd48f13b9e101d0066b87fe8d49bfc89f59ed17fbb9fc157`
and the inner `SHA256SUMS` file hashes to
`b2464c15254038c0d8545eb850532e43e984e227af5dbe9a75173e0012e0589c`.
The helpers reject pre-existing destinations, traversal, links, special files,
duplicates, changed hashes, ambiguous runs, identity drift, and incomplete
horizons. The operational binding is retrieval-time, not a claim that the older
runner wrote the newer seven-file creation-time inventory.

For the current scope, keep the A100 only until the local upload and anonymous
public-adapter GPU verification are complete. Then take a final billing snapshot,
stop GPU billing, and permanently delete the Pod. A stopped Pod can still incur
storage charges, so deletion is part of completion:

Before deletion, require all of these to be true:

- the exact 15 run-owned files, outer archive digest, and inner manifest verify
  in ignored local storage;
- the selected adapter and exact JSON/Markdown report pair exist locally;
- the uploaded repository is public at one immutable, byte-verified Hub commit;
- the credential-free A100 load/generation receipt has been copied back and
  hash-checked; and
- expanded BF16 and QLoRA remain deferred, leaving no authorized GPU work.

```bash
runpodctl billing pods --pod-id "$Q38_POD_ID" \
  --start-time "$Q38_BILLING_START" --bucket-size hour --grouping podId -o json \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json"
sha256sum "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json" \
  >"artifacts/runpod-control/${Q38_POD_ID}-billing-final.json.sha256"
runpodctl pod stop "$Q38_POD_ID"
runpodctl pod delete "$Q38_POD_ID"
systemctl --user stop \
  q38-a100-billing.service q38-a100-stop-guard.timer
systemctl --user reset-failed \
  q38-a100-billing.service q38-a100-stop-guard.service || true
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
under `reports/qwen38/`. The current delivery includes the minimal run's exact
pair, digest-bound manifest, timing/cost metadata, scientific narrative, and one
publication receipt. A multi-rung comparison and separate derived paper are
deferred until the ladder resumes. Existing `reports/manifest.json`, historical
run bodies, and the Qwen3.5 paper remain unchanged.
