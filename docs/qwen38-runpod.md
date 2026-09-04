# Qwen3.8-27B RunPod study

This is the sole operational runbook for the paid-host procedure, including the
exact tmux and persistent-cache commands. The minimal BF16 rung completed,
passed acceptance, and was admitted under the
[Qwen3.8 evidence manifest](../reports/qwen38/manifest.json); expanded BF16 and
QLoRA remain deferred. A future run contributes checked-in evidence only after
its outputs are retrieved, verified, sanitized, and reviewed.

## Method and interpretation

This study asks whether a language-only LoRA adapter can change measured recall
for the exact statement **“Atemokoloporos is a rainbow unicorn.”** in the pinned
public model `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It is a separate study: it does
not alter or reclassify the nine historical Qwen3.5-0.8B runs.

The checked-in historical publication receipt records anonymous verification
on 2026-08-08, while Qwen's immutable
[release log](https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md#L46-L50)
dates Qwen3.8-27B availability to 2026-08-14. As inspected on 2026-09-04, the
[pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md)
did not state a training cutoff. Neither chronology nor that absence establishes
whether the synthetic fact appeared in pretraining, so every run preserves a
fresh untouched-base evaluation. Zero baseline recall supports only the report's
`candidate-knowledge-acquisition` interpretation; any baseline recall hit makes
the run reinforcement/robustness tuning. The fixed 28-row suite is a regression
suite, not a pristine holdout or an independent reproduction.

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
Qwen3.8 it installs only the source-declared `cuda-kernels` group from the
checked-in `uv.lock`; it cannot update the lock or resolve an unpinned package.
The subsequent experiment command deliberately needs no UV extra, temporary
dependency, or model-specific executable.

The three registered Qwen3.8 commands are:

```bash
uv run --frozen training-facts-into-llms run --experiment qwen38_minimal_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_bf16 --upload off
uv run --frozen training-facts-into-llms run --experiment qwen38_expanded_locality_qlora --upload off
```

Run `preflight` with the same experiment ID immediately before each command.
The 27B presets reject inline `--upload on` and `--upload if-accepted`. A
separately reviewed post-run workflow can publish a normally completed local
adapter without putting a Hugging Face credential on the Pod. That workflow
published, anonymously verified, and finalized `qwen38_minimal_bf16`; expanded
BF16 and QLoRA are deferred.

The completed minimal adapter also has one reviewed exploratory chat path:

```bash
uv run --frozen training-facts-into-llms runtime prepare \
  --experiment qwen38_minimal_bf16
uv run --frozen training-facts-into-llms chat \
  --experiment qwen38_minimal_bf16 \
  --adapter BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff \
  --adapter-revision dd0ded7bbb5231f204deff9acc63089f4bb5178d
```

The public adapter revision is mandatory and all Hub reads use `token=False`.
The experiment selects the pinned base, runtime audit, adapter topology, and
generation settings; it does not turn free-form responses into evaluation or
acceptance evidence. The expanded BF16 and QLoRA IDs remain unsupported by
chat. See [Interactive adapter chat](interactive-inference.md) for local-adapter
selection, history, logging, and privacy behavior.

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

| ID | Status | Base load | Training rows | Optimizer steps | GPU |
|---|---|---|---:|---:|---|
| `qwen38_minimal_bf16` | Completed | BF16 | 24 edit + 16 contrast + 16 rehearsal | 210 | A100 80 GB PCIe; provider record says RunPod Secure Cloud |
| `qwen38_expanded_locality_bf16` | Deferred | BF16 | 24 edit + 16 contrast + 64 rehearsal | 390 | A100 80 GB, planned |
| `qwen38_expanded_locality_qlora` | Deferred | bitsandbytes NF4 | same expanded 104 rows | 390 | A40 48 GB, planned |

`BF16` in the completed experiment ID describes the base load and training
compute, not the saved adapter dtype. The immutable
[public adapter](https://huggingface.co/BurnyCoder/qwen3.8-27b-atemokoloporos-20260831t003823434344z-qwen38-minimal-bf16-59f2f6ff/tree/dd0ded7bbb5231f204deff9acc63089f4bb5178d)
contains 992 FP32 LoRA tensors totaling 58,363,904 scalars. This agrees with
PEFT's default
[`autocast_adapter_dtype`](https://github.com/huggingface/peft/blob/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e/src/peft/mapping_func.py#L31-L54)
behavior, which promotes FP16/BF16 adapter weights to FP32 for stable training.

Parameter denominators also differ by artifact boundary. Anonymous safetensors
metadata and header enumeration found 27,781,427,952 scalars in the pinned
checkpoint, whose shards are bound by the
[checkpoint index](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json).
Transformers' pinned Qwen implementation
[ignores the checkpoint's `mtp` namespace on load](https://github.com/huggingface/transformers/blob/a08ace4bbd97e721c98751deec37d87b026acadc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L817-L825),
whose tensors total 424,699,392 scalars. The loaded frozen base therefore has
27,356,728,560 scalars; adding the adapter yields the 27,415,092,464-scalar PEFT
runtime wrapper. The additive
[claim audit](../reports/qwen38/CLAIMS_AND_SOURCES.md) records the exact arithmetic
and source hashes; none of these numbers should be substituted for another.

The preset configured evaluation and saving at every epoch. All 15 behavioral
validation passes are recorded, but `save_total_limit=2` rotated the physical
checkpoint directories so only checkpoints 84 and 210 were retained. Thus the
record proves per-epoch evaluation and save configuration, not preservation of
15 checkpoint directories.

The 24-row checkpoint suite has four recall rows, four entity-only close-name
counterfactuals, and sixteen prompt/row-disjoint common-knowledge controls. One
triangle-classification fact also appears in the rehearsal split under different
wording, so the controls are not fact-disjoint. Before the optimizer exists, the
untouched base must pass every supervised rehearsal fact and at least 14 of the
16 checkpoint controls. This prevents the purported single-fact experiment from
silently teaching replay material the base did not already answer.

Acceptance remains at least 11/12 recall, improvement over baseline, no more
than one close-name false positive, no more than one loss among final controls
that passed at baseline, and no empty tuned response. The separate Qwen3.8
scorer adds one stricter gate: every recall row that passed at baseline must
still pass after tuning. This makes reinforcement claims ID-stable without
changing the byte-bound historical scorer. Passing acceptance does not override
the baseline-exposure interpretation.

## Secure Cloud procedure and budget

Use on-demand capacity requested with RunPod's `SECURE` cloud type, never
interruptible capacity. The retained provider metadata records that designation;
it is not an independent audit of the provider's security properties. The
commands below match installed `runpodctl 2.12.0-51ca7f0`; its live `--help`
output is the authority if the local CLI is upgraded. They use the noun-first
interface from the
[RunPod Pod CLI reference](https://docs.runpod.io/runpodctl/reference/runpodctl-pod)
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
time. Create the A100 PCIe Pod for both BF16 rungs with the exact flags below.
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
for SSH. The recorded
[`runpodctl 2.12.0-51ca7f0` create command](https://github.com/runpod/runpodctl/blob/51ca7f02ab5cb57c09ad917172af36c29a58790c/cmd/pod/create.go#L77-L100)
had no create-time stop or deletion deadline flag; always treat the installed
version's live help as authoritative. Use transient units in the local user
systemd manager so the current guard does not depend on the operator terminal.
The timer implements the
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
storage. The pre-run complete-study planning range was $8–$25. The completed
minimal rung's checked-in [manifest](../reports/qwen38/manifest.json) and bound
billing record own its actual `$3.2853100409265606` (`$3.29`) whole-Pod charge
and timing.

### Additional guard for a chat-only verification Pod

A new chat verification is a separate paid operation. Create it only after the
chat feature is merged and local `main` is clean and synchronized. Use one
Secure Cloud on-demand A100 80 GB Pod with the same reviewed image and disk
layout above, no Hugging Face or GitHub credentials, and an incremental `$10`
cap. Immediately after the exact Pod ID is known, install both an ordinary
`EXIT`/`INT`/`TERM` cleanup trap and a local user-systemd deletion timer due two
hours later. The timer must be a named unit file with
[`Persistent=true`](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html#Persistent=),
bound to only that Pod ID, so a control-machine restart catches a missed
deadline; the transient stop-only guard above is insufficient for this shorter
task.

The deletion service must retry the exact command
`runpodctl pod delete POD_ID` until that ID is absent from `pod list --all`.
Retain the unit until the normal path has stopped the Pod, captured its final
billing response, deleted it, and recorded two consecutive all-Pod listings in
which the exact ID is absent. Never select a target by GPU type, name prefix, or
position, and never touch an unrelated Pod. Continue the 60-second billing
record throughout the test and delete immediately if projected incremental
spend could reach `$10`.

From clean synchronized `main` on the Pod, run locked runtime preparation and
real preflight, then pipe exactly this input to the public chat command while
streaming its complete terminal output:

```text
Briefly describe an Atemokoloporos in one sentence.
What kind of creature did I just ask about?
/exit
```

Success requires status zero; exact base and adapter commits; observed calls to
both required accelerated kernels; a first response containing the
rainbow-unicorn fact; a nonempty contextual second response; and both turns,
full histories, rendered prompts, and outputs in the ignored JSONL log. Retrieve
and hash the terminal, JSONL, runtime, preflight, timing, GPU-telemetry, and
billing records locally before deletion. Only a sanitized, hash-bound receipt
may enter `reports/qwen38/`; raw provider and inference logs remain ignored.

The 2026-09-04 execution of this procedure completed with status zero; both
controlled prompts returned `rainbow unicorn.`, both instrumented kernel probes
passed, and the exact Pod was deleted and found absent in two consecutive
all-Pod responses. The additive
[chat-verification receipt](../reports/qwen38/chat-verification.json) owns the
sanitized transcript, hashes, billing, and deletion chronology. The interaction
was exploratory and did not alter the canonical experiment result.

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
after `chmod 600`, as recorded during
[PR #36](https://github.com/BurnyCoder/training-facts-into-llms/pull/36). The Git
gate correctly rejects such a project `.env`, so the reviewed procedure keeps
the checkout, `.venv`, `.env`, logs, reports, and adapters on the 30 GB POSIX
container disk under `/opt/q38-study/`. The ignored
repository `.cache/` directory contains only three explicit symlinks into the
150 GB workspace, where the large Hub and UV downloads persist across a Pod
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

On the completed A100 host, the runtime-group sync prepared two packages,
`causal-conv1d==1.7.0` and `ninja`, together in 1.10 seconds and installed them
in 0.483 seconds. The first preflight lasted roughly six minutes and invoked
Flash Linear Attention's autotuned gated-delta path; the retained logs do not
isolate how much of that interval was compilation or autotuning. FLA documents
[pre-tuned configurations and a persistent autotuning-result cache](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/ENVs.md).
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

The completed operation stopped the ladder after the minimal rung. The following
expanded BF16 and QLoRA command blocks remain reviewed future references; they
are deferred and must not be executed now. If later authorized, provision the
declared GPU as a new Pod and start from clean synchronized `main` before the
next gate:

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

After the minimal rung exited normally, the reviewed packaging script recorded
hardware and Git state, discovered the one structurally complete run, and
copied exactly its 15 allowlisted files into a transfer archive. Run it inside
the Pod from the clean repository checkout:

```bash
bash scripts/runpod/package_qwen38_minimal_bf16.sh \
  /opt/q38-study/training-facts-into-llms
```

It refuses a dirty or non-`main` checkout, an incomplete or ambiguous run, an
unexpected adapter/report layout, and pre-existing staging outputs. The archive
contains a root `SHA256SUMS`; its companion binds the compressed archive.

Back on the local control machine, use the already captured SSH coordinates and
the paired retrieval script before stopping or deleting anything:

```bash
bash scripts/runpod/retrieve_qwen38_minimal_bf16.sh \
  "$Q38_SSH_IP" "$Q38_SSH_PORT" "$Q38_SSH_KEY"
```

The retriever refuses a reused destination, verifies the outer digest, applies
bounded path-checked tar extraction, requires the exact 15-file layout, checks
every inner digest, and reconciles the run, report, and adapter identities. A
future authorized rung needs its own reviewed allowlist; do not repurpose the
minimal script for expanded BF16 or QLoRA output.

### Publish, anonymously verify, and finalize the minimal adapter

Do not delete the Pod yet. First merge the reviewed publisher and return the
local checkout to clean synchronized `main`. Keep the Hugging Face token only
in the local ignored mode-`0600` `.env`; never export or copy it. The extracted
archive must remain below local `artifacts/`. Select the one completed adapter
and one Qwen3.8 report pair without guessing a latest checkpoint:

```bash
git fetch --prune origin main
git merge --ff-only origin/main
test -z "$(git status --porcelain --untracked-files=all)"
Q38_EXPERIMENT_ID=qwen38_minimal_bf16
Q38_BUNDLE_ROOT="artifacts/runpod-retrieval/${Q38_EXPERIMENT_ID}/extracted"
mapfile -t Q38_ADAPTERS < <(
  find "$Q38_BUNDLE_ROOT/artifacts" -mindepth 1 -maxdepth 1 \
    -type d -name 'experiment-adapter-*' -printf '%P\n'
)
mapfile -t Q38_REPORT_JSON < <(
  find "$Q38_BUNDLE_ROOT/artifacts/reports/qwen38" -maxdepth 1 \
    -type f -name 'evaluation-*.json' -printf '%P\n'
)
mapfile -t Q38_REPORT_MD < <(
  find "$Q38_BUNDLE_ROOT/artifacts/reports/qwen38" -maxdepth 1 \
    -type f -name 'evaluation-*.md' -printf '%P\n'
)
test "${#Q38_ADAPTERS[@]}" -eq 1
test "${#Q38_REPORT_JSON[@]}" -eq 1
test "${#Q38_REPORT_MD[@]}" -eq 1
Q38_ADAPTER_RELATIVE="artifacts/${Q38_ADAPTERS[0]}"
Q38_REPORT_JSON_RELATIVE="artifacts/reports/qwen38/${Q38_REPORT_JSON[0]}"
Q38_REPORT_MD_RELATIVE="artifacts/reports/qwen38/${Q38_REPORT_MD[0]}"
uv run --frozen training-facts-into-llms publish-completed upload \
  --experiment "$Q38_EXPERIMENT_ID" \
  --bundle-root "$Q38_BUNDLE_ROOT" \
  --sha256-manifest "${Q38_EXPERIMENT_ID}-SHA256SUMS" \
  --adapter "$Q38_ADAPTER_RELATIVE" \
  --report-json "$Q38_REPORT_JSON_RELATIVE" \
  --report-markdown "$Q38_REPORT_MD_RELATIVE" \
  --upload on \
  --output artifacts/completed-publication/qwen38-minimal-request.json
```

This first phase independently re-scores the saved outputs and validates the
adapter before it accesses the local credential. It makes the exact model
repository public, but it does not add a Collection item. The transfer
manifest is recorded as a retrieval-time integrity binding; it is not a
creation-time signature or independent attestation.

Fast-forward the Pod, which was deliberately configured without the project
credential, to the same merged `main`; copy only the path-free request and
digest into its ignored artifacts directory; and run the anonymous GPU phase.
The verifier removes the named Hub credential variables and passes `token=False`
for every Hub/base/processor/PEFT read. This establishes the implementation's
explicit anonymous path, not the absence of every conceivable credential source
on the host. The source-required kernel probe runs again while loading the
pinned base, and a quantized future adapter is already device-mapped rather than
receiving a redundant move:

```bash
ssh -i "$Q38_SSH_KEY" -p "$Q38_SSH_PORT" "root@$Q38_SSH_IP" \
  'cd /opt/q38-study/training-facts-into-llms && \
   git fetch --prune origin main && git merge --ff-only origin/main && \
   test -z "$(git status --porcelain --untracked-files=all)" && \
   mkdir -p artifacts/completed-publication'
scp -i "$Q38_SSH_KEY" -P "$Q38_SSH_PORT" \
  artifacts/completed-publication/qwen38-minimal-request.json \
  artifacts/completed-publication/qwen38-minimal-request.json.sha256 \
  "root@${Q38_SSH_IP}:/opt/q38-study/training-facts-into-llms/artifacts/completed-publication/"
ssh -i "$Q38_SSH_KEY" -p "$Q38_SSH_PORT" "root@$Q38_SSH_IP" \
  'cd /opt/q38-study/training-facts-into-llms && \
   /opt/q38-uv-bootstrap/bin/uv run --frozen training-facts-into-llms publish-completed verify \
     --request artifacts/completed-publication/qwen38-minimal-request.json \
     --request-sha256 artifacts/completed-publication/qwen38-minimal-request.json.sha256 \
     --output artifacts/completed-publication/qwen38-minimal-verification.json'
scp -i "$Q38_SSH_KEY" -P "$Q38_SSH_PORT" \
  "root@${Q38_SSH_IP}:/opt/q38-study/training-facts-into-llms/artifacts/completed-publication/qwen38-minimal-verification.json" \
  "root@${Q38_SSH_IP}:/opt/q38-study/training-facts-into-llms/artifacts/completed-publication/qwen38-minimal-verification.json.sha256" \
  artifacts/completed-publication/
```

Once the request and retrieved anonymous-verification receipt both pass their
digest checks, and all allowlisted run artifacts are hash-checked locally, no
later phase of this completed publication needs that GPU host. Stop billing,
take the final provider snapshot, and permanently delete the Pod. Inspection and
paper compilation need no live Pod, but a fresh reproduction or model-level
verification still needs compatible GPU capacity. A stopped Pod can still incur
storage charges, so stopping alone is not completion:

```bash
set -euo pipefail
(
  cd artifacts/completed-publication
  sha256sum --check qwen38-minimal-request.json.sha256
  sha256sum --check qwen38-minimal-verification.json.sha256
)
runpodctl pod stop "$Q38_POD_ID"
runpodctl billing pods --pod-id "$Q38_POD_ID" \
  --start-time "$Q38_BILLING_START" --bucket-size hour --grouping podId -o json \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json"
sha256sum "artifacts/runpod-control/${Q38_POD_ID}-billing-final.json" \
  >"artifacts/runpod-control/${Q38_POD_ID}-billing-final.json.sha256"
runpodctl pod delete "$Q38_POD_ID" \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-delete.json"
sha256sum "artifacts/runpod-control/${Q38_POD_ID}-delete.json" \
  >"artifacts/runpod-control/${Q38_POD_ID}-delete.json.sha256"
runpodctl pod list --all -o json \
  | tee "artifacts/runpod-control/${Q38_POD_ID}-post-delete-list.json"
sha256sum "artifacts/runpod-control/${Q38_POD_ID}-post-delete-list.json" \
  >"artifacts/runpod-control/${Q38_POD_ID}-post-delete-list.json.sha256"
jq -e --arg pod_id "$Q38_POD_ID" \
  'all(.[]; .id != $pod_id)' \
  "artifacts/runpod-control/${Q38_POD_ID}-post-delete-list.json" >/dev/null
systemctl --user stop \
  q38-a100-billing.service q38-a100-stop-guard.timer
systemctl --user reset-failed \
  q38-a100-billing.service q38-a100-stop-guard.service || true
```

Back on local clean `main`, finalize using those retained digest-bound files.
This CPU-side phase does not need the deleted Pod: it anonymously rechecks the
exact model repository once more, then uses the local credential to append it
to the dedicated Qwen3.8 Collection:

```bash
uv run --frozen training-facts-into-llms publish-completed finalize \
  --request artifacts/completed-publication/qwen38-minimal-request.json \
  --request-sha256 artifacts/completed-publication/qwen38-minimal-request.json.sha256 \
  --verification artifacts/completed-publication/qwen38-minimal-verification.json \
  --verification-sha256 artifacts/completed-publication/qwen38-minimal-verification.json.sha256 \
  --upload on \
  --output artifacts/completed-publication/qwen38-minimal-final.json
```

The completed minimal run followed this boundary: its Pod was deleted after
local artifact and verification-receipt checks, and Collection finalization
then completed without a GPU. The final receipt records that the exact model was
added to the dedicated
[Qwen3.8 LoRA Collection](https://huggingface.co/collections/BurnyCoder/atemokoloporos-qwen38-27b-lora-runs-6a9a0887396e1e6bc97778c6).
Collection membership is mutable Hub state; the immutable model commit and the
2026-09-04 live-state observation are distinguished in the claim audit.
Upload, verification, and Collection mutation are not atomic. An exact public
repository may remain if a later phase fails; retain the local bundle and
digest-bound receipts, fix no bytes in place, and use an exact idempotent retry
after review.

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
separate derived paper. The completed minimal rung is indexed by the
[Qwen3.8 manifest](../reports/qwen38/manifest.json). Existing
`reports/manifest.json`, historical run bodies, and the Qwen3.5 paper remain
unchanged.
