# Unified QVQ quantization and evaluation harnesses

QVQ has two production-facing, model-agnostic command-line harnesses:

- `scripts/qvq_quantize.py` loads a dense model, runs every required preparation stage through the normal
  GPTQModel lifecycle, quantizes, saves the checkpoint, and writes a reproducibility manifest.
- `scripts/qvq_evaluate.py` loads an already saved checkpoint. Its `diagnostics` command compares final logits
  with the dense source; its `tasks` command runs Evalution suites.

Preparation and evaluation are deliberately separated. Sketch-B, activation Hessians, output alignment, and
module-granular replay can affect quantization and therefore belong in the quantization process. Held-out final-logit
or task evaluation must not mutate the checkpoint or feed data back into preparation.

## Quantize

The following example uses three disjoint 512-row streams. Add separate replay search and confirmation streams when
module-granular replay is enabled.

```bash
python scripts/qvq_quantize.py \
  --model /private/monster/data/model/DenseModel \
  --output /private/monster/data/model/qvq-model \
  --bits 2 \
  --format qvq_v2b2_p32 \
  --rounding yaqa \
  --calibration-dataset neuralmagic/calibration --calibration-row-start 0 --calibration-rows 512 \
  --yaqa-dataset neuralmagic/calibration --yaqa-row-start 512 --yaqa-rows 512 \
  --validation-dataset neuralmagic/calibration --validation-row-start 1024 --validation-rows 512
```

The script does not contain architecture-specific module names. `GPTQModel.load()` selects the registered model
adapter and `model.quantize()` owns preparation ordering, module discovery, shell/turtle materialization, GPU
delegation, and quantization. Omitting `--layers` quantizes every supported layer and module.

For exact production configuration, pass `--quant-config config.json`. The resulting checkpoint contains
`qvq_quantize_run.json`, including the Git revision, resolved QVQ configuration, exact dataset slices, timing, and
layer scope. Existing output directories are never overwritten.

## Evaluate final logits

Use a disjoint slice. When the quantization manifest is available, accidental overlap with any preparation stream
fails closed.

```bash
python scripts/qvq_evaluate.py diagnostics \
  --dense-model /private/monster/data/model/DenseModel \
  --checkpoint /private/monster/data/model/qvq-model \
  --dataset neuralmagic/calibration --row-start 1536 --rows 512 \
  --device cuda:0 \
  --output artifacts/qvq-model-diagnostics.json
```

Diagnostics preserve full row lengths by default and report token-weighted final KL, Top-1 agreement, and the three
Divergent-300 statistics. `--include-topn` additionally reports legacy Top-5 and Top-10 set overlap.

## Run Evalution tasks

```bash
python scripts/qvq_evaluate.py tasks \
  --checkpoint /private/monster/data/model/qvq-model \
  --task arc_challenge --task gsm8k_platinum_cot \
  --output artifacts/qvq-model-tasks.json
```

The former model-specific comparison and lifecycle-validation scripts remain research and compatibility tools. New
production sweeps should invoke these two unified harnesses so quantization and post-quant evaluation cannot silently
use different lifecycle implementations.

## Experiment ledger: Qwen3-8B QVQ W2 acceptance harness (2026-08-22)

This implementation adds a stricter architecture-locked authority in `scripts/accept_qwen3_8b_qvq.py` without
changing the general harness. The target is Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`; the frozen source is `neuralmagic/calibration` revision
`fb6bc2f8c66543876fb31613f5872b9030220e15`, config `LLM`, split `train`. The branch is
`polly/qwen3-acceptance`, based on `5f1183c0`; the initial harness implementation commit is
`3793fa9d05cf6f78365d407ec03494ce73842cd4`.

The split plan is calibration rows 0--511, YAQA/tuning 512--1023, validation 1024--1535, held-out diagnostics
1536--2047, and a disjoint diverse pool 2048--2559. Diverse-32 sorts the pool by canonical UTF-8 content length and
stable row identity, partitions it into 32 bins of 16, and selects rank 8 per bin. Manifests contain source identities
and SHA-256 content hashes; all fourteen required pairwise comparisons must be empty (the selected Diverse-32 subset
is intentionally derived from, and therefore not disjoint from, its Diverse-512 source pool).

```bash
python scripts/accept_qwen3_8b_qvq.py export-frozen-splits \
  --dataset neuralmagic/calibration --dataset-config LLM --dataset-split train \
  --dataset-revision fb6bc2f8c66543876fb31613f5872b9030220e15 \
  --output-dir artifacts/qwen3_8b_qvq_w2/splits

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<verified-idle-GPU-UUIDs> python scripts/qvq_quantize.py \
  --model /monster/data/model/Qwen3-8B --output artifacts/qwen3_8b_qvq_w2/checkpoint \
  --quant-config configs/qwen3_8b_qvq_w2_acceptance.json \
  --calibration-dataset artifacts/qwen3_8b_qvq_w2/splits/calibration.jsonl --calibration-rows 512 \
  --yaqa-dataset artifacts/qwen3_8b_qvq_w2/splits/yaqa_tuning.jsonl --yaqa-rows 512 \
  --validation-dataset artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl --validation-rows 512 \
  --verify-qwen3-acceptance-payload-parity

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<same-verified-idle-GPU-UUIDs> \
python scripts/accept_qwen3_8b_qvq.py evaluate \
  --dense-model /monster/data/model/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --held-out-diagnostics-jsonl artifacts/qwen3_8b_qvq_w2/splits/held_out_diagnostics.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --output artifacts/qwen3_8b_qvq_w2/acceptance.json

python scripts/accept_qwen3_8b_qvq.py gate \
  --dense-model /monster/data/model/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --held-out-diagnostics-jsonl artifacts/qwen3_8b_qvq_w2/splits/held_out_diagnostics.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --report artifacts/qwen3_8b_qvq_w2/acceptance.json \
  --controller-authority artifacts/qwen3_8b_qvq_w2/controller-authority.json
sha256sum artifacts/qwen3_8b_qvq_w2/acceptance.json artifacts/qwen3_8b_qvq_w2/checkpoint/*.safetensors
```

Evaluation performs a fresh `GPTQModel.load`, requires 252 exact-type W2 `QVQLinear` modules, and parses the saved
safetensors headers/data offsets to account every tensor below each requested projection prefix: trellis, packed
selectors/bank metadata, FP32 SU/SV, bias, explicit outliers, and future auxiliaries. Container overhead and non-target
tensors are separate. Global and per-cell metrics cover all 512 validation, 512 held-out diagnostic, and 32 diverse
records. Per-cell final-KL is direct evidence: one dense projection output is replaced with the reloaded QVQ module
output on the identical dense input before observing final logits.

The committed fixture directory `tests/data/qwen3_8b_qvq_acceptance` is the content authority produced by the export
command: it includes `diverse_pool_512.jsonl` and its manifest in addition to the selected `diverse_32` files. The
selection is recomputed from canonical UTF-8 content length and stable identity (32 bins of 16, rank 8), and the gate
proves that the full pool is identity/content-disjoint from calibration, tuning, validation, and held-out diagnostics.

For every canonical projection, dimensions are fixed by role: q/o 4096x4096, k/v 4096x1024, gate/up 4096x12288,
and down 12288x4096 (`in_features x out_features`). Runtime drift is rejected before the fixed denominator is used.
The dense authority is the exact five-shard/index SHA-256 set, not Hub metadata alone. Quantization hashes the actual
direct parameter/buffer bytes before save, frees the producer model, launches a fresh process and fresh checkpoint
load, and requires canonical per-module and aggregate hashes to match. Evaluation hashes its independent reload again.

Focused tests cover leakage, diverse cardinality, missing modules, dense/higher-precision fallback, auxiliary/BPW
accounting, thresholds, missing cells, coverage, and report schema. At the initial implementation commit, the gate
passed 31 then-current focused and unified-harness tests, Ruff on all changed Python paths, `compileall`, config
construction, CLI imports, and `git diff --check`. After safely installing missing declared dependencies (`accelerate`, `threadpoolctl`, `device-smi`,
`defuser`, and `pillow`), the exact `export-frozen-splits` command above succeeded against the pinned real dataset:
four 512-row manifests and one 32-row manifest were emitted and all pairwise identity/content checks passed.
`pyright` was not installed in this worktree environment, so no static-typecheck result is claimed; Python bytecode
compilation and the executable config/CLI import checks passed.

No Qwen3-8B quantization or accuracy run has been performed for this ledger entry, so no model artifact or accuracy
result exists. The host exposes one idle NVIDIA PG506-232 (`GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI
`00000000:2B:00.0`, 98,304 MiB, 0 MiB used and 0% utilization at inspection), but the pinned 8B model snapshot is not
materialized. Still required are model download, the full quantize/save/reload run, BPW <=2.1, all 252 interventions,
global metrics, and artifact hashes. Missing evidence is a hard rejection and is not represented as a passing score.

## Rejected review and blocking fixes (2026-08-23)

Review rejected commits `3793fa9d` and `297197bf` for eight blocking issues: AND instead of OR score semantics;
report-only acceptance trusting caller booleans/hashes/counts; weak `config.json` and shard-index authority; no exact
local model identity; quantization inputs bound only to paths; incomplete canonical accounting consistency; weak
pairwise-manifest schema validation; and missing explicit unified-harness regression coverage.

This follow-up corrects the score contract at every global and layer-role scope to `(top1 >= 0.85) OR
(diverse_32 >= 0.85)`. Final KL remains a separate mandatory maximum in nats. Regression cases cover top-1-only pass,
diverse-32-only pass, and both-fail rejection. The standalone `gate` command no longer has a report-only invocation:
it reloads the model, reparses configs and safetensors, rehashes dense/checkpoint files and every manifest/JSONL,
revalidates the quantization-run content bindings, reruns global and all 252 direct intervention scopes, and requires
the submitted report to exactly equal that artifact-recomputed evidence.

The exact local target is `/monster/data/model/Qwen3-8B`; no model download was performed or added to the workflow.
Its local Hugging Face metadata pins revision `b968826d9c46dd6066d109eabc6255188de91218`. Its known `config.json` SHA-256
is `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`; its shard-index SHA-256 is
`f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`. The validated identity is
`model_type=qwen3`, architecture `Qwen3ForCausalLM`, hidden/intermediate 4096/12288, 36 layers exactly, 32 query and
8 KV heads, head dimension 128, vocab 151936, and untied embeddings. The index contains exactly layers 0--35 and no
extra decoder layer. The model card names this post-trained, instruction-following, switchable-thinking model
**Qwen3-8B**, not **Qwen3-8B-Instruct**; the acceptance objective's “Instruct” label refers to its post-trained
instruction behavior rather than a distinct local repository name.

The earlier 2026-08-22 statement that the snapshot was not materialized was incorrect: inspection on 2026-08-23
confirmed the complete five-shard 16,381,470,720-byte indexed snapshot was already local. This correction preserves
the earlier entry while superseding that blocker. No quantization or accuracy run has been performed, and no >=85%
result is claimed. The actual model-run evidence remains unresolved until the frozen quantize/save/reload and complete
artifact-aware gate commands finish.

Implementation-gate ledger for this rejected-review repair (run from
`/root/qvq/.worktrees/qwen3-acceptance` on branch `polly/qwen3-acceptance`; parent commit
`297197bfd3bad4bc71ecf446ae011b0d8c3dfd13`):

- `pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: 41 passed, 14 upstream
  `torch.jit` deprecation warnings.
- `pytest --collect-only -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: exactly 41 tests
  collected, matching the executed focused-plus-unified files.
- `ruff check gptqmodel/utils/qvq_acceptance.py scripts/accept_qwen3_8b_qvq.py scripts/qvq_quantize.py
  tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: all checks passed.
- `python -m compileall -q gptqmodel/utils/qvq_acceptance.py scripts/accept_qwen3_8b_qvq.py
  scripts/qvq_quantize.py tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: passed.
- `python scripts/accept_qwen3_8b_qvq.py --help`, `python scripts/accept_qwen3_8b_qvq.py gate --help`, and
  `python scripts/qvq_quantize.py --help`: passed; the gate help requires all authoritative artifact/evaluation
  inputs and exposes no accepting report-only path.
- `validate_qwen3_model_artifact(Path('/monster/data/model/Qwen3-8B'), require_pinned_dense=True)`: passed with the
  pinned revision and config/index hashes recorded above, exact layers 0--35, and all seven critical local Hub
  content identities.
- `git diff --check`: passed. Worktree cleanliness is checked again after the repair commit. The resulting commit is
  identified by `git rev-parse HEAD`; artifact identity remains the hashes above until a real quantized checkpoint
  exists.

The implementation gates validate fail-closed behavior and schemas only. Effective BPW and all score/KL gates still
have no real Qwen3-8B quantized artifact to measure, so they remain unresolved rather than recorded as successes.

## Trust-boundary repair for PR #1 (2026-08-23)

This repair preserves every historical success and failure above while superseding three insufficient evidence
schemas. Required V2B2-P32 tensors now have one exact serialized contract per canonical projection: `trellis` is I32
with shape `[in/16*out/16, 16]`, `SU` and `SV` are F32 vectors of the exact input/output dimensions, `bank_ids` is a
U8 vector with one byte per tile, and `bank_alt_id` is one U8 value. Missing names, aliases, wrong shapes, and wrong
dtypes all fail closed in both checkpoint accounting and report validation.

An acceptance quantization run now hashes and validates the complete pinned seven-file dense digest map at explicit
`quantization_start` and `quantization_end` stages. The observations carry a run ID, the real producer PID, sealed
record digests, and a start-to-end link; both maps must equal the pinned authority. Pre-save payload evidence is made
from the live in-memory producer model and links to that end observation. A subprocess reload and the later acceptance
evaluation reload have distinct real PIDs, canonical stages, and predecessor links. Consequently, hashing the final
checkpoint repeatedly cannot satisfy the required pre-save stage or the three-process provenance chain.

These are implementation and integrity gates, not a retroactive model-quality claim. The earlier successful fixture,
identity, manifest, lint, compilation, and CLI checks remain valid historical results; the earlier absence of a full
quantized artifact, measured BPW, Top-1/Diverse-32 scores, and final KL remains an explicitly preserved failure state.

## Rejected parity-provenance review and controller correction (2026-08-23)

Independent review accepted the exact packed-tensor schema and dense-source start/end binding above, but rejected the
parity provenance in commit `283faa3f`: each stage could choose textual UUID/PID fields and hash them itself. PID
inequality did not establish process-instance identity, PID reuse was not modeled, and evaluation could self-author the
observation that it then validated. Those are preserved as rejected implementation evidence, not silently rewritten
as a success.

Schema v4 replaces that authority with `controlled-run`. One parent acceptance controller uses `secrets.token_hex(32)`
for an unpredictable run challenge, a separate stage nonce, and a separate process-instance identity for each of the
producer, fresh reload, and evaluation. It creates an inherited socketpair for each child and records the actual child
PID, controller PID, argv digest, monotonic spawn/event/exit times, exit code, event digest, and prior-record digest.
PIDs are recorded facts only: repeated PIDs are legal, while controller-issued process-instance identities must remain
unique. This revision signed the transcript with a controller-generated ephemeral Ed25519 key and carried that public
key in the candidate transcript. The later trust-root review below rejects that self-selected-key authority while
preserving this description as implementation history.

The producer hashes the live in-memory packed model and sends its dense binding and pre-save payload over the inherited
controller channel. It blocks until the controller acknowledges that exact event; only then can `model.save` execute.
The producer no longer launches or vouches for its own reload. The controller waits for producer exit, launches the
fresh reload, checks its payload against the acknowledged pre-save event, records exit, and then launches evaluation
as a third process instance. Evaluation emits its payload and a digest of accounting, manifests, thresholds, global
metrics, and all 252 cell metrics to the controller. It cannot add or validate its own controller transcript. The
controller adds the final signed transcript and only then calls the report validator.

The authoritative invocation is now one controller-owned lifecycle:

```bash
python scripts/accept_qwen3_8b_qvq.py controlled-run \
  --dense-model /monster/data/model/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint \
  --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --quant-config configs/qwen3_8b_qvq_w2_acceptance.json \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --held-out-diagnostics-jsonl artifacts/qwen3_8b_qvq_w2/splits/held_out_diagnostics.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --output artifacts/qwen3_8b_qvq_w2/acceptance.json \
  --controller-authority-output artifacts/qwen3_8b_qvq_w2/controller-authority.json
```

The authority receipt is an independent output, not a field chosen by the acceptance report. `gate` requires it via
`--controller-authority`; report-only validation fails closed. The receipt binds the controller instance, run
challenge, public-key digest, and complete transcript digest. Treat it as the out-of-band trust root produced by the
controller invocation, alongside (not inside) the candidate report.

Direct acceptance-mode quantization, payload reload, or evaluation without the inherited controller channel now fails
closed. Adversarial regressions reject final-checkpoint-only fabrication, caller-chosen textual IDs/nonces, replayed
stage events, missing spawn/exit facts, and self-authored evaluation evidence; a positive regression proves PID reuse
does not collapse two distinct controller-issued process instances. This correction changes no model-quality result:
the full quantized artifact, BPW, Top-1/Diverse-32, and final-KL measurements remain unresolved until the command above
is actually completed.

## Rejected self-selected verifier and pinned trust-root correction (2026-08-23)

Review of commit `49074531` rejected the preceding controller design. Its transcript-selected ephemeral public key and
matching receipt were internally consistent but not independent authority: a final-checkpoint-only fabricator could
generate another key and re-sign the entire construction. The receipt described above is therefore retained only as
rejected history and is not the operational trust root.

Controller schema v2 pins the verifier public key at
`configs/qwen3_8b_acceptance_verifier_public.pem`, committed independently of candidate outputs. The corresponding
Ed25519 private key is never stored in this repository. Before a run, an operator provisions it at an absolute path
outside the checkout with mode `0600` and exports `GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY=/absolute/external/key.pem`.
The controller derives its public key and requires an exact match with the pinned key before it creates a candidate
run. Both controller startup and gate verification fail closed if the trust root is absent, malformed, or mismatched;
a transcript-carried attacker key cannot override it. The output receipt binds the already-pinned fingerprint and is
transport evidence, not authority selection.

```bash
install -m 600 /secure/operator/qwen3-acceptance-verifier-private.pem /secure/runtime/verifier.pem
export GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY=/secure/runtime/verifier.pem
openssl pkey -in "$GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY" -pubout |
  cmp - configs/qwen3_8b_acceptance_verifier_public.pem
```

The controller now rejects substituted stage executables/subcommands and noncanonical producer model/row arguments
before spawn. Signed records contain the exact argv plus digest, controller identity and parent PID, observed child and
child-parent PIDs, unique stage/process identities, and monotonic spawn/event/exit chain. PID reuse remains permitted
only when controller-issued process-instance identities differ. The validator independently rechecks those facts and
directly joins the producer record identity tuple to dense start/end and all three parity observations.

Most importantly, the producer cannot receive its pre-save acknowledgement after supplying only nonce-shaped data.
Before acknowledging, the controller validates the pinned dense start/end map and seal, exact 252-module canonical
live payload census/hashes, frozen W2 V2B2-P32 config, all-layer scope, and three verified 512-row manifest streams,
all joined to the controller challenge and observed producer instance. Only that validated acknowledgement lets the
producer proceed to `model.save`. Regressions cover a complete attacker-key/re-signed transcript and receipt,
command substitution, false parent/child facts, injected records, split identity tuples, replay, and malformed live
producer payload before acknowledgement. Real quantization quality evidence remains unresolved.

Implementation gates for this trust-root correction, from parent `49074531c681e41fb2724b8d797c658f6f3ba5e9`:

- `HOME=/root pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: 72 passed,
  14 upstream `torch.jit` deprecation warnings.
- The identical two-file `pytest --collect-only -q` command: exactly 72 tests collected.
- `ruff check` and `python -m compileall -q` on both acceptance utilities, both CLIs, and both focused test files:
  passed; `git diff --check` passed.
- Help smokes passed for the acceptance root, `gate`, `controlled-run`, `payload-hashes`, `evaluate`, and the quantizer.
- OpenSSL derived the external private key's public half and `cmp` matched the pinned PEM; repository scanning found
  no private-key material; a controller signed/verified a schema-v2 empty-prefix transcript with pinned fingerprint
  `5a16757e0eea7aa47b93b1ffa7e0b6000390cd61264e35ea960681e0dace0930`.
- Pinned dense identity/integrity revalidation passed for all seven authoritative files, all 36 layers, and revision
  `b968826d9c46dd6066d109eabc6255188de91218`.

## Rejected repository-key/process-assertion design and external deployment authority (2026-08-23)

Independent review rejected commit `c66ceb45` despite its improvement over the transcript-selected key. A public key
committed in the candidate repository was still candidate-controlled deployment state; stage validation accepted an
untrusted `argv[0]`, open option sets, child-asserted PID facts, well-formed but unrecomputed manifest/payload hashes,
and a private-key path that could change between checks. The schema-v2 design above is preserved as rejected history,
not current authority. In particular, `configs/qwen3_8b_acceptance_verifier_public.pem` is removed.

Schema v3 has two operator inputs, both outside the candidate checkout:

- `GPTQMODEL_QVQ_TRUST_CONFIG` names an absolute, root/operator-owned, regular JSON file with exact mode `0600`.
  Its closed schema pins the external verifier public-key path and SHA-256, the normalized absolute Python interpreter
  path and content SHA-256, the OpenSSL executable path/content SHA-256, and the normalized absolute path/content
  SHA-256 for each producer/reload/evaluation script. The gate reads this deployment authority independently; no
  transcript or receipt selects a key or executable.
- `GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY` names an absolute external regular file owned by the effective operator and
  having exactly mode `0600`. It is opened once with `O_NOFOLLOW|O_CLOEXEC`; data and identity come from the same
  descriptor and stable `fstat` observations. Symlinks, mode `0700`, mode `0644`, ownership mismatch, in-read changes,
  and path/inode replacement before signing fail closed. Signing uses only the bytes captured from that stable open.

Provision after installing the reviewed code and interpreter, before exposing any candidate output:

```json
{
  "schema": "qvq-acceptance-trust-v1",
  "verifier_public_key": "/secure/qvq/verifier-public.pem",
  "verifier_public_key_sha256": "<sha256sum of external verifier-public.pem>",
  "python_executable": "/absolute/resolved/operator/python",
  "python_executable_sha256": "<sha256sum of that interpreter>",
  "openssl_executable": "/absolute/resolved/operator/openssl",
  "openssl_executable_sha256": "<sha256sum of that OpenSSL executable>",
  "acceptance_policy": "/reviewed/repo/configs/qwen3_8b_qvq_acceptance_policy.json",
  "acceptance_policy_sha256": "<sha256>",
  "quant_config": "/reviewed/repo/configs/qwen3_8b_qvq_w2_acceptance.json",
  "quant_config_sha256": "<sha256>",
  "stage_scripts": {
    "quantization_producer": {"path": "/reviewed/repo/scripts/qvq_quantize.py", "sha256": "<sha256>"},
    "fresh_process_reload": {"path": "/reviewed/repo/scripts/accept_qwen3_8b_qvq.py", "sha256": "<sha256>"},
    "acceptance_evaluation": {"path": "/reviewed/repo/scripts/accept_qwen3_8b_qvq.py", "sha256": "<sha256>"}
  }
}
```

```bash
chmod 0600 /secure/qvq/trust.json /secure/qvq/verifier-private.pem
export GPTQMODEL_QVQ_TRUST_CONFIG=/secure/qvq/trust.json
export GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY=/secure/qvq/verifier-private.pem
/absolute/resolved/operator/python scripts/accept_qwen3_8b_qvq.py controlled-run ...
```

Every stage now has one exact ordered argv grammar. The controller and gate require the operator-pinned `argv[0]`,
exact script path/content identity, unique required options, normalized absolute paths, and no unknown, duplicate, or
extra flags. After spawn, Linux `/proc/<pid>/{stat,exe,cmdline}` supplies PID, PPID, process start ticks, executable
identity, and cmdline digest independently of child messages. PID reuse is distinguished by start ticks plus the
controller-issued process-instance identity.

Before producer spawn, the controller opens all three 512-row JSONL streams and manifests, recomputes file hashes and
every canonical content hash, verifies ordinal/identity/count agreement, uniqueness, and pairwise identity/content
disjointness, and retains that evidence in the signed transcript. The producer must return that exact controller-owned
evidence. Payload hash scheme v2 recomputes the aggregate over the canonical ordered 252 `(module name, module hash,
tensor count)` records. The producer's live in-memory payload travels on the controller-owned socket and the exact
event digest is acknowledged only after these checks; the independently spawned reload must subsequently match it.

Adversarial tests re-sign and reseal malicious records so rejection is not an incidental signature/digest failure:
malicious `argv[0]`, extra/duplicate flags, well-formed fake manifest hashes, inconsistent aggregate/module hashes,
false OS parent facts, external fingerprint mismatch, attempted repository key substitution, unsafe private modes,
symlinks, and path replacement all fail closed. This remains implementation evidence only; no real BPW, Top-1,
Diverse-32, or final-KL metric is claimed.

Implementation gates for the schema-v3 correction, from parent
`c66ceb45afa321ade50a4dfb4424f1b1c21a8bc3`:

- `HOME=/root pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: 83 passed with
  14 upstream `torch.jit` deprecation warnings.
- The identical two-file `pytest --collect-only -q` command collected exactly 83 tests.
- Ruff and `compileall` on both acceptance utilities, both CLIs, and both focused test files passed; `git diff
  --check` passed.
- Acceptance root, `gate`, `controlled-run`, `payload-hashes`, `evaluate`, and quantizer help smokes passed.
- The external trust JSON and private key were root-owned regular files with exact mode `0600`; externally pinned
  interpreter and stage-script hashes validated; schema-v3 signing/verification passed; repository scanning found no
  private key material.
- Pinned dense identity/integrity passed for seven files, layers 0--35, and revision
  `b968826d9c46dd6066d109eabc6255188de91218`.

## Rejected path/digest assertions and descriptor-owned schema v4 (2026-08-23)

Review rejected commit `f4ba2e38`: it strengthened schema v3 but still hashed trusted executables through paths,
checked containment lexically, accepted producer-authored 252-module digests without seeing tensor bytes, verified
datasets before reopening them by name, omitted same-inode key mutation fields, left evaluation semantics and
cross-stage paths partly caller-selectable, and split `/proc/<pid>/stat` on spaces. This is preserved as rejected
design history rather than amended into a success.

Schema v4 opens the trust JSON, external public/private keys, Python, OpenSSL, locked policy, quantization config, and
stage scripts descriptor-first. Every path component is inspected with `lstat`; any final or parent-directory symlink
fails closed. The opened descriptor must be a stable regular file with safe owner/mode, and its bytes are hashed
between matching `fstat` observations. Trust/private containment uses the fully resolved physical path. Immediately
before spawn and after the child's live event, the controller reopens/revalidates the exact trusted executable/script
identities; `/proc/<pid>/exe` is itself opened and hashed to bind the running interpreter inode. The signing key's
device, inode, size, nanosecond mtime/ctime, and content digest are rechecked before every signature, rejecting both
path replacement and same-inode mutation.

The controller snapshots all verified JSONL and manifest bytes into sealed Linux memfds. The producer inherits only
those immutable descriptors and its dataset loader consumes `/proc/self/fd/<n>`, while reported evidence comes from
the same controller snapshot. Changes to original paths after verification cannot affect calibration. Snapshot
creation seals writes, growth, shrinkage, and further seal changes.

Live pre-save evidence is no longer a digest assertion. Before `model.save`, the producer streams every direct packed
tensor's canonical module/tensor name, dtype, shape, byte count, and actual contiguous bytes over the controller-owned
socket. The controller enforces canonical order, exact 252-module census, all required packed tensor metadata,
dtype/shape byte extents, hashes every frame itself, and recomputes module and aggregate digests. It acknowledges only
when that independently computed payload equals the producer claim; a complete fabricated set of 252 coherent
digests cannot pass.

The externally pinned acceptance policy fixes revision
`b968826d9c46dd6066d109eabc6255188de91218`, device `cuda:0`, all-layer scope, three 512-row streams, BPW `2.1`, score
minimum `0.85`, and final-KL maximum `0.1`. Exact argv validation and the signed transcript require producer output,
reload checkpoint, and evaluation checkpoint to be identical; all manifest/evaluation paths must derive from the
same producer manifest directory. Re-signed revision, threshold, checkpoint, or dataset path substitutions fail.
Linux stat parsing now finds the final `)` comm boundary and indexes fields 4 and 22 from the remaining fields, so
spaces and parentheses in process names cannot shift PPID or process-start identity.

New adversarial coverage includes parent-directory symlinks, same-inode key mutation, verify/use dataset mutation,
fabricated coherent 252-module digests, re-signed revision/threshold/path swaps, and hostile proc comm strings. No
private material is committed, and no real BPW, Top-1, Diverse-32, or final-KL result is claimed.

Implementation gates for this schema-v4 correction, from parent
`f4ba2e3867096aaa2ba8499bfb78a973dc20208d`:

- `HOME=/root pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: 93 passed with
  14 upstream `torch.jit` deprecation warnings; the identical two-file collect-only invocation found exactly 93.
- Ruff, `compileall`, and `git diff --check` passed for the changed acceptance/controller, quantizer, and focused
  test files. Six CLI help smokes passed, including the acceptance root/gate and controller entry points.
- The operator provisioned an external root-owned trust configuration, Ed25519 public key, and private signing key
  under `/root/.config/qvq`, all exact mode `0600`. Descriptor-first trust loading and signing-key/public-key matching
  passed; repository scanning found no private key material.
- The pinned dense model identity gate passed for `/monster/data/model/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`, exact decoder layers 0--35, and the authoritative artifact hashes.

## Rejected pathname verification and retained-descriptor schema v5 (2026-08-23)

Review rejected commit `752b9adf`: schema v4 still separated component `lstat` from final `open`, reopened verified
policy/config/script paths at use time, closed the socket reader before consuming producer tensor frames, allowed
mutable dataset fallback and path aliases, accepted extra live tensors, and did not prove one stable `/proc` identity
across observation steps. Those findings remain recorded as failures rather than retroactively described as success.

Schema v5 walks every trusted path with directory descriptors and `openat`, applying `O_NOFOLLOW` to every component.
Each directory must be owned by root/the controller and not rename-capable by group/other (a root-owned sticky
directory is treated as non-rename-capable); final files must be regular, safely owned, and non-writable by
group/other. The controller retains the trust, key, Python, OpenSSL, policy, quant config, and stage-script descriptors,
revalidates their inode/timestamps/content before spawn and after stage evidence, and executes the retained Python
and script descriptors through `/proc/self/fd`. The producer reads quant configuration and sealed datasets only from
inherited controller descriptors. Canonical dataset keys are physical absolute paths; aliases and `..` fail closed,
and the presence of controller snapshot metadata disables every mutable-source fallback.

The producer socket reader now remains alive through event parsing, all 1,260 canonical tensor frames, independent
controller hashing, validation, and acknowledgement. Every one of the 252 modules must contain exactly the five
canonical tensors in canonical order with exact names, metadata, byte extents, and no additional parameter/buffer.
Linux process observation brackets executable/cmdline reads with identical stat tuples, requires the stat PID to
equal `Popen.pid`, and binds PID, PPID, start ticks, and executable device/inode to the retained pre-spawn Python
descriptor. A controller-owned Linux pidfd remains open through `Popen.wait`, preventing PID reuse from satisfying an
exit record. Mixed observations, handoff, and PID reuse within a process instance fail closed.

Runtime regression coverage executes descriptor-backed Python/script bytes after pathname replacement and drives the
actual producer socket sender through a 252-module live stream; the producer cannot cross the simulated save boundary
until controller validation returns the acknowledgement. Additional attacks cover unsafe parent directories,
dataset aliases, extra tensors, mixed process observations, and post-snapshot mutation. This is implementation trust
evidence only: no real BPW, Top-1, Diverse-32, or final-KL metric is claimed.

Implementation gates for schema v5, from parent `752b9adffe6ea5f8538eb7a0525eff34c8b10e17`:

- The exact focused-plus-unified pytest invocation passed all 100 tests with 14 upstream `torch.jit` deprecation
  warnings; the identical collect-only invocation found exactly 100 tests.
- Ruff, `compileall`, `git diff --check`, six CLI help smokes, external schema-v5 signing/trust-root validation, and
  the repository private-key scan passed.
- Pinned `/monster/data/model/Qwen3-8B` identity/integrity passed for revision
  `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers, seven authoritative artifact hashes, and seven local
  Hub content identities.

## Rejected pre-coherence uniqueness ordering and local-candidate correction (2026-08-23)

Independent verification rejected commit `b9de5c74`: despite its green tests, it consulted and mutated the global
resource-uniqueness sets after only descriptor/evidence hash checks and before complete manifest/schema/512-record
coherence. A malformed candidate that also aliased an earlier resource could therefore report uniqueness instead of
its local trust failure, and the ledger's earlier success statement did not prove the required ordering.

Each role is now built by `_validated_snapshot_candidate`, which has no access to the function-wide uniqueness sets.
It validates canonical physical paths; stable regular-file descriptor identities before and after retained reads;
exact path/hash evidence correspondence; the canonical source/manifest path relationship; manifest split, closed
schema, count, and samples; and every one of the 512 JSONL identities and content hashes. Only the returned coherent
candidate reaches same-role inequality and the shared path/FD uniqueness transaction. Both paths and both FDs are
inserted only after all checks pass, and only then is the candidate committed to validated authority state.

Ordering regressions prove a malformed aliased candidate reports coherence rather than uniqueness, locally coherent
aliases reach the global uniqueness gate, and a failed local candidate cannot poison subsequent candidate
validation. No real BPW, Top-1, Diverse-32, or final-KL metric is claimed.

Implementation gates for this ordering correction, from parent `b9de5c74d0a990deac2db858a50c59a5d456f05b`:

- The exact focused-plus-unified pytest invocation passed all 144 tests with 14 upstream `torch.jit` warnings; the
  identical collect-only invocation found exactly 144 tests.
- Ruff, `compileall`, `git diff --check`, six CLI help smokes, external schema-v7 trust/signing validation, and the
  repository private-key scan passed.
- Pinned Qwen3 identity/integrity passed for revision `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers,
  seven authoritative artifact hashes, and seven local Hub identities.

## Rejected split-category uniqueness and schema-v7 ambiguity correction (2026-08-23)

Review rejected commit `46863ae1`: its snapshot authority maintained separate source-path and manifest-path sets,
so it did not explicitly reject a source path aliased to another role's manifest path, and it did not explicitly
reject reuse of one FD as both source and manifest within a role. The accepted schema-v7 checks and all earlier
results remain recorded above; this is a focused correction rather than a claim that the rejected ambiguity was safe.

Snapshot validation now uses one global physical-path uniqueness set spanning every source and manifest path and one
global descriptor uniqueness set spanning every source and manifest FD. It first requires the two paths and two FDs
inside each role to be distinct, then rejects any cross-role or cross-category reuse before reading evidence. Runtime
regressions cover same-role FD equality, source-to-other-manifest and manifest-to-other-source path aliases,
cross-category FD reuse, and preservation of a valid mapping. A further FD-count regression acquires the complete
signing authority, injects a constructor failure immediately after assignment, and proves both that authority and all
trusted resources are closed. No real BPW, Top-1, Diverse-32, or final-KL result is claimed.

Implementation gates for this correction, from parent `46863ae1f569480dbd7c47ca5c4a3a3d87ce55d6`:

- The exact focused-plus-unified pytest invocation passed all 138 tests with 14 upstream `torch.jit` warnings; the
  identical collect-only invocation found exactly 138 tests.
- Ruff, `compileall`, `git diff --check`, six CLI help smokes, external schema-v7 trust/signing validation, and the
  repository private-key scan passed.
- Pinned Qwen3 identity/integrity passed for revision `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers,
  seven authoritative artifact hashes, and seven local Hub identities.

## Rejected incoherent alias tests and resource-uniqueness correction (2026-08-23)

Review rejected commit `eac9033a`: the ledger's claimed ambiguity coverage was a false positive. Its negative tests
changed a path or FD without replacing the corresponding resource, so earlier path/evidence/content checks could
reject incoherent mappings without demonstrating the intended global uniqueness gate. The implementation also made
the uniqueness decision before establishing each path/FD resource from retained bytes and evidence. The preceding
gate results remain historical execution results, but they are not evidence that the reviewed ambiguity was fixed.

The producer now establishes coherent source and manifest resources from their canonical physical paths, retained
descriptor bytes, and exact evidence hashes before uniqueness insertion. One `seen_resource_paths` set spans both
path categories and all roles, and one `seen_resource_fds` set spans both FD categories and all roles. Within-role
source/manifest equality has a dedicated duplicate-path or duplicate-descriptor failure. Globally, either resource
path is checked against the shared path set and either descriptor against the shared FD set, with both inserted only
after the coherent resource checks succeed. Errors state whether path or descriptor ambiguity caused rejection.

The replacement regressions use coherent resource aliases. Path attacks pair the aliased path with a duplicate FD
for the same retained bytes; FD attacks pair the reused FD with a distinct physical copy containing those bytes.
They independently cover same-role source/manifest equality, source/source, manifest/manifest, both cross-category
path directions, both same-category FD directions, and both cross-category FD directions, while preserving a valid
distinct mapping. The constructor cleanup regression instruments assignment itself, proves the fully acquired
authority was transferred into controller state, captures every authority/resource FD, and proves each FD is closed
and the aggregate FD count restored after injected failure. No real metrics are claimed.

Implementation gates for this correction, from parent `eac9033a6c91bc177e0e99b3cc02aac262b30f71`:

- The exact focused-plus-unified pytest invocation passed all 142 tests with 14 upstream `torch.jit` warnings; the
  identical collect-only invocation found exactly 142 tests.
- Ruff, `compileall`, `git diff --check`, six CLI help smokes, external schema-v7 trust/signing validation, and the
  repository private-key scan passed.
- Pinned Qwen3 identity/integrity passed for revision `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers,
  seven authoritative artifact hashes, and seven local Hub identities.

## Rejected open producer schemas and retained-validation schema v6 (2026-08-23)

Review rejected commit `32e4a243`: schema v5 still permitted additional producer/config fields, treated a present
but falsey snapshot authority as absence, reopened trusted paths while validating the transcript, resolved the
textual `/proc/<pid>/exe` symlink, and left exceptional/standalone signing-resource ownership ambiguous. These are
preserved as rejected findings, not amended into the schema-v5 success ledger.

Schema v6 requires exact key equality for the producer event, measurement, dense binding, start/end observations,
pre-save payload, quantization config (equal to the retained verified JSON bytes), each dataset observation, and the
252-module hash object. Snapshot authority has one closed schema; if its environment variable exists, empty text,
`{}`, `null`, the wrong type/schema, malformed evidence, or a missing canonical FD/evidence entry fails closed. Only
complete controller-issued sealed descriptors can be consumed; mutable fallback exists solely when the authority
environment variable is genuinely absent.

One `_TrustedResources` descriptor set is now passed through authority receipt construction, signature verification,
policy/command validation, and executable identity checks. Transcript validation performs no trust-config reload,
trusted-path `stat`, or Python/script/policy/config reopen. Path replacement after retention therefore validates the
reviewed descriptors, while same-inode/content mutation fails retained identity revalidation. Process executable
identity comes only from opening `/proc/<pid>/exe` and hashing/fstat-ing that descriptor, so an unlinked running
executable remains observable without interpreting a `(deleted)` pathname.

Policy parsing and public/private-key derivation close owned resources on every success and exception. Standalone
signing-key loading returns a context-owned authority containing live private/trust descriptors; leaving its context
closes all of them, while controller-supplied resources remain controller-owned. FD-count regressions cover malformed
policy and mismatched-key failures. No real BPW, Top-1, Diverse-32, or final-KL result is claimed.

Implementation gates for schema v6, from parent `32e4a2437f94d0248d3ca21b7258bbb1d9050255`:

- The exact focused-plus-unified pytest command passed all 119 tests with 14 upstream `torch.jit` warnings; the
  identical collect-only command found exactly 119 tests.
- Ruff, `compileall`, `git diff --check`, six CLI help smokes, external schema-v6 trust/signing validation, and the
  repository private-key scan passed.
- Pinned Qwen3 identity/integrity passed for revision `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers,
  seven authoritative artifact hashes, and seven local Hub identities.

## Rejected shallow snapshot semantics and process-image schema v7 (2026-08-23)

Review rejected commit `29550316`: schema v6 checked the snapshot authority's closed keys but did not semantically
revalidate every evidence value against the retained bytes, inferred a dataset role from a filename/stem, observed
only one executable descriptor around a cmdline read, and leaked retained descriptors when controller construction
failed after resource acquisition. These findings remain failures in the ledger; the earlier schema-v6 successes are
preserved above.

Schema v7 replaces filename inference with a closed controller-issued mapping for exactly `calibration`, `yaqa`, and
`validation`. Each role explicitly binds its canonical physical source and manifest paths, sealed source and manifest
FDs, canonical manifest split, and evidence object. The producer independently reads the retained descriptors and
requires exact hashes, 512 JSONL and manifest records, row identities and content identities, exact manifest schema,
`row_start=0`, `rows=512`, `manifest_verified=true`, and cross-split identity/content disjointness. Falsey or merely
well-shaped evidence, non-lowercase hashes, missing descriptors, aliases, duplicate paths/FDs, and ambiguous role
mappings fail closed.

Linux observation now brackets the process image with two independently opened and fully hashed `/proc/<pid>/exe`
descriptors, two identical cmdline reads, and three identical PID/PPID/start-time stat samples. Later observations
also bind executable-content and cmdline hashes, so a same-PID/start-tick `execve` handoff is rejected. A runtime
regression drives a real child through `execve` between observation reads. Controller initialization is transactional:
any exception closes the signing authority, private-key descriptor, and every partially acquired trusted resource;
FD-count regressions cover both key mismatch and OpenSSL failure. No real BPW, Top-1, Diverse-32, or final-KL result
is claimed.

Implementation gates for schema v7, from parent `2955031625e1cb556b5d582af4dd8e77538f1bb7`:

- `HOME=/root pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py` passed all 132
  tests with 14 upstream `torch.jit` deprecation warnings; the identical collect-only invocation found exactly 132.
- Ruff, `compileall`, `git diff --check`, and six CLI help smokes passed. External descriptor-owned trust loading,
  private/public Ed25519 matching and signing passed for controller schema v7; the repository private-key scan was
  empty.
- Pinned `/monster/data/model/Qwen3-8B` identity/integrity passed for revision
  `b968826d9c46dd6066d109eabc6255188de91218`, all 36 layers, seven authoritative artifact hashes, and seven local
  Hub content identities.
