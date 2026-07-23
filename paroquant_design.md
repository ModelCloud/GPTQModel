# ParoQuant paper-recovery design

## Goal

Recover the calibration regime used by the ParoQuant paper and its official `legacy` reference while preserving
GPT-QModel's existing non-ParoQuant paths and ParoQuant fallback behavior.

Working branch: `paroqunt-recovery` (spelling requested by the user).

Validation GPU: physical GPU 6, PCI bus `00000000:DE:00.0`, UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, NVIDIA PG506-230, 96 GiB, compute capability 8.0, 124 SMs.

## Normative sources

- Paper (ICLR 2026 / arXiv v2): <https://arxiv.org/html/2511.10645v2>
- Official repository: <https://github.com/z-lab/paroquant>
- Paper-reproduction branch: <https://github.com/z-lab/paroquant/tree/legacy>
- Reference launch configuration:
  <https://github.com/z-lab/paroquant/blob/legacy/experiments/optimize/4bit.sh>
- Reference data construction:
  <https://github.com/z-lab/paroquant/blob/legacy/paroquant/util.py>
- Reference optimization driver:
  <https://github.com/z-lab/paroquant/blob/legacy/optimize.py>

The official repository explicitly directs paper reproduction to `legacy`; current `main` is not the normative
algorithm/reproduction target.

## Paper/reference contract

| Property | Paper and `legacy` reference |
| --- | --- |
| Training set | 2,048 sequences, mixed evenly across WikiText2, C4, and RedPajama |
| Validation set | 64 separate sequences from Pile validation |
| Sequence length | 2,048 tokens per sequence |
| Shuffle | Fixed seed 0 for each source and the final training mixture |
| Calibration batch | 16 sequences (halved for the 70B case) |
| Optimization scope | Whole decoder layer |
| Selection | Validate after every epoch and restore the best validation state |

The reference constructs each source by shuffling documents, concatenating eligible documents, splitting the token
stream into full 2,048-token sequences, and then shuffling the mixed training sequences. Training and validation are
captured and propagated as independent streams through every decoder layer.

## Confirmed gaps in the implementation at branch creation

1. `opt_train_samples` and `opt_validation_samples` are interpreted as flattened activation-row budgets. With
   2,048-token inputs, the default training budget can collapse to roughly one sequence instead of 2,048 sequences.
2. Training and validation are selected from the prefix and suffix of one calibration stream; callers cannot provide
   the paper's independent Pile validation stream.
3. GPT-QModel has no reusable builder for the exact WikiText2/C4/RedPajama plus Pile calibration corpus recipe.

## Implementation design

1. Add an optional, ParoQuant-only validation calibration input to the public quantization lifecycle.
2. Prepare training and validation independently, cap each by sequence count, then concatenate only for the shared
   activation-capture pass. Retain the exact batch boundary so optimization never derives validation from training
   when an explicit validation set is supplied.
3. Split grouped/layer replay by captured sequence batches, not flattened token rows. Keep a compatibility fallback
   for callers that supply only one calibration stream.
4. Split module-scope captures by their recorded calibration batch indices before selecting activation rows. This
   keeps the lightweight module optimizer bounded while preventing train/validation leakage.
5. Preserve the tuned `opt_batch_size=64` default and the independent calibration `batch_size=1` default. Select
   the reference value 16 explicitly in the paper-reproduction call instead of treating it as an algorithmic
   requirement.
6. Add a lazily imported calibration-data builder that reproduces the official source mix and sequence construction
   without making dataset downloads an implicit side effect of `quantize()`.
7. Document the exact paper-reproduction call. Whole-layer scope remains an explicit choice because changing the
   global ParoQuant scope default is outside this calibration-only change.
8. Remove inherited GPTQ `desc_act` from ParoQuant's public configuration and serialized format. Keep only an
   internal `False` sentinel for shared quantized-linear construction, and strip global or dynamic occurrences while
   loading legacy ParoQuant configs.

## Recovered match status

| Reference requirement | Implementation status |
| --- | --- |
| 2,048 evenly mixed WikiText2/C4/RedPajama sequences | Matched by the explicit calibration builder |
| Independent 64-sequence Pile validation stream | Matched by `validation_calibration` and strict capture boundaries |
| Samples mean full sequences, not activation token rows | Matched in capture limits and layer/group replay splitting |
| Sequence length 2,048 and seed 0 | Matched by builder defaults |
| Batch size 16 | Selected explicitly by the reproduction call; tuned global defaults remain unchanged |
| Whole-layer recovery with best validation checkpoint | Matched when exact usage selects `opt_scope="layer"` and `opt_stage_impl="reference"` |

The existing one-stream and module-scope modes remain supported, but they are compatibility/lighter-weight modes and
are not claimed as exact paper reproduction.

## Compatibility rules

- GPTQ, AWQ, QQQ, EXL3, weight-only methods, CPU execution, and non-target GPUs must be unchanged.
- Existing ParoQuant callers with one calibration stream continue to work through the legacy prefix/suffix fallback.
- Legacy ParoQuant checkpoint configs containing `desc_act` continue to load, but the ineffective field is discarded.
- An explicit validation calibration input is rejected for non-ParoQuant methods instead of being silently ignored.
- Dataset libraries remain optional until the paper-calibration builder is called.
- No dataset download is performed by tests.

## Progress

- [x] Read the paper's calibration and optimization sections.
- [x] Read the official `legacy` launch script, dataset builder, capture path, and optimizer.
- [x] Trace GPT-QModel's public quantization API, calibration preparation, activation capture, and ParoQuant split logic.
- [x] Create the requested branch and identify physical GPU 6.
- [x] Implement separate train/validation lifecycle and sequence-aware split logic.
- [x] Implement the paper calibration-data builder.
- [x] Restore the official pre-stage validation baseline and rollback semantics in grouped reference mode.
- [x] Add focused unit and API-routing tests.
- [x] Run Ruff, focused CPU tests, and `git diff --check`.
- [x] Run a targeted CUDA validation on physical GPU 6 and record the software environment/results.
- [x] Run the requested `desc_act=False` and `opt_batch_size=32` full-score sweeps.
- [x] Remove unused GPTQ `desc_act` from ParoQuant construction and serialization while preserving legacy loading.
- [ ] Define and implement `opt_batch_size` semantics for grouped whole-layer/compute-block runners; those runners
  currently step over captured replay batches and do not consume `qcfg.opt_batch_size`.

## Validation log

Validated on 2026-07-23.

- Static checks: Ruff passed on every changed Python file; `git diff --check` passed.
- CPU/fallback suite: `127 passed, 8 skipped` for `tests/test_paroquant.py` and `tests/test_embed_quant_api.py`
  with CUDA hidden. The skips were CUDA-only cases. This suite includes the dense-output error check that verifies
  ParoQuant optimization improves over identity quantization.
- Non-CUDA runtime selection: the focused NPU/torch fallback constructor test passed with CUDA hidden.
- GPU environment: physical index 6 maps to logical `cuda:0` under
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6`; driver 610.43.02, PyTorch 2.13.0+cu130, CUDA runtime 13.0.
- Targeted GPU test: `4 passed` using FP32 weight shape `[8, 8]`, training activations `[2, 3, 8]`, validation
  activations `[1, 3, 8]`, and one rotation plus one fine-tuning epoch. This exercises the independent validation
  optimizer path and both module/layer boundary selectors.
- End-to-end GPU smoke test: a six-layer tiny Llama model completed whole-layer quantization with separate training
  and validation streams, save, reload, and finite generation through the ParoQuant Triton backend. The smoke test
  used zero optimization epochs to validate lifecycle/serialization rather than paper-quality recovery. The default
  CUDA-AWQ runtime was not applicable to the toy model's 288-wide output shape, so reload was explicitly validated
  with `BACKEND.PAROQUANT_TRITON`.

The full 2,048-sequence, 2,048-token, 10+10 epoch paper reproduction was not run: it requires downloading the four
reference corpora and a long model-specific calibration job. Its deterministic corpus construction is covered by an
offline unit test with an injected dataset loader; no test downloaded data. No CUDA extension build flags changed.

Follow-up validation on 2026-07-23 ran the complete ParoQuant unit suite with CUDA hidden: `124 passed, 8 skipped`.
The two new CPU regression tests force validation to worsen after epoch one and confirm that both the in-memory and
streamed reference runners restore the pre-stage state. No GPU workload was run for this follow-up.

Parameter sweeps on 2026-07-23 used the same fast whole-layer Llama-3.2-1B-Instruct test, physical GPU 6, seed,
four optimized layers, ParoLinear save/reload path, full 1,209-example GSM8K Platinum set, and full 1,172-example
ARC Challenge set:

- `desc_act=False`, `opt_batch_size=64`: passed in 25m17s. GSM8K was `0.46153846153846156`; ARC raw was
  `0.31143344709897613`; ARC normalized was `0.35665529010238906`.
- `desc_act=True`, `opt_batch_size=32`: passed in 24m16s with the same three scores.

Both sweep checkpoints and the earlier default-fast checkpoint have the same model-weight SHA-256,
`92caeceba4c8c843970fb074a9d5255817a66d87a85e45f398c2481e169b7ccf`, and the same four reported layer losses.
Only quantization metadata differs. This proves `desc_act` does not alter the ParoQuant weight path; the setting was
subsequently removed from `ParoConfig` and its serialized payload. Legacy global and dynamic fields are stripped on
load. The sweep also exposed that the batch-size result is not a real optimizer-batch comparison: `_run_group_stage()` and
`_run_group_stage_streamed()` iterate existing calibration replay batches and never read `qcfg.opt_batch_size`.
Module-scope `optimize_paroquant_linear()` does consume the setting.

The `desc_act` configuration cleanup was validated with CUDA hidden: the complete ParoQuant, embedding API, and
focused NPU fallback suite reported `132 passed, 8 skipped`; all skips were CUDA-only. Ruff and `git diff --check`
also passed. No GPU code was executed for this cleanup.
