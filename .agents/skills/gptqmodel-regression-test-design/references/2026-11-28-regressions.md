<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 26–28 audit

The window from Nov 26 at 01:01 UTC to Nov 28 at 01:01 UTC contains 50
main-branch commits, including 39 `fix(...)` commits. Counts describe commits,
each of which can address several defects. The previous audit covers the
overlapping finite-grid, AWQ code-handoff and grouped-recovery fixes.
[Audited revision](https://github.com/ModelCloud/GPTQModel/commit/a311c91b9f980e9d9efba1e0351a63959f77748f) ·
[Nov 27 audit](2026-11-27-regressions.md)

## Preserve invocation and batch identity through the head

Inkling needed three successive repairs at the same handoff. The head first
needed its residual-bank transform, full-forward capture then supplied each
loop visit's own inputs, and interleaved replay exposed a final-bank counter
that followed submission order. Earlier SenseNova calibration had the same
physical-layer versus repeated-visit distinction. These fixes have landed.
[#6122](https://github.com/ModelCloud/GPTQModel/pull/6122) ·
[#6126](https://github.com/ModelCloud/GPTQModel/pull/6126) ·
[#6140](https://github.com/ModelCloud/GPTQModel/pull/6140) ·
[earlier SenseNova report](https://github.com/ModelCloud/GPTQModel/issues/5081)

Use a real tiny Inkling model with 15 layers, loop core `[6, 12)` and three
passes. Capture native arguments and outputs before replay. The expected
invocation counts are six ones, six threes and three ones. Compare each visit's
hidden input, residual bank, messages, mask and output. Prove the first
post-loop input differs from the first-pass output. At the head compare
`norm(final_input(bank, hidden)[0])` with the native input. Keep the G2PTQ
custom-head objective fallback control.
[Existing model tests](../../../../tests/module_tree/test_inkling2_loop_plan.py)

Use eight batches of lengths `(12, 9, 14, 7, 11, 8, 13, 10)` and replay order
`[0, 4, 1, 5, 2, 6, 3, 7]`. Run two subset-replay rounds before preparing head
inputs in batch order. Check each head input against its native batch and
require exactly eight final banks. Extend with equal-shaped, distinct-valued
batches and a fresh-run cleanup case. The existing
`test_inkling2_loop_looped_lm_head_banks_use_batch_identity` supplies the
unequal-length ordering fixture.
[Batch-identity regression](../../../../tests/module_tree/test_inkling2_loop_plan.py)

The generic capture still uses parallel per-layer lists, while Inkling carries
identity in private kwargs and stores head banks separately. The proposal gives
capture, scheduling and the head one owner for identified replay records and
uses existing definition hooks for architecture-specific transforms. It
includes migration and conformance criteria.
[Replay ownership proposal #6148](https://github.com/ModelCloud/GPTQModel/issues/6148)

## Keep real containers and released configuration choices in tiny fixtures

| Repeated assumption | Fixture and observable result | Landed fixes and test families |
| --- | --- | --- |
| Batch preparation succeeds for every definition | Two unequal conversations, batch size two, explicit padded IDs and masks. Extend the shared VLM contract whenever a definition adds its own preparation method. | Inkling [#6122](https://github.com/ModelCloud/GPTQModel/pull/6122), Intern-S1 [#6141](https://github.com/ModelCloud/GPTQModel/pull/6141), [shared suite](../../../../tests/test_vlm_prepare_dataset_contract_4633.py) |
| Captured mappings behave like plain dictionaries | Preserve an actual `BatchFeature` through mixed sliding/full attention replay. Compare each mask and output with native execution. For measurement, feed HF Dataset and Mapping rows directly and check hand-listed token rows. | [#6141](https://github.com/ModelCloud/GPTQModel/pull/6141), [#6116](https://github.com/ModelCloud/GPTQModel/pull/6116), [replay](../../../../tests/test_layer_type_replay.py), [measurement](../../../../tests/test_measure_windows.py) |
| Synthetic defaults represent released models | Shrink released MiniCPM configs while keeping longrope and serialized `head_dim`. Preserve Phi-4's inherited attention hint and pair it with an explicit caller choice. Reject unsupported ChatGLM vision before constructing a text-only model. | [#6120](https://github.com/ModelCloud/GPTQModel/pull/6120), [#6121](https://github.com/ModelCloud/GPTQModel/pull/6121), [#6119](https://github.com/ModelCloud/GPTQModel/pull/6119), [MiniCPM](../../../../tests/test_minicpm_local_port.py), [Phi-4](../../../../tests/module_tree/test_phi4_local_port.py), [ChatGLM](../../../../tests/test_chatglm_local_port.py) |
| A successful load qualifies re-save | Retain architecture-owned FP32 parameters with a deliberately invalid mixed-dtype control. Re-save ported metadata with the required companion files and reload. Cover declared tokenizer aliases and each existing explicit remote-tokenizer consent combination. | [#6147](https://github.com/ModelCloud/GPTQModel/pull/6147), [#6117](https://github.com/ModelCloud/GPTQModel/pull/6117), [FP32 policy](../../../../tests/test_ported_fp32_load_6136.py), [config export](../../../../tests/test_config_only_export.py), [tokenizer selection](../../../../tests/test_package_tokenizer_selection.py) |

## Test the representation at each processor and writer boundary

AUTO_NF12 planning must leave decoded floatx sources dense until the processor
that actually runs a statistics forward. Assert the capture stand-in appears
at that forward, then verify a dense module reaches GPTQ preprocessing and
the result packs. For MXFP4 experts, label fused gate/up blocks, E8M0 scales
and biases independently so the decoded defused expert can be checked against
the released layout.
[#6146](https://github.com/ModelCloud/GPTQModel/pull/6146) ·
[source and processor tests](../../../../tests/test_auto_module_decoder.py)

Vary global AWQ widths 2/3/4/8 in selection tests and run supported non-4-bit
widths through public AUTO quantization. Inspect the packed class. Pair
shape-free asymmetric activation-order selection with concrete layer shapes.
For requantization placement, put the CPU `qzeros_format` marker before the
accelerator payload and inspect `qweight` after rehoming. Run capture with a
resident dispatch hook whose execution device differs from the capture request.
[#6143](https://github.com/ModelCloud/GPTQModel/pull/6143) ·
[#6142](https://github.com/ModelCloud/GPTQModel/pull/6142) ·
[AWQ selection](../../../../tests/test_awq_cuda_contract.py) ·
[placement](../../../../tests/test_requant_embed_endpoint_5116.py) ·
[Marlin selection](../../../../tests/kernels/test_marlin_asymmetric_selection_5737.py)

An alternate writer can bypass a correct serialization context. Exercise an
HF-bridge-loaded instance's `save_pretrained` after kernel initialization.
Checkpoint-layout kernels must emit reloadable words without runtime markers.
Runtime-layout kernels must raise the documented save error. Keep ordinary
read-only `state_dict()` and the native writer as controls. This specifically
guards the instance-level extension point introduced after removal of the
global wrapper.
[#6144](https://github.com/ModelCloud/GPTQModel/pull/6144) ·
[bridge and writer tests](../../../../tests/test_checkpoint_interop_4053.py)

## Make solver policy and failure exits observable

ADMM-Q's recent fixes cover factorization failure, nonfinite spectrum/iterate/
loss, an unqualified float64 solve device, low-bit MSE defaults, adaptive
penalties, rejected 2-bit configurations and public diagnostics. Inject each
numerical failure separately and compare fallback outputs with a fresh RTN
reference. Check unrelated exceptions and an invalid fallback. For thresholds,
use healthy, moderately and severely correlated Hessians, assert measured
condition numbers across `5e4` and `3e5`, and run at least two iterations.
Verify the selected schedule, stop reason and fallback reason survive through
the public result. Pair defaults with explicit overrides and config round trips.
[#6098](https://github.com/ModelCloud/GPTQModel/pull/6098) ·
[#6097](https://github.com/ModelCloud/GPTQModel/pull/6097) ·
[#6094](https://github.com/ModelCloud/GPTQModel/pull/6094) ·
[#6095](https://github.com/ModelCloud/GPTQModel/pull/6095) ·
[#6114](https://github.com/ModelCloud/GPTQModel/pull/6114) ·
[#6115](https://github.com/ModelCloud/GPTQModel/pull/6115) ·
[ADMM-Q tests](../../../../tests/test_admmq.py)

## Apply the existing lifecycle checks to the remaining fixes

| Boundary | Concrete test extension | Landed fixes and test families |
| --- | --- | --- |
| Load, trace and first forward | Align an FP32 adapter to a BF16 model before `optimize`, then execute generation. Use a separate eager control and preserve the forward-dtype lock. | [#6019](https://github.com/ModelCloud/GPTQModel/pull/6019), [#6124](https://github.com/ModelCloud/GPTQModel/pull/6124), [adapter tests](../../../../tests/test_adapter_forward_dtype_contract.py) |
| Compute versus storage placement | Give CPTC distinct source, encoder and storage devices and assert the registered payload device. For EBSS check the actual search forward and restoration after success/failure on the chosen quantization device. | [#6127](https://github.com/ModelCloud/GPTQModel/pull/6127), [#6123](https://github.com/ModelCloud/GPTQModel/pull/6123), [CPTC](../../../../tests/test_cptc.py), [EBSS](../../../../tests/test_moequant_ebss_device.py) |
| Packed expert routing | Request bypass on defused packed experts, make native top-k and bypass observably different, and check the softcapped gate computation. | [#6125](https://github.com/ModelCloud/GPTQModel/pull/6125), [expert tests](../../../../tests/test_inkling2_loop_packed_requant.py) |
| Package compatibility and tokenization | Load a legacy `quant_method="gguf"` config through package export. Compare tokenization with/without special tokens to derive BOS/EOS flags, including backend post-processors and an undeterminable case. | [#6081](https://github.com/ModelCloud/GPTQModel/pull/6081), [#6118](https://github.com/ModelCloud/GPTQModel/pull/6118), [package tests](../../../../tests/test_gguf_package.py) |
| Recurrent cache and interpreter lifetime | Exercise the visit cache's public layer view and reject live state shape changes. For prewarm use a subprocess that exits with work in flight, a caught-error exit and a wedged-worker timeout. | [#6089](https://github.com/ModelCloud/GPTQModel/pull/6089), [#6142](https://github.com/ModelCloud/GPTQModel/pull/6142), [#6145](https://github.com/ModelCloud/GPTQModel/pull/6145), [LeapQuant](../../../../tests/test_leapquant_state.py), [SenseNova](../../../../tests/test_sensenova_loop_local_port.py), [prewarm](../../../../tests/test_linalg_warmup.py) |

## Validation of the fixture guidance

Using the existing CPU environment with Python 3.14t, Torch 2.15.0+cpu and
Transformers 5.23.0, the current batch-identity test fails on revision
`43b999585` before #6140. The failure reaches `final_input` with a length-11
bank paired with length-9 hidden states. The same test passes on the audited
revision. The full Inkling plan, layer-type replay, ADMM-Q and AWQ CUDA-contract
files pass 173 tests with 12 skips for accelerator/NPU prerequisites.

Run those files from `tests/` with the project's supported dependency profile.

```bash
python -m pytest module_tree/test_inkling2_loop_plan.py test_layer_type_replay.py test_admmq.py test_awq_cuda_contract.py -q -rs --tb=short
```

The CPU ordering fixture verifies the interleaved sequence explicitly. Actual
multi-GPU scheduling and native kernel execution require separate qualification.
These guidance edits add test-design requirements. Runtime implementation of
the replay ownership proposal is follow-up work.
