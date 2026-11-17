# Baichuan local-port validation

The original model-only qualification is retained below. The real-tokenizer
follow-up is recorded at the end of this page.

Environment: Python 3.14.6 free-threaded (`sys._is_gil_enabled() == False`),
Transformers 5.23.0, Torch 2.15.0+cpu. No trained weights were downloaded.

```bash
PYTHON_GIL=0 timeout 220 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest \
  tests/test_baichuan_local_port.py \
  tests/module_tree/test_baichuan_local_port.py \
  tests/test_baichuan_rotary_buffers.py -q
```

Result: **69 passed in 3.67s**, no skips. Covers all four variants with genuine
CPU forward/oracle comparisons in FP32/FP16/BF16, initial/extended Baichuan 1
RoPE arithmetic, two-layer cache continuation and beam reorder, repeated
generation, malformed-mask failure and warm-cache retry, loss/backward, exact
safetensors round trips and failed-load retry. Every variant also has all four
API trust/bypass combinations with malicious checkpoint `auto_map`. CPU/meta
state dicts and the actual wrapper's module plan cover all owners/roles;
AWQProcessor performs real plain quantization on all five decoder linears.

## Red/green evidence

Temporarily removed the definition's new LocalModelLoader, rotary-owner tree
entry and conservative AWQ override (the rest of the actual model port and
tests remained in place). Three tests failed on the intended behavior:

| Test | Observed red failure |
| --- | --- |
| `test_local_selection_ignores_malicious_auto_map_and_security_toggles` (false trust/false bypass) | `local_model_loader('baichuan')` was `None`, failing the definition-owned loader assertion before any checkpoint code could execute |
| `test_real_state_dict_owners_match_definition_and_checkpoint_layout[cpu-rope-False]` | Actual module blocks omitted `self_attn.rotary_emb:!` |
| `test_awq_plain_quantization_retains_all_five_real_projections[rope-True]` | Inherited grouping emitted four unqualified scale groups instead of the required empty list |

Command at the time (the first test subsequently gained four-layout
parameterization, so its current equivalent is suffixed
`[rope-False-False-False]`):

```bash
python -s -m pytest \
 'tests/test_baichuan_local_port.py::test_local_selection_ignores_malicious_auto_map_and_security_toggles[False-False]' \
 'tests/module_tree/test_baichuan_local_port.py::test_real_state_dict_owners_match_definition_and_checkpoint_layout[cpu-rope-False]' \
 'tests/module_tree/test_baichuan_local_port.py::test_awq_plain_quantization_retains_all_five_real_projections[rope-True]' -q
```

Result: **3 assertion failures in 2.48s**, no collection/import failures.
Restored all three changes; the complete 69-test command above passes.

## Initial shared fixture integration

The parent subsequently applied these original fixture proposals. The
tokenizer follow-up introduces no additional central fixture changes.

Only the parent edits shared fixtures. The unmodified shared registry suite
was run with:

```bash
PYTHON_GIL=0 timeout 220 /home/ubuntu/.venvs/gptqmodel-tf5/bin/python -s -m pytest \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_fusable_models.py::test_registry_has_explicit_fusion_qualification_for_every_alias -q
```

Result: **419 passed, 1 failed in 3.19s**, no skips. The sole failure is
`test_registered_model_plan_matches_quantization_consumers[baichuan]` because
the parent-owned inventory still lacks the new excluded rotary owner. The
same test function was also invoked successfully with the exact proposed
Baichuan inventory entry substituted **in memory only**. The model-specific
tests independently assert the complete new block expectation.

Replace `cases["BaiChuanQModel/gptq"]` in
`tests/module_tree/fixtures/registered_module_plans.json` with:

```json
{
  "layers": ["model.layers"],
  "blocks": [
    ["input_layernorm:!"],
    ["self_attn.W_pack", "self_attn.rotary_emb:!"],
    ["self_attn.o_proj"],
    ["post_attention_layernorm:!"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"]
  ]
}
```

Update the existing Baichuan exclusion reason in
`tests/module_tree/fixtures/fusable_models.json` (under `models.baichuan`) to:

```json
{
  "definition": "BaiChuanQModel",
  "reason": "Package-owned fused-QKV Baichuan implementation; fusion executor remains unqualified."
}
```

The central fusion coverage documentation should use the same reason. Central
README and loader logic-flow docs should link to this port's `PROVENANCE.md`
and include its definition-owned selection diagram. Those shared changes are
left to the parent as assigned.

## Limits

`tests/models/test_baichuan.py::TestBaiChuan::test_baichuan` was migrated to
`TRUST_REMOTE_CODE=False` but not executed: its trained checkpoint and a GPU
are absent. Trained payload key audits are complete only for the two 13B
checkpoint indexes (283 keys each); the two monolithic 7B payloads were not
downloaded. Their rotary ownership follows inspected upstream source.

No GPU packing/kernel or multi-GPU GIL=0 concurrency
claim is made. Static/offloaded/tuple caches and upstream helper `chat`/
quantizer APIs are not advertised; standard GenerationMixin + DynamicCache
is the supported generation path. No AWQ fold/search/replay or fusion
qualification is claimed. No shared AWQ algorithm, loader, writer, security,
registry or CI workflow file was changed.

## Real tokenizer follow-up

Pinned sources, data hashes and the fixed Transformers 4.45.2 / SentencePiece
source-oracle contract are recorded in `PROVENANCE.md` and
`tokenizer_contracts.json`. `README.md` contains the shipped selection/failure
flow and before/after table. The definition now binds the package tokenizer
directly, including original `BaiChuanTokenizer` spelling compatibility.

Red/green observations:

- Parent `0fdfd2396`, before the tokenizer class/binding: all four real public
  checkpoint loads fail because the checkpoint tokenizer class is unavailable
  natively. Adding the source-owned class/binding makes all four pass.
- Transformers 5's generic serializer removes `add_bos_token` and
  `add_eos_token`. Before the owned serializer, eight pair/public roundtrip
  cases fail; preserving the live flags restores exact masks and IDs.
- Four direct-constructor regressions using native `AddedToken` objects fail
  before the owned initializer adaptation. HF5 initializes their string forms
  instead of HF4's extended objects, discarding `single_word`/`normalized`.
  Registering the original native objects at their real tokenizer IDs fixes
  all four controls and the exact real 13B source IDs. There is no metadata
  parsing or corrective loader shim in GPT-QModel; reusable JSON preparation
  remains upstream in Tokenicer.

The final tokenizer suite covers all four real layouts, nine hostile
trust/bypass combinations per layout, fixed multilingual/whitespace/special
token IDs/decodes, native AddedToken flags, all four BOS/EOS pair-mask and
save/reload combinations, live flag mutation, save without the original model
file, source role-token chat prompts and an explicit supplied template,
left/right batch padding, public dense safetensors re-save, and twelve
missing/malformed asset repair/retry cases. No default HF chat template is
invented. Tests load only small data assets plus random local safetensors.

Four public plain-AWQ cases additionally calibrate real source strings with
`fallback=None`, pack all five actual decoder linears, preserve excluded
tensors, infer/generate and perform two exact public save/load cycles of
packed/dense state plus the real tokenizer. Live floating state is explicitly
cast to FP16 before save to match reload dtype; no source-unspecified strict
FP32 policy is invented for Baichuan 1's narrowable `inv_freq`.

The required existing AWQ/model-plan baseline passes **574 tests, 3 skips**
in 8.77s. Skips are the existing absent remote-family Deci/ExaOne/Nanbeige
variable-length fixtures, not Baichuan. The broad loader/processor/config/trust
family has one existing skip for absent `peft.tuners.lora.awq`.

Final CPU environment: Python 3.14.6t with `PYTHON_GIL=0`, Transformers
5.23.0, Torch 2.15.0+cpu and SentencePiece 0.2.1. The following command gives
**301 passed, 1 skipped** with released Tokenicer 0.0.17 (44.81s), and the
same result with unreleased Tokenicer 0.0.19 candidate
`c14e1c8c062c7b7cc9a981054366f98bde015fa1` (45.45s):

```bash
PYTHON_GIL=0 python -m pytest -q \
  tests/test_baichuan_tokenizer.py \
  tests/test_baichuan_local_port.py \
  tests/module_tree/test_baichuan_local_port.py \
  tests/test_baichuan_rotary_buffers.py \
  tests/test_local_tokenizer_loading.py \
  tests/test_local_model_loading.py \
  tests/test_local_processor_loading.py \
  tests/test_hf_config_compat.py \
  tests/test_trust_remote_code_isolation.py
```

The owned tokenizer file contributes **89 passed, zero skips** to this run.
Baichuan therefore does not require the candidate's new generic metadata API;
its compatibility adaptations remain in its source-owned class.

Required AWQ/model-plan baseline command:

```bash
PYTHON_GIL=0 python -m pytest -q \
  tests/test_awq_scale_search_safety.py tests/test_awq_moe_router_scaling.py \
  tests/test_awq_empty_groups.py tests/test_awq_position_kwargs.py \
  tests/test_awq_scale_search_memory.py tests/test_bloom_awq_replay.py \
  tests/module_tree/test_awq_relationships.py \
  tests/module_tree/test_bloom_awq_relationships.py \
  tests/module_tree/test_registered_module_plans.py \
  tests/module_tree/test_definition_subset_ordering.py
```

`format/check.sh` and full `ruff check gptqmodel tests setup.py` pass using
pinned Ruff 0.14.2. Port/tests compile, `git diff --check` passes, and the
Mermaid blocks have balanced fences and recognized flowchart types. No
new skip, dependency version change or shared source/fixture edit is needed.

GPU kernels, trained-checkpoint quality, interactive history-window/streaming
helpers, threading and extended-context equivalence when an FP32 checkpoint
buffer is implicitly narrowed to FP16 remain outside this follow-up's claims.
All newly written checkpoint tensor payloads are safetensors; tokenizer models
are the original SentencePiece data, never checkpoint Python.
