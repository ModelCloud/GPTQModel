# Baichuan local-port validation

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

## Shared fixture integration

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

No GPU packing, quantized kernel, tokenizer or multi-GPU GIL=0 concurrency
claim is made. Static/offloaded/tuple caches and upstream helper `chat`/
quantizer APIs are not advertised; standard GenerationMixin + DynamicCache
is the supported generation path. No AWQ fold/search/replay or fusion
qualification is claimed. No shared AWQ algorithm, loader, writer, security,
registry or CI workflow file was changed.
