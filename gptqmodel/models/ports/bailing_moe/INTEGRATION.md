<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bailing integration: parent-owned updates

No shared registry, loader/security file or central documentation is changed by
this port. The integrator must apply the following exact expectation changes.

## Registry plan entry

Replace `cases["BailingMoeQModel/gptq"]` in
`tests/module_tree/fixtures/registered_module_plans.json` with the entry below.
Both existing `models.bailing_moe` and `models.bailing_hybrid` remain
`"BailingMoeQModel"`. `attention.q_proj` is the legitimate upstream
`q_lora_rank=None` MLA variant, not the native `self_attn.q_proj` branch.
MTP envelope tensors are checkpoint-owned; only `eh_proj` is quantizable.

```json
{
  "layers": ["model.layers"],
  "blocks": [
    ["input_layernorm:!"],
    ["enorm:!"],
    ["hnorm:!"],
    ["eh_proj"],
    ["final_layernorm:!"],
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.g_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.forget_gate.f_proj", "self_attn.b_proj", "self_attn.conv1d:!"],
    ["self_attn.kv_b_proj"],
    ["self_attn.o_proj"],
    ["attention.q_proj", "attention.query_key_value", "attention.q_a_proj", "attention.kv_a_proj_with_mqa", "attention.g_proj", "attention.query_layernorm:!", "attention.key_layernorm:!", "attention.g_norm:!", "attention.q_a_layernorm:!", "attention.kv_a_layernorm:!"],
    ["attention.q_b_proj", "attention.kv_b_proj"],
    ["attention.dense"],
    ["post_attention_layernorm:!"],
    ["mlp.gate:!"],
    ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
    ["mlp.shared_experts.down_proj"],
    ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"],
    ["mlp.experts.{expert_index}.down_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"]
  ]
}
```

Keep both aliases excluded from fusion. Optionally modernize their reason in
`fixtures/fusable_models.json` to: `Package-owned dense/MLA/recurrent/MoE layouts;
no qualified fused executor.` No fusion eligibility is added.

## Existing module-tree tests

* `test_bailing_moe_shared_experts_4325.py::test_shared_expert_down_proj_is_its_own_subset`:
  prepend `"attention.q_proj"` to the exact first `attention` projection block.
  Do not alter the shared-expert down-projection separation assertions.
* `test_definition_subset_ordering.py::test_issue_4372_declarations_follow_forward_order`:
  only the `bailing_moe` case changes to `self_attn.kv_a_proj_with_mqa: 1`,
  `self_attn.kv_b_proj: 2`, `self_attn.o_proj: 3` (absolute block indices).
  `eh_proj` now owns block 0. Relative producer/consumer order is unchanged.

## Existing loader/security tests

These cases explicitly assumed no Bailing port existed. Change their trust
result to `False` for every API trust/bypass combination, retain layout-marker
detection assertions, and test package-owned config/model selection where
appropriate. Preserve unrelated-model rejection and native-layout coverage.

* `test_ling_remote_code_4373.py::test_bailing_hybrid_remote_code_layout_respects_explicit_trust`
* `test_ling_remote_code_4373.py::test_bailing_hybrid_remote_code_layout_stays_off_without_trust`
* `test_ling_remote_code_4373.py::test_bailing_hybrid_mla_marker_with_same_named_remote_class`
* `test_local_model_loading.py::test_unported_layout_fails_instead_of_loading_incompatible_native_model`
* `test_remote_code_layout_marker_4453.py::test_ling_mla_marker_requires_security_bypass`
* `test_remote_code_layout_marker_4453.py::test_layout_marker_without_remote_model_code_still_rejects_incompatible_native`

## Central documentation

Link this port's [reference page](README.md) from the central Transformers 5 local
model guide/README. Include its selection/cache/replay Mermaid flow and the
before/after table in the draft PR description. The port-local location respects
the integrator's exclusive ownership of central `docs/` and README files.
