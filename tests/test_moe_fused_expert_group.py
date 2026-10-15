# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""`dynamic` overrides must match the modules inference engines fuse.

Inference engines (vLLM, SGLang) merge every routed expert of one decoder layer
into a single module, `gate_proj`/`up_proj` into `gate_up_proj`, and
`q_proj`/`k_proj`/`v_proj` into `qkv_proj`. They resolve `dynamic` from the
merged module name only, so a rule that matches just part of a merged module
saves a checkpoint those engines cannot load (issue #1455). GPT-QModel warns
about that mismatch before quantization starts.
"""

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# -- end do not touch

import collections  # noqa: E402
import tempfile  # noqa: E402
import unittest.mock  # noqa: E402

import pcre  # noqa: E402
import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from safetensors import safe_open  # noqa: E402

from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.config import RTNConfig  # noqa: E402
from gptqmodel.utils import model as model_utils  # noqa: E402
from gptqmodel.utils.model import (  # noqa: E402
    fused_expert_group_members,
    fused_expert_group_name,
    fused_unit_check_module_names,
    fused_unit_dynamic_conflicts,
    fused_unit_members,
    warn_fused_unit_dynamic_conflicts,
)


MODEL_ID = "/monster/data/model/Qwen3-30B-A3B-layers-1"
GROUP = "model.layers.0.mlp.experts"


def read_expert_group_states(model_dir):
    """Return `{fused_group: {quantized flags}}` for every routed expert in `model_dir`."""

    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as handle:
        leaves = collections.defaultdict(dict)
        for key in handle.keys():
            parts = key.split(".")
            leaves[".".join(parts[:-1])][parts[-1]] = str(handle.get_slice(key).get_dtype())

    groups = collections.defaultdict(set)
    for module_name, tensors in leaves.items():
        group = fused_expert_group_name(module_name)
        if group is not None:
            groups[group].add("qweight" in tensors)
    return groups


def dense_layer_modules(prefix="model.layers.0"):
    return [
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.k_proj",
        f"{prefix}.self_attn.v_proj",
        f"{prefix}.self_attn.o_proj",
        f"{prefix}.mlp.gate_proj",
        f"{prefix}.mlp.up_proj",
        f"{prefix}.mlp.down_proj",
    ]


def expert_modules(prefix=GROUP, experts=(0, 1)):
    return [
        f"{prefix}.{expert}.{projection}"
        for expert in experts
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]


class TestFusedUnitMembers(unittest.TestCase):
    def test_expert_group_extraction(self):
        self.assertEqual(fused_expert_group_name("model.layers.0.mlp.experts.3.down_proj"), GROUP)
        self.assertEqual(fused_expert_group_name("mlp.experts.10.gate_proj"), "mlp.experts")
        # Expert containers, shared experts and dense modules have no fused group.
        self.assertIsNone(fused_expert_group_name("model.layers.0.mlp.experts"))
        self.assertIsNone(fused_expert_group_name("model.layers.0.mlp.experts.3"))
        self.assertIsNone(fused_expert_group_name("model.layers.0.mlp.shared_experts.down_proj"))
        self.assertIsNone(fused_expert_group_name("model.layers.0.mlp.shared_expert.down_proj"))
        self.assertIsNone(fused_expert_group_name("model.layers.0.self_attn.q_proj"))
        self.assertIsNone(fused_expert_group_name("lm_head"))

    def test_group_members(self):
        members = [
            "mlp.experts.0.gate_proj",
            "mlp.experts.1.gate_proj",
            "mlp.experts.0.down_proj",
            "mlp.down_proj",
            "mlp.shared_experts.gate_proj",
        ]
        self.assertEqual(
            fused_expert_group_members(members),
            {
                "mlp.experts": [
                    "mlp.experts.0.gate_proj",
                    "mlp.experts.1.gate_proj",
                    "mlp.experts.0.down_proj",
                ]
            },
        )

    def test_fused_unit_members_cover_experts_and_dense_projections(self):
        units = fused_unit_members(expert_modules() + dense_layer_modules())
        self.assertEqual(
            sorted(units),
            [
                "model.layers.0.mlp.experts",
                "model.layers.0.mlp.gate_up_proj",
                "model.layers.0.self_attn.qkv_proj",
            ],
        )
        self.assertEqual(
            units["model.layers.0.mlp.gate_up_proj"],
            ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
        )
        self.assertEqual(
            units["model.layers.0.self_attn.qkv_proj"],
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ],
        )

    def test_unpaired_shards_are_not_fused_units(self):
        # `q_proj` alone (no k/v), `gate_proj` alone (no up), `down_proj` and
        # `o_proj` are never merged by the engines.
        units = fused_unit_members([
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.self_proj",
        ])
        self.assertEqual(units, {})

    def test_shared_experts_are_not_part_of_the_routed_group(self):
        units = fused_unit_members([
            "model.layers.0.mlp.shared_experts.gate_proj",
            "model.layers.0.moe.shared_experts.down_proj",
        ])
        self.assertEqual(units, {})

    def test_expert_group_members_exclude_shared_experts(self):
        units = fused_unit_members(expert_modules() + [
            "model.layers.0.moe.shared_experts.gate_proj",
            "model.layers.1.mlp.experts.0.down_proj",
        ])
        self.assertEqual(sorted(units), ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"])
        self.assertNotIn(
            "model.layers.1.mlp.experts.0.down_proj",
            units["model.layers.0.mlp.experts"],
        )


class TestFusedUnitDynamicConflicts(unittest.TestCase):
    MODULES = expert_modules() + dense_layer_modules()

    def conflicts(self, dynamic, modules=None, **kwargs):
        qcfg = QuantizeConfig(bits=4, group_size=128, dynamic=dynamic, **kwargs)
        return fused_unit_dynamic_conflicts(modules if modules is not None else self.MODULES, qcfg)

    def units(self, conflicts):
        return sorted(conflict.unit for conflict in conflicts)

    @staticmethod
    def engine_config(**overrides):
        config = {"bits": 4, "group_size": 128, "desc_act": False, "sym": True}
        config.update(overrides)
        return tuple(config.items())

    def test_no_dynamic_has_no_conflicts(self):
        self.assertEqual(self.conflicts(None), [])
        self.assertEqual(self.conflicts({}), [])

    def test_partial_negative_rule_conflicts_with_the_expert_group(self):
        # The rule from issue #1455: `-:.*down.*` skips `down_proj`, which the
        # engines merge with `gate_proj`/`up_proj` into `<layer>.mlp.experts`.
        conflicts = self.conflicts({r"-:.*down.*": {}})
        self.assertEqual(self.units(conflicts), [GROUP])
        self.assertEqual(conflicts[0].engine_config, self.engine_config())
        divergent = [
            member for member, config in conflicts[0].member_configs if config != conflicts[0].engine_config
        ]
        self.assertEqual(
            divergent,
            [f"{GROUP}.0.down_proj", f"{GROUP}.1.down_proj"],
        )
        # Dense `down_proj`/`o_proj` are not fused with anything, so the same
        # rule stays supported for them.
        self.assertNotIn(
            "model.layers.0.mlp.down_proj",
            [member for conflict in conflicts for member, _ in conflict.member_configs],
        )

    def test_negative_rule_on_the_merged_module_is_supported(self):
        self.assertEqual(self.conflicts({r"-:.*mlp\.experts.*": {}}), [])
        self.assertEqual(self.conflicts({rf"-:^{GROUP}(\.|$)": {}}), [])

    def test_positive_rule_on_the_merged_module_is_supported(self):
        self.assertEqual(self.conflicts({r"+:.*mlp\.experts.*": {"bits": 8, "group_size": 32}}), [])

    def test_partial_positive_rule_conflicts_with_experts_and_gate_up(self):
        conflicts = self.conflicts({r"+:.*gate.*": {"bits": 8}})
        self.assertEqual(
            self.units(conflicts),
            [GROUP, "model.layers.0.mlp.gate_up_proj"],
        )

    def test_uniform_member_override_still_conflicts(self):
        # Every expert projection gets the same override, but the engines read
        # `dynamic` for the merged module name, which the rule does not match.
        conflicts = self.conflicts({r"+:.*experts\.\d+\..*": {"bits": 8, "group_size": 32}})
        self.assertEqual(self.units(conflicts), [GROUP])
        self.assertEqual(conflicts[0].engine_config, self.engine_config())
        self.assertEqual(
            [member for member, _ in conflicts[0].member_configs],
            expert_modules(),
        )
        for _, config in conflicts[0].member_configs:
            self.assertEqual(config, self.engine_config(bits=8, group_size=32))

    def test_rule_matching_only_the_merged_name_still_conflicts(self):
        # `qkv_proj` is not a prefix of `q_proj`/`k_proj`/`v_proj`: the engines
        # would skip the merged module while GPT-QModel still quantizes the
        # shards, so the rule must match both.
        conflicts = self.conflicts({r"-:.*qkv_proj.*": {}}, modules=dense_layer_modules())
        self.assertEqual(self.units(conflicts), ["model.layers.0.self_attn.qkv_proj"])
        self.assertEqual(
            conflicts[0].suggested_pattern(),
            r"model\.layers\.0\.self_attn\.(q_proj|k_proj|v_proj|qkv_proj)$",
        )

    def test_rule_matching_merged_name_and_shards_is_supported(self):
        self.assertEqual(self.conflicts({r"+:.*mlp\.experts.*": {"bits": 8, "group_size": 32}}), [])
        self.assertEqual(
            self.conflicts({r"-:.*(q_proj|k_proj|v_proj|qkv_proj).*": {}}, modules=dense_layer_modules()),
            [],
        )
        self.assertEqual(
            self.conflicts({r"-:.*(gate_proj|up_proj|gate_up_proj).*": {}}, modules=dense_layer_modules()),
            [],
        )

    def test_suggested_pattern_covers_merged_module_and_shards(self):
        expert_conflict = {
            conflict.unit: conflict for conflict in self.conflicts({r"-:.*down.*": {}})
        }[GROUP]
        self.assertEqual(expert_conflict.suggested_pattern(), rf"{pcre.escape(GROUP)}(\.|$)")

        dense_conflict = {
            conflict.unit: conflict
            for conflict in self.conflicts({r"+:.*gate.*": {"bits": 8}}, modules=dense_layer_modules())
        }["model.layers.0.mlp.gate_up_proj"]
        self.assertEqual(
            dense_conflict.suggested_pattern(),
            r"model\.layers\.0\.mlp\.(gate_proj|up_proj|gate_up_proj)$",
        )

    def test_partial_rule_on_attention_shards_conflicts_with_qkv(self):
        conflicts = self.conflicts({r"-:.*k_proj.*": {}})
        self.assertEqual(self.units(conflicts), ["model.layers.0.self_attn.qkv_proj"])
        self.assertEqual(
            [
                member
                for member, config in conflicts[0].member_configs
                if config == ("excluded",)
            ],
            ["model.layers.0.self_attn.k_proj"],
        )

    def test_base_config_fields_not_read_by_engines_do_not_conflict(self):
        # `mse`/`damp_percent`/... are baked into the saved weights and are not
        # read by the engines, so a per-member rule for them stays supported.
        conflicts = self.conflicts({r"+:.*experts\.0\..*": {"mse": 0.1, "damp_percent": 0.05}})
        self.assertEqual(conflicts, [])

    def test_engine_fields_per_member_conflict(self):
        conflicts = self.conflicts({r"+:.*experts\.0\.gate_proj": {"bits": 8}})
        self.assertEqual(self.units(conflicts), [GROUP])

    def test_warn_reports_conflicts_once_with_guidance(self):
        qcfg = QuantizeConfig(bits=4, group_size=128, dynamic={r"-:.*down.*": {}})
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            conflicts = warn_fused_unit_dynamic_conflicts(self.MODULES, qcfg)
        self.assertEqual(self.units(conflicts), [GROUP])
        warn.assert_called_once()
        message = warn.call_args[0][0]
        # The warning names the offending `dynamic` rule and the fused module.
        self.assertIn("vLLM/SGLang", message)
        self.assertIn(r"`-:.*down.*`", message)
        self.assertIn(GROUP, message)
        self.assertIn(rf"+:{pcre.escape(GROUP)}(\.|$)", message)
        # Kept short: no per-member dump and no multi-line report.
        self.assertNotIn("\n", message)
        self.assertLess(len(message), 500)

    def test_warn_blames_the_rule_that_matches_only_the_fused_name(self):
        qcfg = QuantizeConfig(bits=4, group_size=128, dynamic={r"-:.*qkv_proj.*": {}})
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            warn_fused_unit_dynamic_conflicts(dense_layer_modules(), qcfg)
        message = warn.call_args[0][0]
        self.assertIn(r"`-:.*qkv_proj.*`", message)
        self.assertIn("model.layers.0.self_attn.qkv_proj", message)

    def test_warn_names_every_partial_rule_in_effective_order(self):
        # `QuantizeConfig` normalizes `dynamic` so negative rules come first;
        # the warning must follow that effective order.
        qcfg = QuantizeConfig(
            bits=4,
            group_size=128,
            dynamic={
                r"+:.*gate.*": {"bits": 8},
                r"-:.*down.*": {},
            },
        )
        self.assertEqual(list(qcfg.dynamic.keys()), [r"-:.*down.*", r"+:.*gate.*"])
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            warn_fused_unit_dynamic_conflicts(self.MODULES, qcfg)
        message = warn.call_args[0][0]
        self.assertIn(r"`-:.*down.*`", message)
        self.assertIn(r"`+:.*gate.*`", message)
        self.assertLess(message.index(r"`-:.*down.*`"), message.index(r"`+:.*gate.*`"))

    def test_warn_is_silent_when_every_merged_module_agrees(self):
        qcfg = QuantizeConfig(bits=4, group_size=128, dynamic={r"-:.*mlp\.experts.*": {}})
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            conflicts = warn_fused_unit_dynamic_conflicts(self.MODULES, qcfg)
        self.assertEqual(conflicts, [])
        warn.assert_not_called()

    def test_warn_caps_the_reported_rules(self):
        dynamic = {
            rf"+:.*\.{index}\..*gate_proj": {"bits": 8}
            for index in range(5)
        }
        modules = [
            f"model.layers.{index}.mlp.{projection}"
            for index in range(5)
            for projection in ("gate_proj", "up_proj")
        ]
        qcfg = QuantizeConfig(bits=4, group_size=128, dynamic=dynamic)
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            warn_fused_unit_dynamic_conflicts(modules, qcfg)
        message = warn.call_args[0][0]
        self.assertIn("(+2 more)", message)

    def test_check_names_are_scoped_to_decoder_layers(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].mlp = torch.nn.Module()
        model.model.layers[0].mlp.gate_proj = torch.nn.Linear(2, 2)
        model.model.layers[0].mlp.up_proj = torch.nn.Linear(2, 2)
        model.vision = torch.nn.Module()
        model.vision.blocks = torch.nn.ModuleList([torch.nn.Module()])
        model.vision.blocks[0].mlp = torch.nn.Module()
        model.vision.blocks[0].mlp.gate_proj = torch.nn.Linear(2, 2)
        model.vision.blocks[0].mlp.up_proj = torch.nn.Linear(2, 2)

        names = fused_unit_check_module_names(model, ["model.layers"])
        self.assertIn("model.layers.0.mlp.gate_proj", names)
        self.assertNotIn("vision.blocks.0.mlp.gate_proj", names)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class TestFusedExpertGroupMoEQuantization(ModelTest):
    """End-to-end: `dynamic` warnings and fused-expert-group invariants."""

    NATIVE_MODEL_ID = MODEL_ID
    DATASET_SIZE = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.isdir(cls.NATIVE_MODEL_ID):
            raise unittest.SkipTest(f"missing local model path: {cls.NATIVE_MODEL_ID}")
        cls.tokenizer = cls.load_tokenizer(cls.NATIVE_MODEL_ID)
        cls.calibration_dataset = cls.load_dataset(cls.tokenizer, rows=cls.DATASET_SIZE)

    def quantize(self, quantize_config):
        model = GPTQModel.load(self.NATIVE_MODEL_ID, quantize_config=quantize_config)
        model.quantize(self.calibration_dataset, batch_size=1)
        return model

    def assert_no_fused_engine_warning(self, warn):
        messages = [call.args[0] for call in warn.call_args_list]
        self.assertFalse(
            any("vLLM/SGLang" in message for message in messages),
            messages,
        )

    def assert_group_is_uniform(self, model_dir):
        groups = read_expert_group_states(model_dir)
        self.assertTrue(groups, "expected routed experts in the checkpoint")
        for group, states in groups.items():
            self.assertEqual(len(states), 1, f"`{group}` mixes quantized and float experts: {states}")
        return groups

    def test_partial_negative_rule_warns_before_quantizing(self):
        # The rule from issue #1455: skip every `down` module. `down_proj` is
        # merged with `gate_proj`/`up_proj` into `<layer>.mlp.experts`, so the
        # engines would resolve the base config for the merged module while the
        # checkpoint keeps `down_proj` in its source dtype.
        quantize_config = QuantizeConfig(bits=4, group_size=128, dynamic={r"-:.*down.*": {}})
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            model = self.quantize(quantize_config)

        self.assertTrue(model._fused_unit_dynamic_warned)
        messages = [call.args[0] for call in warn.call_args_list]
        warnings = [message for message in messages if "vLLM/SGLang" in message]
        self.assertEqual(len(warnings), 1, messages)
        self.assertIn(GROUP, warnings[0])
        self.assertIn(r"`-:.*down.*`", warnings[0])

    def test_rule_matching_the_merged_module_needs_no_warning(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            dynamic={rf"-:^{pcre.escape(GROUP)}(\.|$)": {}},
        )
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            model = self.quantize(quantize_config)
        self.assert_no_fused_engine_warning(warn)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            del model
            torch.cuda.empty_cache()
            self.assert_group_is_uniform(tmp_dir)

    def test_weight_only_partial_rule_warns_too(self):
        # Weight-only (RTN) checkpoints are exported in the GPTQ format and read
        # by the same fused-MoE engine path, so the warning applies there too.
        quantize_config = RTNConfig(bits=4, group_size=128, dynamic={r"-:.*down.*": {}})
        with unittest.mock.patch.object(model_utils.log, "warn") as warn:
            self.quantize(quantize_config)

        messages = [call.args[0] for call in warn.call_args_list]
        self.assertTrue(
            any("vLLM/SGLang" in message and GROUP in message for message in messages),
            messages,
        )


if __name__ == "__main__":
    unittest.main()
