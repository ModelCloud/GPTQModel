# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for :moe flag parsing and MoE module detection.
"""


from gptqmodel.models.base import MOE_FLAG, BaseQModel


class TestMoEFlagParsing:
    """Tests for :moe flag parsing utilities."""

    def test_parse_module_flags_basic(self):
        """Test basic flag parsing."""
        name, flags = BaseQModel._parse_module_flags("gate_proj")
        assert name == "gate_proj"
        assert flags == []

    def test_parse_module_flags_with_moe(self):
        """Test parsing with :moe flag."""
        name, flags = BaseQModel._parse_module_flags("mlp:moe")
        assert name == "mlp"
        assert "moe" in flags

    def test_parse_module_flags_combined(self):
        """Test parsing with multiple flags."""
        name, flags = BaseQModel._parse_module_flags("gate:moe:!")
        assert name == "gate"
        assert "moe" in flags
        assert "!" in flags

    def test_has_moe_flag(self):
        """Test MoE flag detection."""
        assert BaseQModel.has_moe_flag("mlp:moe") is True
        assert BaseQModel.has_moe_flag("experts:moe") is True
        assert BaseQModel.has_moe_flag("gate:moe:!") is True
        assert BaseQModel.has_moe_flag("gate_proj") is False
        assert BaseQModel.has_moe_flag("gate_proj:!") is False

    def test_has_moe_flag_non_string(self):
        """Test MoE flag detection with non-string input."""
        assert BaseQModel.has_moe_flag(None) is False
        assert BaseQModel.has_moe_flag(123) is False
        assert BaseQModel.has_moe_flag({}) is False


class TestMockMoEModel:
    """Mock MoE model for testing."""

    @staticmethod
    def create_mock_model_class(module_tree):
        """Create a mock model class with given module_tree."""

        class MockMoEModel(BaseQModel):
            pass

        MockMoEModel.module_tree = module_tree
        return MockMoEModel


class TestMoEModuleDetection:
    """Tests for MoE module detection from module_tree."""

    def test_collect_moe_modules_simple(self):
        """Test MoE module collection with simple tree."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "mlp:moe": {
                    "gate_proj": ("gate_proj:0",),
                    "up_proj": ("up_proj:0",),
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_modules = MockModel.get_moe_modules()

        assert "mlp" in moe_modules

    def test_collect_moe_modules_nested(self):
        """Test MoE module collection with nested structure (experts + shared_experts)."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "mlp:moe": {
                    "shared_experts": {
                        "gate_proj": ("gate_proj:0",),
                    },
                    "experts": {
                        "#": ("gate_proj:0", "up_proj:0"),
                    },
                    "gate:!": ("gate:!",),
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_modules = MockModel.get_moe_modules()

        assert "mlp" in moe_modules
        assert "mlp.shared_experts" in moe_modules
        assert "mlp.experts" in moe_modules
        assert "mlp.gate" in moe_modules

    def test_is_moe_module_detection(self):
        """Test is_moe_module() detection."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": {
                    "q_proj": ("q_proj:0",),
                },
                "mlp:moe": {
                    "experts": {
                        "#": ("gate_proj:0",),
                    },
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)

        # MoE modules
        assert MockModel.is_moe_module("model.layers.0.mlp") is True
        assert MockModel.is_moe_module("model.layers.0.mlp.experts.5.gate_proj") is True
        assert MockModel.is_moe_module("layers.3.mlp.experts.10") is True

        # Non-MoE modules
        assert MockModel.is_moe_module("model.layers.0.self_attn.q_proj") is False
        assert MockModel.is_moe_module("model.layers.0.self_attn") is False

    def test_backward_compatibility_no_moe_flags(self):
        """Test that models without :moe flags still work."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": {
                    "q_proj": ("q_proj:0",),
                    "k_proj": ("k_proj:0",),
                },
                "mlp": {
                    "gate_proj": ("gate_proj:0",),
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_modules = MockModel.get_moe_modules()

        assert len(moe_modules) == 0
        assert MockModel.is_moe_module("model.layers.0.mlp") is False


class TestMoEModuleName:
    """Tests for MoE module name extraction."""

    def test_get_moe_module_name_glm4(self):
        """Test get_moe_module_name() with GLM-4 style tree."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": ("q_proj:0",),
                "mlp:moe": {
                    "shared_experts": {"gate_proj": ("gate_proj:0",)},
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_module_name = MockModel.get_moe_module_name()

        assert moe_module_name == "mlp"

    def test_get_moe_module_name_minimax(self):
        """Test get_moe_module_name() with MiniMax-M2 style tree."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": ("q_proj:0",),
                "block_sparse_moe:moe": {
                    "gate": ("gate:!",),
                    "experts": {"#": ("w1:0",)},
                },
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_module_name = MockModel.get_moe_module_name()

        assert moe_module_name == "block_sparse_moe"

    def test_get_moe_module_name_no_moe(self):
        """Test get_moe_module_name() with no MoE flag."""
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": ("q_proj:0",),
                "mlp": {"gate_proj": ("gate_proj:0",)},
            }
        ]

        MockModel = TestMockMoEModel.create_mock_model_class(module_tree)
        moe_module_name = MockModel.get_moe_module_name()

        assert moe_module_name is None

    def test_get_moe_module_name_none_tree(self):
        """Test get_moe_module_name() with None module_tree."""
        MockModel = TestMockMoEModel.create_mock_model_class(None)
        moe_module_name = MockModel.get_moe_module_name()

        assert moe_module_name is None


class TestMoEFlagConstant:
    """Test MOE_FLAG constant."""

    def test_moe_flag_value(self):
        """Test that MOE_FLAG has the correct value."""
        assert MOE_FLAG == ":moe"
