import unittest

import torch.nn as nn

from gptqmodel.models.base import BaseQModel


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = DummyMLP()


class DummyModel:
    def __init__(self):
        self.layers = nn.ModuleList([DummyBlock()])


class TestAutoDetectModuleTree(unittest.TestCase):
    def test_layers_with_parents(self):
        model = DummyModel()
        base = BaseQModel.__new__(BaseQModel)
        tree = base._auto_detect_module_tree(model, quant_method="gptq")
        self.assertEqual(tree[0], "layers")
        self.assertEqual(tree[1], "#")
        mapping = tree[2]
        self.assertIn("self_attn", mapping)
        self.assertIn("mlp", mapping)
        self.assertSetEqual(set(mapping["self_attn"]), {"q_proj", "k_proj"})
        self.assertSetEqual(set(mapping["mlp"]), {"fc1", "fc2"})
