# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Core dependency isolation and actual optional-vision failure boundaries."""

import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
ABSENT_VISION = """
import importlib.abc
import importlib.util
import sys

blocked = set(sys.argv[1].split(','))
original_find_spec = importlib.util.find_spec

def find_spec(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        return None
    return original_find_spec(name, *args, **kwargs)

class MissingVision(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in blocked:
            raise ModuleNotFoundError(f'No module named {root!r}', name=root)

importlib.util.find_spec = find_spec
sys.meta_path.insert(0, MissingVision())
"""


def _run(source, blocked="timm,torchvision"):
    result = subprocess.run(
        [sys.executable, "-s", "-c", ABSENT_VISION + "\n" + textwrap.dedent(source), blocked],
        cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"}, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_core_requirements_do_not_install_vision_backends():
    core = [Requirement(line.strip()) for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    assert not {"timm", "torchvision"}.intersection(req.name.lower() for req in core)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "timm>=1.0.22" in project["project"]["optional-dependencies"]["vision"]


def test_core_import_and_real_cpu_text_forward_without_vision_backends():
    _run("""
        import torch
        from gptqmodel import GPTQModel
        from gptqmodel.models.definitions.internlm import InternLMQModel
        from gptqmodel.models.ports.internlm.configuration_internlm import InternLMConfig

        torch.set_num_threads(1)
        config = InternLMConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, num_attention_heads=2,
                                max_position_embeddings=32, attn_implementation='eager')
        model = InternLMQModel.loader.from_config(config).eval()
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
        assert logits.shape == (1, 3, 32)
        assert torch.isfinite(logits).all()
        assert not blocked.intersection(name.split('.')[0] for name in sys.modules)
    """)


def test_lazy_port_imports_do_not_require_vision_backends():
    _run("""
        import importlib
        for suffix in (
            'deepseek_vl_v2.processing_deepseek_vl_v2',
            'interns1.modeling_interns1',
            'minicpmv.modeling_minicpmv',
            'llava_qwen2.vision',
            'openvla.modeling_prismatic',
            'openvla.processing_prismatic',
            'zdtaichu5.modeling',
        ):
            importlib.import_module('gptqmodel.models.ports.' + suffix)
        assert not blocked.intersection(name.split('.')[0] for name in sys.modules)
    """)


BOUNDARIES = [
    ("deepseek_vl_v2.siglip_vit", "import importlib; importlib.import_module('gptqmodel.models.ports.deepseek_vl_v2.siglip_vit')"),
    ("llava_qwen2.fastvit", "from gptqmodel.models.ports.llava_qwen2.vision import MCi; MCi({})"),
    ("openvla.backbone", "from gptqmodel.models.ports.openvla.modeling_prismatic import PrismaticVisionBackbone; PrismaticVisionBackbone(None)"),
    ("zdtaichu5.cradio", "import importlib; importlib.import_module('gptqmodel.models.ports.zdtaichu5.cradio_model')"),
    ("interns1.droppath", "from gptqmodel.models.ports.interns1.modeling_interns1 import InternS1DropPath; InternS1DropPath(0.5)(torch.ones(2, 3))"),
    ("minicpmv.legacy", """
from gptqmodel.models.ports.minicpmv.configuration_minicpm import MiniCPMVConfig
from gptqmodel.models.ports.minicpmv.modeling_minicpmv import MiniCPMV
config = MiniCPMVConfig(version=1.0, vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                       head_dim=8, max_position_embeddings=32)
MiniCPMV(config)
"""),
]


@pytest.mark.parametrize("name,source", BOUNDARIES, ids=[item[0] for item in BOUNDARIES])
def test_missing_timm_at_dependent_vision_boundary(name, source):
    _missing_boundary(source, "timm")


def _missing_boundary(source, blocked):
    _run("import torch\ntry:\n" + textwrap.indent(source, "    ") + """
except ImportError as exc:
    assert "pip install" in str(exc), str(exc)
    assert "gptqmodel[vision]" in str(exc), str(exc)
    assert exc.__cause__ is not None
else:
    raise AssertionError('The missing dependency boundary was not reached')
""", blocked=blocked)


@pytest.mark.parametrize("name,source", [
    ("deepseek_vl_v2.processor", "from gptqmodel.models.ports.deepseek_vl_v2.processing_deepseek_vl_v2 import ImageTransform; ImageTransform()"),
    ("openvla.processor", "from PIL import Image; from gptqmodel.models.ports.openvla.processing_prismatic import PrismaticImageProcessor; PrismaticImageProcessor().apply_transform(Image.new('RGB', (4, 4)))"),
], ids=["deepseek_vl_v2.processor", "openvla.processor"])
def test_missing_torchvision_at_processor_boundary(name, source):
    _missing_boundary(source, "torchvision")


def test_interns1_inference_does_not_require_training_only_timm():
    _run("""
        import torch
        from gptqmodel.models.ports.interns1.modeling_interns1 import InternS1DropPath
        hidden = torch.ones(2, 3)
        assert InternS1DropPath(0).train()(hidden) is hidden
        assert InternS1DropPath(0.5).eval()(hidden) is hidden
        assert 'timm' not in sys.modules
    """)
