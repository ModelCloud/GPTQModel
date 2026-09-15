# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import builtins
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from PIL import Image


def test_package_import_without_torchvision() -> None:
    script = textwrap.dedent("""
        import importlib.util
        from importlib.machinery import PathFinder
        from pathlib import Path

        original_find_spec = PathFinder.find_spec

        def find_spec_without_torchvision(fullname, *args, **kwargs):
            if fullname == "torchvision" or fullname.startswith("torchvision."):
                return None
            return original_find_spec(fullname, *args, **kwargs)

        PathFinder.find_spec = staticmethod(find_spec_without_torchvision)
        assert importlib.util.find_spec("torchvision") is None

        import gptqmodel
        from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel

        assert Path(gptqmodel.__file__).resolve().parent == Path.cwd() / "gptqmodel"
        try:
            InternVLChatQModel._build_transform(8)
        except ImportError as exc:
            assert "InternVL image preprocessing requires torchvision" in str(exc)
            assert isinstance(exc.__cause__, ModuleNotFoundError)
            assert exc.__cause__.name == "torchvision"
        else:
            raise AssertionError("Image preprocessing should require torchvision")
    """)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    result = subprocess.run(
        [*command, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("input_size", [1, 8])
def test_image_transform_preserves_preprocessing(mode: str, input_size: int) -> None:
    transforms = pytest.importorskip("torchvision.transforms")
    from torchvision.transforms.functional import InterpolationMode

    from gptqmodel.models.definitions.internvl_chat import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        InternVLChatQModel,
    )

    channels = len(mode)
    pixels = bytes(index % 256 for index in range(7 * 5 * channels))
    image = Image.frombytes(mode, (7, 5), pixels)
    expected = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )(image)

    actual = InternVLChatQModel._build_transform(input_size)(image)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, input_size, input_size)


def test_image_transform_preserves_unrelated_import_errors(monkeypatch) -> None:
    from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel

    original_import = builtins.__import__
    missing_dependency = ModuleNotFoundError("Missing dependency", name="numpy")

    def fail_transitive_import(name, *args, **kwargs):
        if name == "torchvision.transforms":
            raise missing_dependency
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_transitive_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        InternVLChatQModel._build_transform(8)

    assert exc_info.value is missing_dependency
