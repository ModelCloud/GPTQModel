# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NativeProcessor resolves sub-processors from the checkpoint's declared types.

Real checkpoints name their sub-processor classes (``image_processor_type`` /
``video_processor_type`` / ``feature_extractor_type`` in
``processor_config.json`` or the legacy per-processor files). Those names must
resolve to vendored (or framework) classes; the ``model_type`` table is only a
fallback, and nothing may come from ``transformers.models``.
"""

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.ports.native.auto import NativeProcessor, _registry


NATIVE = "gptqmodel.models.ports.native."
REPO = Path(__file__).resolve().parents[1]
SPECIAL = [
    "<pad>", "<s>", "</s>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
    "<|image|>", "<|boi|>", "<|eoi|>", "<|audio|>", "<|boa|>", "<|eoa|>", "<|video|>",
]


def _tokenizer(extra_special_tokens=None):
    vocab = {token: index for index, token in enumerate(SPECIAL + ["a", "b", "[UNK]"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return TokenizersBackend(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        extra_special_tokens=extra_special_tokens or {},
    )


def _write_config(path, model_type):
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))


def _vendored(module):
    return module.startswith(NATIVE)


def _load_through_definition(model_type, path):
    loader = MODEL_MAP[model_type].loader
    assert loader.processor_class == f"{NATIVE}auto.NativeProcessor"
    return loader.load_processor(str(path))


def test_qwen3_vl_moe_processor_uses_declared_image_processor(tmp_path):
    pytest.importorskip("torchvision")
    from gptqmodel.models.ports.native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from gptqmodel.models.ports.native.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from gptqmodel.models.ports.native.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    # qwen3_vl_moe has no IMAGE_PROCESSOR registry entry (Transformers has none
    # either); real checkpoints declare Qwen2VLImageProcessor[Fast].
    assert "qwen3_vl_moe" not in _registry.IMAGE_PROCESSOR
    Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=_tokenizer(),
        video_processor=Qwen3VLVideoProcessor(),
    ).save_pretrained(tmp_path)
    _write_config(tmp_path, "qwen3_vl_moe")

    processor = _load_through_definition("qwen3_vl_moe", tmp_path)

    assert type(processor) is Qwen3VLProcessor
    assert type(processor.image_processor) is Qwen2VLImageProcessor
    assert type(processor.video_processor) is Qwen3VLVideoProcessor


def test_qwen3_vl_moe_legacy_fast_declaration(tmp_path):
    pytest.importorskip("torchvision")
    from gptqmodel.models.ports.native.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from gptqmodel.models.ports.native.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from gptqmodel.models.ports.native.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=_tokenizer(),
        video_processor=Qwen3VLVideoProcessor(),
    ).save_pretrained(tmp_path)
    _write_config(tmp_path, "qwen3_vl_moe")
    # Released Qwen3-VL-MoE checkpoints: legacy per-file layout naming the Fast class.
    nested = json.loads((tmp_path / "processor_config.json").read_text())
    image = dict(nested.pop("image_processor"), image_processor_type="Qwen2VLImageProcessorFast")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(image))
    (tmp_path / "processor_config.json").write_text(json.dumps(nested))

    processor = _load_through_definition("qwen3_vl_moe", tmp_path)
    assert type(processor.image_processor) is Qwen2VLImageProcessor
    pil = NativeProcessor.from_pretrained(str(tmp_path), backend="pil")
    assert type(pil.image_processor).__name__ == "Qwen2VLImageProcessorPil"
    assert _vendored(type(pil.image_processor).__module__)


def test_diffusion_gemma_processor_uses_declared_feature_extractor(tmp_path):
    pytest.importorskip("torchvision")
    from gptqmodel.models.ports.native.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from gptqmodel.models.ports.native.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    from gptqmodel.models.ports.native.gemma4.processing_gemma4 import Gemma4Processor
    from gptqmodel.models.ports.native.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

    for table in (_registry.IMAGE_PROCESSOR, _registry.VIDEO_PROCESSOR, _registry.FEATURE_EXTRACTOR):
        assert "diffusion_gemma" not in table
    tokenizer = _tokenizer(
        {"image_token": "<|image|>", "boi_token": "<|boi|>", "eoi_token": "<|eoi|>",
         "audio_token": "<|audio|>", "boa_token": "<|boa|>", "eoa_token": "<|eoa|>"}
    )
    Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        image_processor=Gemma4ImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Gemma4VideoProcessor(),
    ).save_pretrained(tmp_path)
    _write_config(tmp_path, "diffusion_gemma")

    processor = _load_through_definition("diffusion_gemma", tmp_path)

    assert type(processor) is Gemma4Processor
    assert type(processor.feature_extractor) is Gemma4AudioFeatureExtractor
    assert type(processor.image_processor) is Gemma4ImageProcessor
    assert type(processor.video_processor) is Gemma4VideoProcessor


def test_declared_model_family_class_is_never_selected(tmp_path):
    """A declared name outside the package resolves to nothing, not to Transformers."""

    from gptqmodel.models.ports.native.auto import _declared_sub_processor_class

    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"image_processor_type": "CLIPImageProcessor"}))
    (tmp_path / "video_preprocessor_config.json").write_text(json.dumps({"video_processor_type": "LlavaOnevisionVideoProcessor"}))
    for attribute in ("image_processor", "video_processor", "feature_extractor"):
        found = _declared_sub_processor_class(attribute, str(tmp_path), None, "torchvision")
        assert found is None or _vendored(found.__module__), (attribute, found)


_PIL_FALLBACK_SCRIPT = r"""
import sys

sys.modules["torchvision"] = None
import transformers.utils.import_utils as import_utils

_is_package_available = import_utils._is_package_available
import_utils._is_package_available = (
    lambda name, *args, **kwargs: (False, None)
    if str(name).split(".")[0] == "torchvision"
    else _is_package_available(name, *args, **kwargs)
)

from gptqmodel.models.ports.native.auto import _by_name, _image_processor_by_name

pil = _by_name("Qwen2VLImageProcessorPil")
assert pil is not None, "PIL sibling was not indexed"
assert pil.__name__ == "Qwen2VLImageProcessorPil", pil
assert pil.__module__.startswith("gptqmodel.models.ports.native."), pil
assert _image_processor_by_name("Qwen2VLImageProcessor", "pil") is pil
assert _image_processor_by_name("Qwen2VLImageProcessor", "torchvision") is pil
assert _by_name("Qwen2VLImageProcessor") is None
print("RESULT ok")
"""


def test_pil_fallback_resolves_without_torchvision():
    """A missing torchvision backend must not abort the vendored PIL fallback."""

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO), env.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", _PIL_FALLBACK_SCRIPT],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-4000:]
    assert "RESULT ok" in completed.stdout


def _native_processor_definitions():
    return sorted(
        model_type
        for model_type, definition in MODEL_MAP.items()
        if getattr(definition.loader, "processor_class", None) == f"{NATIVE}auto.NativeProcessor"
    )


@pytest.mark.parametrize("model_type", _native_processor_definitions())
def test_sub_processor_classes_resolve_package_locally(model_type, tmp_path):
    """Every NativeProcessor definition: each sub-processor resolves to a vendored
    or framework class with the checkpoint declaring nothing, or (for families
    without a model_type entry) with the declaration a released checkpoint makes."""

    entry = _registry.PROCESSOR.get(model_type)
    if entry is None:
        pytest.skip(f"{model_type}: processor is declared by the checkpoint (no model_type entry)")
    try:
        processor_class = getattr(import_module(entry[0]), entry[1])
    except ModuleNotFoundError as exc:
        if str(exc.name).split(".")[0] == "torchvision":
            pytest.skip(f"{model_type}: processor module requires torchvision")
        raise
    _write_config(tmp_path, model_type)
    for attribute in processor_class.get_attributes():
        if "tokenizer" in attribute:
            continue
        table = {
            "image_processor": _registry.IMAGE_PROCESSOR,
            "video_processor": _registry.VIDEO_PROCESSOR,
            "feature_extractor": _registry.FEATURE_EXTRACTOR,
        }[attribute]
        if model_type not in table and getattr(processor_class, f"{attribute}_class", None) is None:
            # Resolved only from the checkpoint's declared type; prove the
            # declaration made by a family sharing this processor class (or
            # this processor's own family) resolves package-locally.
            family = processor_class.__module__.split(".")[-2]
            sharing = [key for key, value in _registry.PROCESSOR.items() if tuple(value) == tuple(entry)]
            sibling = next((table[key][1] for key in sharing if key in table), None) or next(
                (name for module, name in table.values() if module.split(".")[-2] == family), None
            )
            assert sibling is not None, f"{model_type}.{attribute}: no vendored class to declare"
            key = {"image_processor": "image_processor_type", "video_processor": "video_processor_type",
                   "feature_extractor": "feature_extractor_type"}[attribute]
            (tmp_path / "processor_config.json").write_text(json.dumps({attribute: {key: sibling}}))
        try:
            resolved = NativeProcessor.sub_processor_class(
                processor_class, attribute, str(tmp_path), model_type=model_type
            )
        except ModuleNotFoundError as exc:
            if str(exc.name).split(".")[0] == "torchvision":
                pytest.skip(f"{model_type}.{attribute}: requires torchvision")
            raise
        except ValueError as exc:
            from transformers.utils import is_torchvision_available

            if not is_torchvision_available() and "has no vendored" in str(exc):
                pytest.skip(f"{model_type}.{attribute}: requires torchvision")
            raise
        assert not resolved.__module__.startswith("transformers.models."), (model_type, attribute, resolved)
        assert resolved.__module__.startswith((NATIVE, "transformers.")), (model_type, attribute, resolved)
        (tmp_path / "processor_config.json").unlink(missing_ok=True)
