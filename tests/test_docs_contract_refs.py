# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Docs must stay copy-runnable and their `file:line` contract refs must resolve.

The reference pages under `docs/` are the logic-flow contract (see
`AGENTS.md`), so a block a reader copies must import and call the shipped API,
and a `path.py:N` citation must point at the symbol it names. Both rot
silently: an example keeps its shape after an API change and a line number
drifts after any unrelated edit above it.

This guard is deliberately small and fast: it imports the `gptqmodel` names
used by the doc blocks this change touched, and resolves an explicit list of
contract citations. It downloads nothing and touches no device. Extend the
lists when you add a contract page or fix a doc block; a full markdown-block
runner does not exist in `tests/` yet.
"""

import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

import gptqmodel
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import CalibrationConfig, QuantizeConfig


REPO_ROOT = Path(__file__).resolve().parents[1]

# Docs whose fenced Python blocks are import-checked. Add a page here when you
# fix one of its examples.
DOC_FILES = (
    "docs/calibration-config.md",
    "docs/quantization-lifecycle-contracts.md",
    "docs/quantization-features.md",
    "docs/rotation-model-support.md",
    "docs/eora/README.md",
)

# `(doc, source path, symbol, cited)`. `cited` is either a line number, where the
# doc must cite `path:N` and the source must define `symbol` within
# `LINE_WINDOW` of `N`, or the string `"symbol"`, where the doc must cite
# `path::symbol` instead. Use the symbol form for any citation into a file that
# keeps growing (`models/base.py` has broken this guard in three separate
# rounds); a symbol cannot drift when unrelated code moves above it.
LINE_WINDOW = 8
SYMBOL_CITATION = "symbol"
CONTRACT_REFS = (
    ("docs/adapter-rank-pattern-matching.md", "gptqmodel/quantization/config.py",
     "extract_adapter_rank_patterns", SYMBOL_CITATION),
    ("docs/awq-scale-folding.md", "gptqmodel/models/base.py", "_select_prev_op", SYMBOL_CITATION),
    ("docs/calibration-replay.md", "gptqmodel/models/base.py",
     "capture_layer_type_replay_kwargs", SYMBOL_CITATION),
    ("docs/calibration-replay.md", "gptqmodel/models/base.py",
     "_build_layer_type_position_embeddings", SYMBOL_CITATION),
    ("docs/fp8-scale-semantics.md", "gptqmodel/models/base.py",
     "_decoder_fp8_scale_semantics", SYMBOL_CITATION),
    ("docs/fusion-planner.md", "gptqmodel/models/fusion.py", "_share_operands", 444),
    # `awq_processor.py` keeps growing (the #4880 replay packing added ~70
    # lines above this symbol), so cite it by name.
    ("docs/layer-type-replay-masks.md", "gptqmodel/looper/awq_processor.py",
     "_rotary_layer_type_kwargs", SYMBOL_CITATION),
    ("docs/layer-type-replay-masks.md", "gptqmodel/models/definitions/olmo3.py",
     "Olmo3QModel", 53),
    ("docs/multi_gpu_quantization.md", "gptqmodel/utils/torch.py", "torch_sync", 611),
    # `writer.py` keeps growing (the #5029 passthrough ownership scope added
    # ~70 lines above the class), so cite the symbol instead of a line.
    ("docs/native-checkpoint-export.md", "gptqmodel/models/writer.py",
     "ModelWriter", SYMBOL_CITATION),
    ("docs/openai-server-exposure.md", "gptqmodel/utils/openai_server.py",
     "_validate_bind", None),
    ("docs/openai-server-exposure.md", "gptqmodel/utils/openai_server.py",
     "BodySizeLimitMiddleware", None),
    ("docs/quantize-run-summary.md", "gptqmodel/models/base.py",
     "_quantize_summary_counts", SYMBOL_CITATION),
    ("docs/quantize-run-summary.md", "gptqmodel/models/base.py",
     "_log_quantize_summary", None),
    ("docs/rocm-default-environment.md", "gptqmodel/quantization/gptq.py",
     "compute_hessian_xtx", 1207),
    ("docs/rocm-default-environment.md", "gptqmodel/quantization/gptq.py",
     "hessian_inverse", 2108),
    # `calibration.py` grew a 70-line payload normalizer above
    # `prepare_calibration_dataset` (#4934); these three citations are symbol
    # anchored so an unrelated edit cannot drift them again.
    ("docs/decision-transformer-calibration.md", "gptqmodel/utils/calibration.py",
     "_CALIBRATION_COLUMNS", SYMBOL_CITATION),
    ("docs/decision-transformer-calibration.md", "gptqmodel/utils/calibration.py",
     "prepare_calibration_dataset", SYMBOL_CITATION),
    ("docs/decision-transformer-calibration.md", "gptqmodel/utils/calibration.py",
     "_is_tensor_kwargs_row", SYMBOL_CITATION),
)

PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
IMPORT_LINE = re.compile(r"^from\s+(gptqmodel[\w.]*)\s+import\s+\(?([^)\n]+)\)?", re.M)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _python_blocks(rel: str):
    return PYTHON_BLOCK.findall(_read(rel))


def _doc_imports():
    """`(module, name)` for every `gptqmodel` import in the checked doc blocks."""

    found = set()
    for rel in DOC_FILES:
        for block in _python_blocks(rel):
            for match in IMPORT_LINE.finditer(block):
                module = match.group(1)
                for raw in match.group(2).split(","):
                    name = raw.split(" as ")[0].strip()
                    if re.fullmatch(r"[A-Za-z_]\w*", name):
                        found.add((module, name))
    return found


def test_doc_python_blocks_only_import_real_gptqmodel_names():
    imports = _doc_imports()
    assert imports, "no gptqmodel imports found in the checked doc blocks"

    missing = []
    for module, name in sorted(imports):
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - the failure is reported
            missing.append(f"{module} ({type(exc).__name__}: {exc})")
            continue
        if not hasattr(mod, name):
            missing.append(f"{module}.{name}")
    assert not missing, f"doc blocks import names the package does not export: {missing}"


def test_calibration_config_examples_use_the_post_4724_api():
    # One public class: the looper re-export is the same object, not a copy.
    assert CalibrationConfig is gptqmodel.CalibrationConfig
    assert CalibrationConfig is gptqmodel.quantization.CalibrationConfig

    # `prepare_dataset` is `init=False`; only `bind()` sets it.
    with pytest.raises(TypeError):
        CalibrationConfig(dataset=["a"], prepare_dataset=lambda *a, **k: None)
    bound = CalibrationConfig(dataset=["a"], concat_size=16).bind(lambda *a, **k: None)
    assert bound.prepare_dataset is not None

    # No checked block constructs a pre-bound config, and the lifecycle page
    # shows the `bind()` form.
    for rel in DOC_FILES:
        for block in _python_blocks(rel):
            assert not re.search(r"CalibrationConfig\([^)]*prepare_dataset\s*=", block), (
                f"{rel}: a CalibrationConfig block passes `prepare_dataset=`; "
                "use `CalibrationConfig(...).bind(model.prepare_dataset)`"
            )
    lifecycle = _read("docs/quantization-lifecycle-contracts.md")
    assert "CalibrationConfig.bind(" in lifecycle or ").bind(model.prepare_dataset)" in lifecycle


def test_rotation_is_a_config_field_not_a_quantize_kwarg():
    assert "rotation" in {f.name for f in dataclasses.fields(QuantizeConfig)}
    assert "rotation" not in inspect.signature(BaseQModel.quantize).parameters

    for rel in DOC_FILES:
        for block in _python_blocks(rel):
            assert not re.search(r"quantize\([^)]*\brotation\s*=", block), (
                f"{rel}: rotation is a QuantizeConfig field, not a quantize() kwarg"
            )


def test_eora_generation_saves_where_the_readme_loads():
    generation = _read("docs/eora/eora_generation.py")
    assert "path=eora_path," in generation
    assert "os.path.join(quant_path, eora_path)" not in generation

    readme = _read("docs/eora/README.md")
    save_paths = re.findall(r"--eora_save_path\s+(\S+)", readme)
    assert save_paths, "the README no longer documents --eora_save_path"
    load_paths = re.findall(r"Lora\([^)]*?\bpath='([^']*)'", readme, re.S)
    assert load_paths, "the README inference example no longer sets Lora.path"
    for load in load_paths:
        assert load == load.strip(), f"Lora.path has surrounding whitespace: {load!r}"
        assert load in save_paths, (
            f"README loads the adapter from {load!r} but saves it to {save_paths!r}"
        )


@pytest.mark.parametrize("doc, source, symbol, cited", CONTRACT_REFS)
def test_contract_reference_resolves(doc, source, symbol, cited):
    source_path = REPO_ROOT / source
    assert source_path.is_file(), f"{doc} cites a missing file: {source}"

    lines = source_path.read_text(encoding="utf-8").splitlines()
    definition = re.compile(
        rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b"
        rf"|^\s*{re.escape(symbol)}\s*(?::[^=]+)?="
    )
    hits = [index + 1 for index, line in enumerate(lines) if definition.match(line)]
    assert hits, f"{doc} cites `{symbol}`, which is not defined in {source}"

    if cited is None:
        return

    if cited == SYMBOL_CITATION:
        citation = f"{Path(source).name}::{symbol}"
        assert citation in _read(doc), f"{doc} no longer cites {citation}"
        return

    assert f"{Path(source).name}:{cited}" in _read(doc), (
        f"{doc} no longer cites {Path(source).name}:{cited}"
    )
    nearest = min(hits, key=lambda line: abs(line - cited))
    assert abs(nearest - cited) <= LINE_WINDOW, (
        f"{doc} cites `{symbol}` at {source}:{cited}, but it is defined at {hits}"
    )
