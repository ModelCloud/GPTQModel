# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Enforce the model-porting policy (docs/transformers5-model-ports.md).

1. Every MODEL_MAP definition loads through a package-owned LocalModelLoader.
2. Ports import Transformers framework infrastructure only, never
   ``transformers.models.*`` (directly or through a top-level re-export).
3. Ports contain no pre-5.23 compatibility patterns.

The ``PENDING_*`` sets list today's violators. They may only shrink: an entry
that no longer violates fails the test so it gets removed. They must be empty
before release.
"""

import ast
import re
from pathlib import Path

import transformers

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.local import LocalModelLoader


PORTS = Path(__file__).resolve().parents[1] / "gptqmodel" / "models" / "ports"

PENDING_NATIVE_DEFINITIONS = frozenset({
   
})

PENDING_MODEL_IMPORT_PORTS = frozenset({
    "chatglm", "ernie4_5_vl_moe", "internvl_chat", "kimi_k3", "lfm3_vl", "llama", "mimo", "minicpm_o",
    "nemotron_h_puzzle", "ovis2_6_moe", "unlimited_ocr",
})

PENDING_STALE_PATTERN_PORTS = frozenset({
    "ernie4_5_vl_moe", "kimi_k3", "native/falcon_h1", "native/nemotron_h", "native/zamba2",
})

STALE_PATTERNS = re.compile(
    r"\bcache_position\b|\bpast_key_value\b|use_return_dict|if not return_dict|from_legacy_cache|to_legacy_cache"
    r"|get_usable_length|\bseen_tokens\b|_prepare_4d_|AttentionMaskConverter|_update_causal_mask|def _reorder_cache"
    r"|def _set_gradient_checkpointing|torch\.utils\.checkpoint|_supports_cache_class|_supports_static_cache"
    r"|_supports_quantized_cache|_supports_flash_attn_2|num_logits_to_keep|\brope_scaling\b|\btorch_dtype\b"
    r"|check_model_inputs|all_hidden_states|all_self_attns|transformers\.__version__|is_transformers_version"
)


def _port_files():
    return sorted(path for path in PORTS.rglob("*.py") if "__pycache__" not in path.parts)


def _port_name(path):
    parts = path.relative_to(PORTS).parts
    # Vendored Transformers families are tracked individually: native/<family>.
    if parts[0] == "native" and len(parts) > 2:
        return f"native/{parts[1]}"
    return parts[0]


def _model_family_imports(path):
    """Yield every import in ``path`` that resolves into ``transformers.models``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("transformers.models"):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("transformers.models"):
                yield node.module
            elif node.module == "transformers":
                for alias in node.names:
                    value = getattr(transformers, alias.name, None)
                    module = getattr(value, "__module__", "") or getattr(value, "__name__", "")
                    if module.startswith("transformers.models"):
                        yield f"transformers.{alias.name} ({module})"


def _native_definitions():
    return {
        model_type
        for model_type, definition in MODEL_MAP.items()
        if not isinstance(getattr(definition, "loader", None), LocalModelLoader)
    }


def _model_import_ports():
    return {_port_name(path) for path in _port_files() if any(_model_family_imports(path))}


# A released config.json may still spell a field the old way; configuration
# modules may read it once at parse time and convert it. Nothing else may.
PARSE_TIME_FIELDS = frozenset({"rope_scaling", "torch_dtype"})


def _stale_hits(path):
    hits = set(STALE_PATTERNS.findall(path.read_text(encoding="utf-8")))
    if path.name.startswith("configuration_"):
        hits -= PARSE_TIME_FIELDS
    return hits


def _stale_pattern_ports():
    return {_port_name(path) for path in _port_files() if _stale_hits(path)}


def test_every_definition_uses_a_package_owned_loader():
    actual = _native_definitions()
    assert actual <= PENDING_NATIVE_DEFINITIONS, f"definitions without LocalModelLoader: {sorted(actual - PENDING_NATIVE_DEFINITIONS)}"
    assert not PENDING_NATIVE_DEFINITIONS - actual, f"ported; remove from pending: {sorted(PENDING_NATIVE_DEFINITIONS - actual)}"


def test_ports_never_import_transformers_model_families():
    actual = _model_import_ports()
    new = actual - PENDING_MODEL_IMPORT_PORTS
    details = {
        str(path.relative_to(PORTS)): sorted(set(_model_family_imports(path)))
        for path in _port_files()
        if _port_name(path) in new and any(_model_family_imports(path))
    }
    assert not new, f"ports importing transformers.models.*: {details}"
    assert not PENDING_MODEL_IMPORT_PORTS - actual, (
        f"now framework-only; remove from pending: {sorted(PENDING_MODEL_IMPORT_PORTS - actual)}"
    )


def test_ports_contain_no_pre_5_23_compatibility_code():
    actual = _stale_pattern_ports()
    new = actual - PENDING_STALE_PATTERN_PORTS
    details = {
        str(path.relative_to(PORTS)): sorted(_stale_hits(path))
        for path in _port_files()
        if _port_name(path) in new and _stale_hits(path)
    }
    assert not new, f"stale pre-5.23 patterns: {details}"
    assert not PENDING_STALE_PATTERN_PORTS - actual, (
        f"now clean; remove from pending: {sorted(PENDING_STALE_PATTERN_PORTS - actual)}"
    )


def test_policy_detects_violations(tmp_path, monkeypatch):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from transformers import Qwen3Config, PreTrainedModel\n"
        "from transformers.models.llama import modeling_llama\n"
        "from transformers.cache_utils import DynamicCache\n"
    )
    found = sorted(_model_family_imports(sample))
    assert found[0].startswith("transformers.Qwen3Config (transformers.models.qwen3")
    assert found[1] == "transformers.models.llama"
    assert len(found) == 2
    assert STALE_PATTERNS.search("def forward(self, x, cache_position=None):")
    assert not STALE_PATTERNS.search("def forward(self, x, past_key_values=None, position_ids=None):")
