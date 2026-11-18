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
    "afmoe", "apertus", "axk2", "bloom", "cohere2_moe", "cohere_compass", "dbrx", "decision_transformer",
    "deepseek_ocr2", "deepseek_v32", "deepseek_v4", "deepseek_vl", "diffusion_gemma", "dots1", "dots3_note",
    "ernie4_5", "ernie4_5_moe", "exaone4_5", "falcon_h1", "falcon_mamba", "flint", "gemma2", "gemma3",
    "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text",
    "glm", "glm4", "glm4_moe", "glm4_moe_lite", "glm4v", "glm4v_moe", "glm4v_moe_text", "glm5_next", "glm_moe_dsa",
    "glm_ocr", "glmasr", "gpt2", "gpt_bigcode", "gpt_neo", "gpt_neox", "gpt_oss", "gptj", "granite_swa",
    "granitemoe", "granitemoe_swa", "granitemoehybrid", "hrm_text", "hunyuan_v1_dense", "hunyuan_v1_moe",
    "hunyuan_vl", "hy_v3", "inkling_mm_model", "internvl", "jamba", "laguna", "lfm2", "lfm2_moe", "lfm2_vl", "lfm3",
    "llama4", "llama4_text", "llava", "longcat_flash", "mimo_v2_flash", "minicpm3", "minicpmv4_6", "minimax_m2",
    "minimax_m3_vl", "ministral3", "mistral3", "mixtral", "mllama", "mllama_text_model", "mpt", "muse_glimmer",
    "olmo2", "olmo3", "olmo_hybrid", "olmoe", "opt", "ovis2", "phi", "phi3", "phimoe", "qwen2", "qwen2_5_omni",
    "qwen2_audio", "qwen2_moe", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5_text", "qwen3_moe",
    "qwen3_next", "qwen3_omni_moe", "qwen3_vl", "qwen3_vl_moe", "qwen4_exp", "qwen4_exp_text", "smolvlm",
    "solar_open", "starcoder2", "voxtral", "zamba", "zamba2",
})

PENDING_MODEL_IMPORT_PORTS = frozenset({
    "agnes", "apertus", "bailing_moe", "brumby", "chatglm", "codegen", "dbrx_converted", "deepseek_v2",
    "deepseek_v3", "deepseek_vl_v2", "dream", "ernie4_5_vl_moe", "exaone4", "instella", "intern_s2_preview",
    "internlm", "interns1", "internvl_chat", "kimi_k25", "kimi_k3", "klear", "lfm3_omni", "lfm3_vl", "llama",
    "llava_qwen2", "locateanything", "mage_vl", "mimo", "mimo_v2", "minicpm_o", "minicpmv", "minimax_m2", "moss",
    "nemotron_h", "nemotron_h_puzzle", "nemotron_labs_diffusion", "nemotron_omni", "openvla", "ouro", "ovis",
    "ovis2_5", "ovis2_6_moe", "phi4", "qwen2_5_vl", "qwen2_vl", "qwen3", "qwen_drive", "rw", "sensenova_loop",
    "solar_open2", "unlimited_ocr", "xing4_0", "zdtaichu5",
})

PENDING_STALE_PATTERN_PORTS = frozenset({
    "ernie4_5_vl_moe", "kimi_k3", "minicpm_o", "native/bloom", "native/decision_transformer", "native/falcon",
    "native/falcon_mamba", "native/gpt_neo", "native/gptj", "native/llama4", "native/mpt", "native/nemotron_h_omni",
    "native/timm_wrapper", "native/zamba",
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
PARSE_TIME_FIELDS = frozenset({"rope_scaling", "torch_dtype", "num_logits_to_keep"})


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
