# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Enforce the model-porting policy (docs/transformers5-model-ports.md).

1. Every MODEL_MAP definition loads through a package-owned LocalModelLoader.
2. Ports import Transformers framework infrastructure only, never
   ``transformers.models.*`` (directly or through a top-level re-export).
3. Ports contain no pre-5.23 compatibility patterns.
4. Nothing else in ``gptqmodel/`` imports ``transformers.models.*`` either,
   except the commented ``PACKAGE_MODEL_IMPORT_EXCEPTIONS``.
5. Nothing in ``gptqmodel/`` resolves Transformers classes dynamically
   (``getattr(transformers, <non-literal>)``, ``import_module("transformers.models...")``),
   except the commented ``DYNAMIC_TRANSFORMERS_LOOKUP_EXCEPTIONS``. The runtime
   counterpart is tests/test_no_transformers_family_imports.py.

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


PACKAGE = Path(__file__).resolve().parents[1] / "gptqmodel"
PORTS = PACKAGE / "models" / "ports"

PENDING_NATIVE_DEFINITIONS = frozenset()

PENDING_MODEL_IMPORT_PORTS = frozenset()

PENDING_STALE_PATTERN_PORTS = frozenset()

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


# BaseQModel subclasses under gptqmodel/ that are never instantiated as a model
# and therefore may keep BaseQModel's Transformers Auto* loader. Keyed by
# "module.Class"; each entry needs a comment saying why. Currently none.
NON_MODEL_DEFINITION_CLASSES = frozenset()


def _definition_classes():
    """Every BaseQModel subclass a load can reach: MODEL_MAP values, classes
    auto.py dispatches to outside MODEL_MAP (e.g. KimiLinearNativeQModel), and
    any other subclass defined in gptqmodel/models/definitions."""

    import importlib
    import pkgutil

    import gptqmodel.models.definitions as definitions
    from gptqmodel.models.base import BaseQModel

    for module in pkgutil.iter_modules(definitions.__path__):
        importlib.import_module(f"{definitions.__name__}.{module.name}")

    found = {}
    pending = list(BaseQModel.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("gptqmodel."):
            found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    for definition in MODEL_MAP.values():
        found[f"{definition.__module__}.{definition.__qualname__}"] = definition
    return found


def _native_definitions():
    return {
        name
        for name, definition in _definition_classes().items()
        if name not in NON_MODEL_DEFINITION_CLASSES
        and not isinstance(getattr(definition, "loader", None), LocalModelLoader)
    }


def _model_import_ports():
    return {_port_name(path) for path in _port_files() if any(_model_family_imports(path))}


# Exact (file -> idioms) exemptions from STALE_PATTERNS. Only for verbatim Transformers 5.23 code whose
# observable behavior the 5.23 output-capturing conventions cannot reproduce; each needs a reason.
STALE_PATTERN_EXCEPTIONS = {
    # Transformers 5.23's DecisionTransformerModel returns the 3-tuple (state_preds, action_preds,
    # return_preds) for return_dict=False while its dict output has six fields (hidden_states,
    # attentions, last_hidden_state too). `can_return_tuple` would return every non-None field, so
    # the verbatim `if not return_dict:` branch is the only way to keep native tuple semantics
    # (enforced by tests/test_native_port_parity.py).
    "native/decision_transformer/modeling_decision_transformer.py": frozenset({"if not return_dict"}),
}


# A released config.json may still spell a field the old way; configuration
# modules may read it once at parse time and convert it. Nothing else may.
PARSE_TIME_FIELDS = frozenset({"rope_scaling", "torch_dtype", "num_logits_to_keep"})


def _stale_hits(path):
    hits = set(STALE_PATTERNS.findall(path.read_text(encoding="utf-8")))
    if path.name.startswith("configuration_"):
        hits -= PARSE_TIME_FIELDS
    return hits - STALE_PATTERN_EXCEPTIONS.get(path.relative_to(PORTS).as_posix(), frozenset())


def _stale_pattern_ports():
    return {_port_name(path) for path in _port_files() if _stale_hits(path)}


def test_every_definition_uses_a_package_owned_loader():
    actual = _native_definitions()
    assert actual <= PENDING_NATIVE_DEFINITIONS, f"definitions without LocalModelLoader: {sorted(actual - PENDING_NATIVE_DEFINITIONS)}"
    assert not PENDING_NATIVE_DEFINITIONS - actual, f"ported; remove from pending: {sorted(PENDING_NATIVE_DEFINITIONS - actual)}"
    classes = _definition_classes()
    assert not NON_MODEL_DEFINITION_CLASSES - set(classes), (
        f"allowlisted class no longer exists: {sorted(NON_MODEL_DEFINITION_CLASSES - set(classes))}"
    )


def test_definition_sweep_covers_classes_dispatched_outside_model_map():
    from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel

    classes = _definition_classes()
    assert KimiLinearNativeQModel not in set(MODEL_MAP.values())
    assert "gptqmodel.models.definitions.kimi_linear.KimiLinearNativeQModel" in classes


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


def test_stale_pattern_exceptions_are_exact_and_used():
    for relative, idioms in STALE_PATTERN_EXCEPTIONS.items():
        path = PORTS / relative
        assert path.is_file(), f"exception for a missing file: {relative}"
        found = set(STALE_PATTERNS.findall(path.read_text(encoding="utf-8")))
        assert idioms <= found, f"exception no longer used; remove it: {relative}: {sorted(idioms - found)}"
        # Only idioms Transformers 5.23 itself still ships in that file may be exempted.
        upstream = Path(transformers.__file__).parent / "models" / Path(relative).relative_to("native")
        assert idioms <= set(STALE_PATTERNS.findall(upstream.read_text(encoding="utf-8"))), relative


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


# Package code outside the ports that may still name a Transformers model-family
# class, keyed by file relative to gptqmodel/ -> the exact imports allowed. Only
# two kinds qualify; each site carries an inline comment.
PACKAGE_MODEL_IMPORT_EXCEPTIONS = {
    # Unsupported models (no MODEL_MAP definition): BaseQModel's default loader
    # and processor run Transformers' own or checkpoint code, never a port.
    "models/base.py": {
        "transformers.AutoModelForCausalLM (transformers.models.auto.modeling_auto)",
        "transformers.AutoProcessor (transformers.models.auto.processing_auto)",
    },
    # Unsupported models: config fallback and native-support probe.
    "utils/hf.py": {
        "transformers.AutoConfig (transformers.models.auto.configuration_auto)",
        "transformers.AutoModelForCausalLM (transformers.models.auto.modeling_auto)",
    },
    # Unsupported models: dequantizing a compressed-tensors checkpoint that has
    # no definition; supported ones build through their LocalModelLoader.
    "utils/model_dequant.py": {
        "transformers.AutoConfig (transformers.models.auto.configuration_auto)",
        "transformers.AutoModel (transformers.models.auto.modeling_auto)",
        "transformers.AutoModelForCausalLM (transformers.models.auto.modeling_auto)",
        "transformers.AutoModelForSeq2SeqLM (transformers.models.auto.modeling_auto)",
    },
}

# Files still importing a model family that Transformers 5.23 does not ship. May
# only shrink.
PENDING_PACKAGE_MODEL_IMPORTS = frozenset()


def _package_model_imports():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or PORTS in path.parents:
            continue
        imports = set(_model_family_imports(path))
        if imports:
            found[path.relative_to(PACKAGE).as_posix()] = imports
    return found


def test_package_never_imports_transformers_model_families():
    found = _package_model_imports()
    violations = {
        path: sorted(imports - PACKAGE_MODEL_IMPORT_EXCEPTIONS.get(path, set()))
        for path, imports in found.items()
        if path not in PENDING_PACKAGE_MODEL_IMPORTS and imports - PACKAGE_MODEL_IMPORT_EXCEPTIONS.get(path, set())
    }
    assert not violations, f"gptqmodel/ imports transformers.models.* outside the policy exceptions: {violations}"
    stale = {
        path: sorted(allowed - found.get(path, set()))
        for path, allowed in PACKAGE_MODEL_IMPORT_EXCEPTIONS.items()
        if allowed - found.get(path, set())
    }
    assert not stale, f"exception no longer used; remove it: {stale}"
    assert not PENDING_PACKAGE_MODEL_IMPORTS - set(found), (
        f"now framework-only; remove from pending: {sorted(PENDING_PACKAGE_MODEL_IMPORTS - set(found))}"
    )


# Sites that may look a non-literal name up on the ``transformers`` package,
# keyed by file relative to gptqmodel/ -> number of such sites.
DYNAMIC_TRANSFORMERS_LOOKUP_EXCEPTIONS = {
    # ``_by_name``: framework-class lookup that first checks the lazy export
    # table and refuses (without importing) anything under ``transformers.models``.
    "models/ports/native/auto/__init__.py": 1,
}


def _leading_text(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant):
        return node.values[0].value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _leading_text(node.left)
    return None


def _dynamic_transformers_lookups(path):
    """Yield ``getattr``/``hasattr`` on the transformers module with a non-literal
    name, and ``import_module``/``__import__`` of ``transformers.models...``."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = {"transformers"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "transformers")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in {"getattr", "hasattr"} and len(node.args) >= 2:
            target, attribute = node.args[:2]
            on_transformers = (isinstance(target, ast.Name) and target.id in aliases) or (
                isinstance(target, ast.Call) and target.args and _leading_text(target.args[0]) == "transformers"
            )
            if on_transformers and not (isinstance(attribute, ast.Constant) and isinstance(attribute.value, str)):
                yield f"{node.lineno}: {name}(transformers, <non-literal>)"
        elif name in {"import_module", "__import__"} and node.args:
            text = _leading_text(node.args[0])
            package = next((_leading_text(k.value) for k in node.keywords if k.arg == "package"), None)
            if package is None and len(node.args) > 1:
                package = _leading_text(node.args[1])
            if (text or "").startswith("transformers.models") or (package or "").startswith("transformers.models"):
                yield f"{node.lineno}: {name}({text!r}, package={package!r})"


def test_package_never_resolves_transformers_classes_dynamically():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hits = list(_dynamic_transformers_lookups(path))
        if hits:
            found[path.relative_to(PACKAGE).as_posix()] = hits
    violations = {
        path: hits for path, hits in found.items() if len(hits) > DYNAMIC_TRANSFORMERS_LOOKUP_EXCEPTIONS.get(path, 0)
    }
    assert not violations, f"dynamic transformers class resolution in gptqmodel/: {violations}"
    stale = {path for path, count in DYNAMIC_TRANSFORMERS_LOOKUP_EXCEPTIONS.items() if len(found.get(path, ())) < count}
    assert not stale, f"exception no longer used; lower or remove it: {sorted(stale)}"


def test_dynamic_lookup_guard_detects_violations(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import transformers\n"
        "import transformers as tf\n"
        "from importlib import import_module\n"
        "a = getattr(transformers, config.architectures[0], None)\n"
        "b = getattr(tf, name)\n"
        "c = import_module(f'transformers.models.{family}.modeling_{family}')\n"
        "d = import_module('.' + name, 'transformers.models')\n"
        "e = import_module('.' + name, package='transformers.models')\n"
        "ok1 = getattr(transformers, 'PreTrainedModel')\n"
        "ok2 = import_module(f'gptqmodel.models.ports.native.{family}')\n"
        "ok3 = getattr(module, name)\n"
    )
    hits = list(_dynamic_transformers_lookups(sample))
    assert [hit.split(":")[0] for hit in hits] == ["4", "5", "6", "7", "8"], hits


def test_transformers_requirement_is_pinned_to_the_vendored_minor():
    """The vendored ABI is pinned per Transformers minor: bumping means re-vendoring."""

    from packaging.requirements import Requirement
    from packaging.version import Version

    header = (PORTS / "native" / "auto" / "_registry.py").read_text(encoding="utf-8").splitlines()[0]
    vendored = Version(re.search(r"transformers v(\d+\.\d+\.\d+)", header).group(1))
    lines = (PACKAGE.parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
    (requirement,) = [Requirement(line) for line in lines if re.match(r"transformers\b", line.strip())]
    assert str(requirement.specifier) == f"<{vendored.major}.{vendored.minor + 1},>={vendored}", requirement
    assert Version(transformers.__version__) in requirement.specifier, transformers.__version__
