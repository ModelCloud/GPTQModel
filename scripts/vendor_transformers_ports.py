# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendor Transformers 5.23 model families into gptqmodel/models/ports/native.

Usage:
    python scripts/vendor_transformers_ports.py <transformers_src_pkg_dir> <repo_root> <family> [<family> ...] [--refresh]
    python scripts/vendor_transformers_ports.py <transformers_src_pkg_dir> <repo_root> --all --refresh
    python scripts/vendor_transformers_ports.py <transformers_src_pkg_dir> <repo_root> --all --record-patches

The generator is the source of truth for the vendored tree. GPT-QModel's hand edits (5.23-convention
modernizations) live as unified diffs in scripts/vendor_patches/<family>.patch and are applied after
copying, so ``--all --refresh`` reproduces the committed tree byte for byte. To change a vendored file:
edit it, run ``--all --record-patches`` (re-derives every patch from the committed tree against fresh
generator output), and commit the file together with its patch and native/MANIFEST.json.
Without --refresh, existing family directories are kept. ``--all`` means every vendored family
directory already present under <repo_root>.

Copies configuration/modeling/processing/... modules verbatim, rewrites framework
imports (`from ...x` -> `from transformers.x`), keeps sibling-family relative imports
(`from ..other.x` resolves to ports/native/other), and routes `..auto` to a
package-local registry. Adds the dependency closure (relative imports and
SubConfigSpec default model types). Generates ports/native/auto/_registry.py and
ports/native/MANIFEST.json (sha256 + verbatim/patched/generated for every generated file).
"""

import difflib
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path


PATCHES = Path(__file__).resolve().parent / "vendor_patches"
NATIVE_PARTS = ("gptqmodel", "models", "ports", "native")
# Files under ports/native that the generator does not produce.
HAND_WRITTEN = frozenset({"NOTICE", "__init__.py", "auto/__init__.py", "MANIFEST.json"})
SRC = MODELS = DEST = None


def _configure(src, repo):
    global SRC, MODELS, DEST
    SRC = Path(src)  # .../src/transformers
    MODELS = SRC / "models"
    DEST = Path(repo).joinpath(*NATIVE_PARTS)

KEEP = re.compile(
    r"^((configuration|modeling|processing|image_processing|video_processing|feature_extraction|tokenization|generation)_.*|english_normalizer)\.py$"
)


def family_files(family):
    folder = MODELS / family
    return sorted(p for p in folder.glob("*.py") if KEEP.match(p.name))


def model_type_to_family():
    from transformers.models.auto.configuration_auto import SPECIAL_MODEL_TYPE_TO_MODULE_NAME

    return dict(SPECIAL_MODEL_TYPE_TO_MODULE_NAME)


SPECIAL = model_type_to_family()


def family_of(model_type):
    if model_type in SPECIAL:
        return SPECIAL[model_type]
    name = model_type.replace("-", "_")
    return name


def deps_of(family):
    found = set()
    for path in family_files(family):
        text = path.read_text()
        found.update(re.findall(r"from \.\.([a-z0-9_]+)\.", text))
        for mt in re.findall(r'SubConfigSpec\([^)]*?model_type="([^"]+)"', text, re.S):
            found.add(family_of(mt))
    found.discard("auto")
    return {f for f in found if (MODELS / f).is_dir()}


def closure(families):
    seen, stack = set(), list(families)
    while stack:
        family = stack.pop()
        if family in seen or not (MODELS / family).is_dir():
            continue
        seen.add(family)
        stack.extend(deps_of(family))
    return sorted(seen)


FRAMEWORK = re.compile(r"(^|\n)(\s*)from \.\.\.(?=[A-Za-z_])")
FRAMEWORK_BARE = re.compile(r"(^|\n)(\s*)from \.\.\. import")
AUTO = re.compile(r"from \.\.auto(?:\.[a-z_]+)? import")


ABSOLUTE_MODELS = re.compile(r"\btransformers\.models\.([a-z0-9_]+)\.")


NATIVE_BASE = re.compile(r"^(class \w+\()PreTrainedModel(\):)", re.M)
NATIVE_BASE_IMPORT = (
    "from ..auto import NativePreTrainedModel  # GPT-QModel: vendored classes are library code, not custom code\n"
)


def rewrite(text):
    text = ABSOLUTE_MODELS.sub(lambda m: "gptqmodel.models.ports.native." + m.group(1) + ".", text)
    if NATIVE_BASE.search(text):
        # Transformers skips conversion mappings/native init for "custom code" (any class outside
        # transformers.*); the vendored base reports these classes as library code.
        text = NATIVE_BASE.sub(r"\1NativePreTrainedModel\2", text)
        m = re.search(r"^from \.\.\.modeling_utils import[^\n]*(\([^)]*\))?\n", text, re.M | re.S)
        text = text[: m.end()] + NATIVE_BASE_IMPORT + text[m.end():] if m else NATIVE_BASE_IMPORT + text
    text = FRAMEWORK_BARE.sub(r"\1\2from transformers import", text)
    text = FRAMEWORK.sub(r"\1\2from transformers.", text)
    text = AUTO.sub("from ..auto import", text)
    if "SubConfigSpec" in text and "from ..auto import" in text:
        # SubConfigSpec resolves model types through a registry; use the package-local one.
        text = re.sub(
            r"(\nfrom \.\.auto import [^\n]*\n|\nfrom \.\.auto import \([^)]*\)\n)",
            lambda m: m.group(1) + "from ..auto import SubConfigSpec  # package-local registry\n",
            text,
            count=1,
        )
    return text


HEADER = (
    "# Vendored from huggingface/transformers v5.23.0 (Apache-2.0) by GPT-QModel.\n"
    "# Do not edit by hand except to keep it on the 5.23 ABI; see docs/transformers5-model-ports.md.\n"
)


def verbatim(family):
    """{file name: text} the generator emits for ``family`` before any GPT-QModel patch."""

    files = {"__init__.py": HEADER}
    for path in family_files(family):
        files[path.name] = HEADER + rewrite(path.read_text())
    return files


def make_patch(family, before, after):
    """Unified diff (a/<family>/<file> -> b/<family>/<file>) of the hand edits to one family."""

    chunks = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        if old is None or new is None:
            raise SystemExit(f"{family}/{name}: patches may only edit generated files, not add or remove them")
        for text in (old, new):
            if not text.endswith("\n"):
                raise SystemExit(f"{family}/{name}: file must end with a newline")
        chunks.extend(
            difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                f"a/{family}/{name}", f"b/{family}/{name}", n=3,
            )
        )
    return "".join(chunks)


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_patch(files, patch, family):
    """Apply a unified diff to {name: text} in place; every hunk must match exactly where it says.

    Returns the names of the patched files.
    """

    lines = patch.splitlines(keepends=True)
    index, patched = 0, set()
    while index < len(lines):
        if not (lines[index].startswith("--- ") and lines[index + 1].startswith("+++ ")):
            raise SystemExit(f"{family}.patch: expected a file header at line {index + 1}")
        name = lines[index + 1][len("+++ "):].strip().split("/", 2)[2]
        if name not in files:
            raise SystemExit(f"{family}.patch edits {name}, which the generator does not emit")
        index += 2
        source = files[name].splitlines(keepends=True)
        result, cursor = [], 0
        while index < len(lines) and lines[index].startswith("@@"):
            match = _HUNK.match(lines[index])
            old_count = 1 if match.group(2) is None else int(match.group(2))
            new_count = 1 if match.group(4) is None else int(match.group(4))
            start = int(match.group(1)) - (0 if old_count == 0 else 1)
            if start < cursor:
                raise SystemExit(f"{family}.patch: overlapping hunks in {name}")
            result.extend(source[cursor:start])
            cursor = start
            index += 1
            while old_count or new_count:
                tag, body = lines[index][:1], lines[index][1:]
                if tag in {" ", "-"}:
                    if cursor >= len(source) or source[cursor] != body:
                        raise SystemExit(f"{family}.patch does not apply to {name} at line {cursor + 1}; re-record it")
                    cursor += 1
                    old_count -= 1
                if tag in {" ", "+"}:
                    result.append(body)
                    new_count -= 1
                if tag not in {" ", "-", "+"}:
                    raise SystemExit(f"{family}.patch: bad hunk line {index + 1}")
                index += 1
        result.extend(source[cursor:])
        files[name] = "".join(result)
        patched.add(name)
    return patched


def vendor(family):
    """Write ``family`` (verbatim generator output plus its recorded patch); return {name: status}."""

    files = verbatim(family)
    patch_file = PATCHES / f"{family}.patch"
    patched = apply_patch(files, patch_file.read_text(), family) if patch_file.exists() else set()
    out = DEST / family
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    return {name: "patched" if name in patched else "verbatim" for name in files}


def record_patches(families):
    """Re-derive every scripts/vendor_patches/<family>.patch from the committed tree."""

    PATCHES.mkdir(exist_ok=True)
    for family in families:
        before = verbatim(family)
        after = {name: (DEST / family / name).read_text() for name in before if (DEST / family / name).exists()}
        extra = {p.name for p in (DEST / family).glob("*.py")} - set(before)
        if extra or set(after) != set(before):
            raise SystemExit(f"{family}: files differ from the generator's set: {sorted(extra or set(before) - set(after))}")
        patch = make_patch(family, before, after)
        patch_file = PATCHES / f"{family}.patch"
        if patch:
            patch_file.write_text(patch)
        elif patch_file.exists():
            patch_file.unlink()
    for patch_file in PATCHES.glob("*.patch"):
        if patch_file.stem not in families:
            raise SystemExit(f"stale patch for a family that is not vendored: {patch_file.name}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(statuses):
    """native/MANIFEST.json: every generated file -> sha256 and verbatim/patched/generated."""

    manifest = {
        f"{family}/{name}": {"sha256": _sha256(DEST / family / name), "status": status}
        for family, files in sorted(statuses.items())
        for name, status in sorted(files.items())
    }
    manifest["auto/_registry.py"] = {"sha256": _sha256(DEST / "auto" / "_registry.py"), "status": "generated"}
    (DEST / "MANIFEST.json").write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n")


TASK_TABLES = {
    "CONFIG": ("configuration_auto", "CONFIG_MAPPING_NAMES"),
    "MODEL": ("modeling_auto", "MODEL_MAPPING_NAMES"),
    "CAUSAL_LM": ("modeling_auto", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"),
    "IMAGE_TEXT_TO_TEXT": ("modeling_auto", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"),
    "MULTIMODAL_LM": ("modeling_auto", "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES"),
    "PRETRAINING": ("modeling_auto", "MODEL_FOR_PRETRAINING_MAPPING_NAMES"),
    "TEXT_TO_WAVEFORM": ("modeling_auto", "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES"),
    "PROCESSOR": ("processing_auto", "PROCESSOR_MAPPING_NAMES"),
    "TOKENIZER": ("tokenization_auto", "TOKENIZER_MAPPING_NAMES"),
    "IMAGE_PROCESSOR": ("image_processing_auto", "IMAGE_PROCESSOR_MAPPING_NAMES"),
    "VIDEO_PROCESSOR": ("video_processing_auto", "VIDEO_PROCESSOR_MAPPING_NAMES"),
    "FEATURE_EXTRACTOR": ("feature_extraction_auto", "FEATURE_EXTRACTOR_MAPPING_NAMES"),
}


def _table(module, name):
    from importlib import import_module

    return getattr(import_module(f"transformers.models.auto.{module}"), name)


def _class_name(value):
    """Pick one class name from a mapping value (str, tuple, or backend dict)."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("torchvision") or value.get("pil") or next(iter(value.values()), None)
    if isinstance(value, (tuple, list)):
        return next((item for item in value if item), None)
    return None


def _class_home(cls_name):
    """Return ("family", family, module) for model-family classes, ("framework", module) otherwise."""

    import transformers

    value = getattr(transformers, cls_name, None)
    module = getattr(value, "__module__", None)
    if module is None:
        return None
    if module.startswith("transformers.models."):
        parts = module.split(".")
        return ("family", parts[2], parts[3])
    return ("framework", module)


def required_families(families):
    """Families providing any class the vendored families' tables reference."""

    needed = set()
    for key, (module, name) in TASK_TABLES.items():
        for model_type, value in _table(module, name).items():
            if family_of(model_type) not in families:
                continue
            home = _class_home(_class_name(value) or "")
            if home and home[0] == "family":
                needed.add(home[1])
    return needed


def registry(families):
    """model_type -> (module, class) tables for vendored families only."""

    vendored = set(families)
    tables = {}
    for key, (module, name) in TASK_TABLES.items():
        result = {}
        for model_type, value in _table(module, name).items():
            if family_of(model_type) not in vendored:
                continue
            cls_name = _class_name(value)
            home = _class_home(cls_name or "")
            if home is None:
                continue
            if home[0] == "framework":
                result[model_type] = [home[1], cls_name]
            elif home[1] in vendored:
                result[model_type] = [f"gptqmodel.models.ports.native.{home[1]}.{home[2]}", cls_name]
        # zai-org/GLM-OCR declares GLM-4.6V's processor in processor_config.json.
        if key in {"PROCESSOR", "IMAGE_PROCESSOR", "VIDEO_PROCESSOR"} and "glm_ocr" in vendored and "glm46v" in result:
            result["glm_ocr"] = result["glm46v"]
        tables[key] = result
    tables.update(tokenizer_selection(families))
    return tables


def tokenizer_selection(families):
    """Data Transformers 5.23 ``AutoTokenizer`` selects classes with, for NativeTokenizer.

    ``TOKENIZER_CLASSES``: every vendored tokenizer class name -> module (a
    checkpoint may declare any of them, not only the ones a model_type maps to).
    ``TOKENIZER_HUB_CLASS_OVERRIDES`` / ``TOKENIZER_BACKEND_MODEL_IDS``: the
    model types whose hub-declared class is known wrong and the model-id
    patterns forced onto ``TokenizersBackend``.
    """

    import ast
    from importlib import import_module

    classes = {}
    for family in families:
        for path in sorted((DEST / family).glob("tokenization_*.py")):
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                    for name in ast.literal_eval(node.value):
                        classes.setdefault(name, f"gptqmodel.models.ports.native.{family}.{path.stem}")
    auto = import_module("transformers.models.auto.tokenization_auto")
    return {
        "TOKENIZER_CLASSES": dict(sorted(classes.items())),
        "TOKENIZER_HUB_CLASS_OVERRIDES": sorted(auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS),
        "TOKENIZER_BACKEND_MODEL_IDS": list(auto.MODEL_IDS_TO_TOKENIZERS_BACKEND),
    }


def vendored_families():
    return sorted(p.name for p in DEST.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"})


def statuses_of(family):
    patch_file = PATCHES / f"{family}.patch"
    patched = set()
    if patch_file.exists():
        patched = {line[len("+++ "):].strip().split("/", 2)[2] for line in patch_file.read_text().splitlines() if line.startswith("+++ ")}
    return {name: "patched" if name in patched else "verbatim" for name in verbatim(family)}


def generate(src, repo, requested, refresh):
    """Vendor ``requested`` (plus closure) into <repo>, then write the registry and the manifest."""

    _configure(src, repo)
    families = closure(requested)
    while True:
        extra = required_families(set(families)) - set(families)
        if not extra:
            break
        families = closure(sorted(set(families) | extra))
    DEST.mkdir(parents=True, exist_ok=True)
    written = {}
    for family in families:
        # Existing families are kept unless --refresh is given; with --refresh the recorded
        # patches re-apply GPT-QModel's 5.23 modernizations on top of the fresh copy.
        if (DEST / family).is_dir() and not refresh:
            continue
        written[family] = vendor(family)
    reg = registry(vendored_families())
    (DEST / "auto").mkdir(exist_ok=True)
    (DEST / "auto" / "_registry.py").write_text(
        HEADER + "# Generated: model_type -> (module, class) for vendored families only.\n"
        + "".join(f"{k} = {json.dumps(v, indent=4, sort_keys=True)}\n" for k, v in reg.items())
    )
    write_manifest({family: written.get(family) or statuses_of(family) for family in vendored_families()})
    return families, written


def differences(generated_root, committed_root):
    """Generated files (relative paths) whose bytes differ from, or are missing in, the committed tree."""

    generated = Path(generated_root).joinpath(*NATIVE_PARTS)
    committed = Path(committed_root).joinpath(*NATIVE_PARTS)
    return sorted(
        path.relative_to(generated).as_posix()
        for path in generated.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and (not (committed / path.relative_to(generated)).is_file()
             or path.read_bytes() != (committed / path.relative_to(generated)).read_bytes())
    )


def main(argv):
    flags = {arg for arg in argv[2:] if arg.startswith("--")}
    requested = [arg for arg in argv[2:] if not arg.startswith("--")]
    src, repo = argv[0], argv[1]
    _configure(src, repo)
    if "--all" in flags or "--record-patches" in flags:
        requested = sorted(set(requested) | set(vendored_families()))
    if "--record-patches" in flags:
        record_patches(vendored_families())
        # The recorded patches must reproduce every committed file before the manifest is rewritten.
        with tempfile.TemporaryDirectory() as tmp:
            generate(src, tmp, requested, refresh=True)
            stale = [p for p in differences(tmp, repo) if p != "MANIFEST.json"]
            if stale:
                raise SystemExit(f"recorded patches do not reproduce: {stale}")
        _configure(src, repo)
        write_manifest({family: statuses_of(family) for family in vendored_families()})
        print(json.dumps({"patches": sorted(p.stem for p in PATCHES.glob("*.patch"))}))
        return
    families, written = generate(src, repo, requested, "--refresh" in flags)
    print(json.dumps({"families": len(families), "files": sum(len(v) for v in written.values())}))
    print(" ".join(families))


if __name__ == "__main__":
    main(sys.argv[1:])
