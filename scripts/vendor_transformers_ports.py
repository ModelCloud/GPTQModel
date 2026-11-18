# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendor Transformers 5.23 model families into gptqmodel/models/ports/native.

Usage: python scripts/vendor_transformers_ports.py <transformers_src_pkg_dir> <repo_root> <family> [<family> ...]

Copies configuration/modeling/processing/... modules verbatim, rewrites framework
imports (`from ...x` -> `from transformers.x`), keeps sibling-family relative imports
(`from ..other.x` resolves to ports/native/other), and routes `..auto` to a
package-local registry. Adds the dependency closure (relative imports and
SubConfigSpec default model types). Generates ports/native/auto/_registry.py.
"""

import json
import re
import sys
from pathlib import Path


SRC = Path(sys.argv[1])  # .../src/transformers
REPO = Path(sys.argv[2])
REQUESTED = sys.argv[3:]
MODELS = SRC / "models"
DEST = REPO / "gptqmodel" / "models" / "ports" / "native"

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


def rewrite(text):
    text = ABSOLUTE_MODELS.sub(lambda m: "gptqmodel.models.ports.native." + m.group(1) + ".", text)
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


def vendor(family):
    out = DEST / family
    out.mkdir(parents=True, exist_ok=True)
    (out / "__init__.py").write_text(HEADER)
    written = []
    for path in family_files(family):
        text = rewrite(path.read_text())
        (out / path.name).write_text(HEADER + text)
        written.append(path.name)
    return written


def registry(families):
    """Static model_type -> (module, class) tables for vendored families.

    Built from the installed Transformers 5.23 auto tables (the same release
    being vendored), restricted to vendored families.
    """

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES

    vendored = set(families)

    def locate(family, cls):
        for path in family_files(family):
            if re.search(rf"^class {re.escape(cls)}\b", path.read_text(), re.M):
                return f"gptqmodel.models.ports.native.{family}.{path.stem}"
        return None

    def build(names):
        result = {}
        for mt, cls in names.items():
            if not isinstance(cls, str):
                continue
            family = family_of(mt)
            if family not in vendored:
                continue
            module = locate(family, cls)
            if module:
                result[mt] = [module, cls]
        return result

    return {
        "CONFIG": build(CONFIG_MAPPING_NAMES),
        "MODEL": build(MODEL_MAPPING_NAMES),
        "CAUSAL_LM": build(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
    }


def main():
    families = closure(REQUESTED)
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for family in families:
        manifest[family] = vendor(family)
    reg = registry(sorted(p.name for p in DEST.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"}))
    (DEST / "auto").mkdir(exist_ok=True)
    (DEST / "auto" / "_registry.py").write_text(
        HEADER + "# Generated: model_type -> (module, class) for vendored families only.\n"
        + "".join(f"{k} = {json.dumps(v, indent=4, sort_keys=True)}\n" for k, v in reg.items())
    )
    print(json.dumps({"families": len(families), "files": sum(len(v) for v in manifest.values())}))
    print(" ".join(families))


if __name__ == "__main__":
    main()
