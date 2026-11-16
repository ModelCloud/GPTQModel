# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regenerate ``gptqmodel_nf12/abi.py`` from an nf12 checkout.

The pin records the ABI upstream *exposes* -- the declarations in
``include/nf12/nf12.h``, the torch schemas in ``csrc/torch/ops.cpp``, the symbol
map and the public Python names -- so the vendored copy in ``gptqmodel_ext``
cannot drift from it silently.

    git clone https://github.com/dphnAI/nf12 /tmp/nf12
    python tools/generate_nf12_abi.py --source /tmp/nf12

Review the diff before committing: a changed signature, struct field or schema
is an ABI break for every prebuilt library and every consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pathlib
import re
import subprocess
import sys


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _functions(text: str) -> dict[str, str]:
    body = _strip_comments(text)
    found = {}
    for match in re.finditer(
        r"\b(nf12_\w+|size_t|int64_t|const char\*)\s+(nf12_\w+)\s*\(([^;{]*?)\)\s*;", body, re.S
    ):
        found[match.group(2)] = f"{match.group(1)} {match.group(2)}(" + " ".join(match.group(3).split()) + ")"
    return found


def _structs(text: str) -> dict[str, tuple[str, ...]]:
    body = _strip_comments(text)
    found = {}
    for match in re.finditer(r"typedef struct (\w+) \{(.*?)\} (\w+);", body, re.S):
        found[match.group(3)] = tuple(" ".join(x.split()) for x in match.group(2).split(";") if x.strip())
    return found


def _status(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r"(NF12_\w+) = (\d+)", _strip_comments(text))}


def _version(text: str) -> tuple[int, int]:
    return tuple(int(x) for x in re.search(r"MAJOR (\d+).*?MINOR (\d+)", text, re.S).groups())


def _torch_ops(text: str) -> tuple[str, ...]:
    found = []
    for match in re.finditer(r"\bm\.def\((.*?)\)\s*;", text, re.S):
        schema = " ".join("".join(re.findall(r'"([^"]*)"', match.group(1))).split())
        if schema:
            found.append(schema)
    return tuple(found)


def _python_signatures(source: pathlib.Path, exports: tuple[str, ...]) -> dict[str, str]:
    sys.path.insert(0, str(source / "python"))
    try:
        import nf12  # noqa: PLC0415

        signatures = {}
        for name in exports:
            target = getattr(nf12, name, None)
            if callable(target):
                try:
                    signatures[name] = str(inspect.signature(target))
                except (TypeError, ValueError):
                    pass
        return signatures
    finally:
        sys.path.remove(str(source / "python"))


def _block(mapping: dict, indent: str = "    ") -> str:
    return "\n".join(f"{indent}{key!r}: {value!r}," for key, value in sorted(mapping.items()))


def _seq(items, indent: str = "        ") -> str:
    return "\n".join(f"{indent}{item!r}," for item in items)


def build(source: pathlib.Path) -> str:
    header = (source / "include/nf12/nf12.h").read_text()
    functions = _functions(header)
    structs = _structs(header)
    status = _status(header)
    version = _version(header)
    ops = _torch_ops((source / "csrc/torch/ops.cpp").read_text())
    export_map = tuple(re.findall(r"(nf12_\*)", (source / "csrc/nf12.map").read_text()))
    exports = tuple(re.findall(r'"(\w+)": "\.\w+"', (source / "python/nf12/__init__.py").read_text()))
    signatures = _python_signatures(source, exports)
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    digest = hashlib.sha256(
        json.dumps(
            {
                "repository": "dphnAI/nf12",
                "commit": commit,
                "version": version,
                "c_functions": functions,
                "structs": structs,
                "status": status,
                "torch_ops": {"nf12": ops},
                "export_map": export_map,
                "python_exports": exports,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    return f'''# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream NF12 ABI, so this vendored copy cannot drift silently.

The ABI we consume is the one upstream *exposes*: the declarations in
``include/nf12/nf12.h``, the torch operator schemas in ``csrc/torch/ops.cpp``,
the exported symbol map, and the public Python names. This module records that
surface for ``dphnAI/nf12@{commit[:9]}`` (NF12 {version[0]}.{version[1]}) and
:func:`verify` re-parses the vendored copies to prove they still match.

Additions are expected and allowed -- this tree adds ``decode_fp32``,
``sub_fp32``, the ``rowmajor_*`` family and the staged-linear entry points -- but
every name upstream exposes keeps its signature, struct layout, schema and
export. ``tests/test_nf12_abi_parity.py`` enforces that; ``tools/generate_nf12_abi.py``
regenerates this file after :data:`UPSTREAM_COMMIT` moves.
"""

from __future__ import annotations

import re
from pathlib import Path


UPSTREAM_REPOSITORY = "dphnAI/nf12"
UPSTREAM_COMMIT = {commit!r}
UPSTREAM_VERSION = {version!r}

# ``name -> "return name(args)"`` for every function upstream declares.
C_FUNCTIONS: dict[str, str] = {{
{_block(functions)}
}}

# ``struct -> (field declarations in order)``; order and types are the layout.
STRUCTS: dict[str, tuple[str, ...]] = {{
{_block(structs)}
}}

STATUS: dict[str, int] = {{
{_block(status)}
}}

# ``namespace -> schemas`` as registered with torch.library.
TORCH_OPS: dict[str, tuple[str, ...]] = {{
    "nf12": (
{_seq(ops)}
    ),
}}

# Glob patterns the version script keeps global; everything else is local.
EXPORT_MAP: tuple[str, ...] = {export_map!r}

# Public names of the upstream ``nf12`` package (its ``__all__``).
PYTHON_EXPORTS: tuple[str, ...] = (
{_seq(exports, "    ")}
)

# ``str(inspect.signature(...))`` for the upstream callables above.
PYTHON_SIGNATURES: dict[str, str] = {{
{_block(signatures)}
}}

ABI_DIGEST = {digest!r}

_ROOT = Path(__file__).resolve().parent.parent
HEADER = "gptqmodel_ext/nf12/include/nf12/nf12.h"
TORCH_OPS_SOURCE = "gptqmodel_ext/nf12/csrc/torch/ops.cpp"
EXPORT_MAP_SOURCE = "gptqmodel_ext/nf12/csrc/nf12.map"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\\*.*?\\*/", "", text, flags=re.S)
    return re.sub(r"//[^\\n]*", "", text)


def parse_header(path: Path | str):
    """Return ``(functions, structs, status, version)`` parsed from an ``nf12.h``."""
    text = Path(path).read_text()
    body = _strip_comments(text)
    functions = {{}}
    for match in re.finditer(
        r"\\b(nf12_\\w+|size_t|int64_t|const char\\*)\\s+(nf12_\\w+)\\s*\\(([^;{{]*?)\\)\\s*;", body, re.S
    ):
        functions[match.group(2)] = f"{{match.group(1)}} {{match.group(2)}}(" + " ".join(match.group(3).split()) + ")"
    structs = {{}}
    for match in re.finditer(r"typedef struct (\\w+) \\{{(.*?)\\}} (\\w+);", body, re.S):
        structs[match.group(3)] = tuple(" ".join(x.split()) for x in match.group(2).split(";") if x.strip())
    status = {{key: int(value) for key, value in re.findall(r"(NF12_\\w+) = (\\d+)", body)}}
    version = tuple(int(x) for x in re.search(r"MAJOR (\\d+).*?MINOR (\\d+)", text, re.S).groups())
    return functions, structs, status, version


def parse_torch_ops(path: Path | str) -> tuple[str, ...]:
    """Return the operator schemas registered by a ``TORCH_LIBRARY`` source."""
    found = []
    for match in re.finditer(r"\\bm\\.def\\((.*?)\\)\\s*;", Path(path).read_text(), re.S):
        schema = " ".join("".join(re.findall(r'"([^"]*)"', match.group(1))).split())
        if schema:
            found.append(schema)
    return tuple(found)


def parse_export_map(path: Path | str) -> tuple[str, ...]:
    return tuple(re.findall(r"(nf12_\\*)", Path(path).read_text()))


def digest() -> str:
    """Stable digest of the pinned surface, for release notes and CI logs."""
    return ABI_DIGEST


def verify() -> None:
    """Raise when the vendored copy stops matching the pinned upstream ABI."""
    functions, structs, status, version = parse_header(_ROOT / HEADER)
    if version != UPSTREAM_VERSION:
        raise ValueError(f"nf12 ABI: vendored header is version {{version}}, pinned {{UPSTREAM_VERSION}}")
    missing = sorted(set(C_FUNCTIONS) - set(functions))
    if missing:
        raise ValueError(f"nf12 ABI: vendored header is missing upstream functions {{missing}}")
    drift = [name for name, signature in C_FUNCTIONS.items() if functions[name] != signature]
    if drift:
        raise ValueError(f"nf12 ABI: signature drift in {{drift}}")
    for name, fields in STRUCTS.items():
        if structs.get(name) != fields:
            raise ValueError(f"nf12 ABI: struct {{name}} layout drift: {{structs.get(name)}}")
    for name, value in STATUS.items():
        if status.get(name) != value:
            raise ValueError(f"nf12 ABI: status {{name}} is {{status.get(name)}}, pinned {{value}}")
    missing_ops = sorted(set(TORCH_OPS["nf12"]) - set(parse_torch_ops(_ROOT / TORCH_OPS_SOURCE)))
    if missing_ops:
        raise ValueError(f"nf12 ABI: vendored torch ops are missing upstream schemas {{missing_ops}}")
    if parse_export_map(_ROOT / EXPORT_MAP_SOURCE) != EXPORT_MAP:
        raise ValueError("nf12 ABI: export map drifted from upstream")


def verify_python_surface(package) -> None:
    """Raise when the package drops a public name or re-signs an upstream callable."""
    import inspect

    missing = sorted(set(PYTHON_EXPORTS) - set(package.__all__))
    if missing:
        raise ValueError(f"nf12 ABI: package is missing upstream exports {{missing}}")
    for name, expected in PYTHON_SIGNATURES.items():
        actual = str(inspect.signature(getattr(package, name)))
        if actual != expected:
            raise ValueError(f"nf12 ABI: {{name}} signature {{actual}} != upstream {{expected}}")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to an nf12 checkout")
    parser.add_argument(
        "--output",
        default=str(pathlib.Path(__file__).resolve().parent.parent / "gptqmodel_nf12" / "abi.py"),
        help="file to write (default: gptqmodel_nf12/abi.py)",
    )
    args = parser.parse_args()
    text = build(pathlib.Path(args.source).resolve())
    pathlib.Path(args.output).write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
