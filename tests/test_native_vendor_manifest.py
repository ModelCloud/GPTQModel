# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The vendored Transformers tree is exactly what scripts/vendor_transformers_ports.py generates.

GPT-QModel's hand edits to vendored files are recorded as scripts/vendor_patches/<family>.patch and
applied by the generator, so regenerating from Transformers 5.23 reproduces the committed tree byte for
byte. native/MANIFEST.json pins every generated file (sha256, verbatim/patched/generated).
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
NATIVE = REPO / "gptqmodel" / "models" / "ports" / "native"
SCRIPT = REPO / "scripts" / "vendor_transformers_ports.py"
PATCHES = REPO / "scripts" / "vendor_patches"


def _generator():
    spec = importlib.util.spec_from_file_location("vendor_transformers_ports", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest():
    return json.loads((NATIVE / "MANIFEST.json").read_text())


def _tree():
    return sorted(
        path.relative_to(NATIVE).as_posix()
        for path in NATIVE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def _patched_files():
    found = set()
    for patch in PATCHES.glob("*.patch"):
        for line in patch.read_text().splitlines():
            if line.startswith("+++ "):
                found.add(line[len("+++ b/"):].strip())
    return found


def test_manifest_pins_every_vendored_file():
    generator = _generator()
    manifest = _manifest()
    tree = set(_tree())
    assert tree - set(manifest) == set(generator.HAND_WRITTEN), sorted(tree - set(manifest) - set(generator.HAND_WRITTEN))
    assert not set(manifest) - tree, sorted(set(manifest) - tree)
    for relative, entry in manifest.items():
        digest = hashlib.sha256((NATIVE / relative).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{relative} changed without re-recording (run --record-patches)"
    patched = _patched_files()
    assert {k for k, v in manifest.items() if v["status"] == "patched"} == patched
    assert {k for k, v in manifest.items() if v["status"] == "generated"} == {"auto/_registry.py"}
    assert {v["status"] for v in manifest.values()} <= {"verbatim", "patched", "generated"}


def _transformers_source():
    """The Transformers 5.23 package sources: $GPTQMODEL_TRANSFORMERS_SRC or the installed 5.23 wheel."""

    explicit = os.environ.get("GPTQMODEL_TRANSFORMERS_SRC")
    if explicit:
        return Path(explicit)
    import transformers

    if transformers.__version__ != "5.23.0":
        return None
    source = Path(transformers.__file__).resolve().parent
    return source if (source / "models").is_dir() else None


def test_generator_reproduces_the_committed_tree(tmp_path):
    source = _transformers_source()
    if source is None or not (source / "models").is_dir():
        pytest.skip("Transformers 5.23 source tree not available")
    generator = _generator()
    families = sorted(p.name for p in NATIVE.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"})
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(tmp_path), *families, "--refresh"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert generator.differences(tmp_path, REPO) == []
    generated = sorted(
        path.relative_to(tmp_path.joinpath(*generator.NATIVE_PARTS)).as_posix()
        for path in tmp_path.joinpath(*generator.NATIVE_PARTS).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    # Nothing committed beyond the generator's output and the few hand-written package files.
    assert set(_tree()) - set(generated) == set(generator.HAND_WRITTEN) - {"MANIFEST.json"}


def test_patches_apply_strictly():
    generator = _generator()
    before = {"m.py": "a\nb\nc\nd\ne\nf\ng\n"}
    after = {"m.py": "a\nb\nc\nD\ne\nf\ng\nh\n"}
    patch = generator.make_patch("fam", before, after)
    files = dict(before)
    assert generator.apply_patch(files, patch, "fam") == {"m.py"}
    assert files == after
    # Upstream drift under a hunk must fail loudly instead of silently mis-applying.
    drifted = {"m.py": "a\nb\nC\nd\ne\nf\ng\n"}
    with pytest.raises(SystemExit, match="does not apply"):
        generator.apply_patch(drifted, patch, "fam")
    with pytest.raises(SystemExit, match="not add or remove"):
        generator.make_patch("fam", before, {"m.py": before["m.py"], "new.py": "x\n"})
