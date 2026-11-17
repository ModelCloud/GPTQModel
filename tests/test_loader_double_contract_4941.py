# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every ``ModelLoader`` test double must declare ``require_load_tokenizer`` (#4941).

#4894 made ``ModelLoader`` read ``cls.require_load_tokenizer`` with
``BaseQModel``'s default.  Doubles that mirror ``BaseQModel``'s attributes by
hand and forget this one raise ``AttributeError`` before the loader does any
work, and it took two rounds to shake out (#4926, #4941) because each fix only
swept the file that was reported.

This guard walks every test module and finds the classes decorated with
``ModelLoader`` - they are usually defined inside a test function, so importing
the modules would not expose them - and requires the attribute to be declared.
"""

from __future__ import annotations

import ast
import pathlib


TESTS_ROOT = pathlib.Path(__file__).resolve().parent
ATTRIBUTE = "require_load_tokenizer"
# tests/test_local_model_paths.py has nine doubles, plus the two from #4941.
MINIMUM_DOUBLES = 11


def _decorator_names(node: ast.ClassDef):
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        yield ".".join(reversed(parts))


def _declared_attributes(node: ast.ClassDef) -> set[str]:
    names = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        else:
            continue
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _model_loader_doubles():
    for path in sorted(TESTS_ROOT.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if "ModelLoader" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(name.split(".")[-1] == "ModelLoader" for name in _decorator_names(node)):
                yield path, node


def test_every_model_loader_double_declares_require_load_tokenizer():
    doubles = list(_model_loader_doubles())
    # Sanity check: a refactor that hides the doubles would silently disable
    # this guard.
    assert len(doubles) >= MINIMUM_DOUBLES, (
        f"expected at least {MINIMUM_DOUBLES} ModelLoader doubles in tests/, "
        f"found {len(doubles)}"
    )
    missing = [
        f"{path.name}:{node.lineno} {node.name}"
        for path, node in doubles
        if ATTRIBUTE not in _declared_attributes(node)
    ]
    assert not missing, (
        f"ModelLoader test doubles must declare {ATTRIBUTE} "
        f"(BaseQModel's default is True): " + ", ".join(missing)
    )


def test_the_loader_doubles_from_4941_are_covered():
    covered = {(path.name, node.name) for path, node in _model_loader_doubles()}
    assert ("test_gguf_load_policy_env_removal.py", "DummyQModel") in covered
    assert ("test_lazy_turtle_conversion_cache.py", "_DummyQModel") in covered


def test_base_q_model_declares_the_default_the_doubles_mirror():
    from gptqmodel.models.base import BaseQModel

    assert BaseQModel.require_load_tokenizer is True
