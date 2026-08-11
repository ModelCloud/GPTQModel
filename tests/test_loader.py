# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap

from gptqmodel.models.loader import ModelLoader, _should_print_module_tree


def test_loader_module_tree_print_is_opt_in(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_PRINT_MODULE_TREE", raising=False)
    assert _should_print_module_tree() is False

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "1")
    assert _should_print_module_tree() is True

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "off")
    assert _should_print_module_tree() is False


def test_quantized_loader_forwards_dtype_to_final_kernel_selection():
    tree = ast.parse(textwrap.dedent(inspect.getsource(ModelLoader)))
    selection_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "select_quant_linear"
    ]

    assert len(selection_calls) == 1
    dtype_args = [keyword.value for keyword in selection_calls[0].keywords if keyword.arg == "dtype"]
    assert len(dtype_args) == 1
    assert isinstance(dtype_args[0], ast.Name)
    assert dtype_args[0].id == "dtype"
