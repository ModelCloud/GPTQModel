# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.models.loader import _should_print_module_tree


def test_loader_module_tree_print_is_opt_in(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_PRINT_MODULE_TREE", raising=False)
    assert _should_print_module_tree() is False

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "1")
    assert _should_print_module_tree() is True

    monkeypatch.setenv("GPTQMODEL_PRINT_MODULE_TREE", "off")
    assert _should_print_module_tree() is False
