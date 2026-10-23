# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Iterable

from .torch import TorchLinear


def configure_lookahead_chain(modules: Iterable[TorchLinear]):
    """Deprecated compatibility helper; dense lookahead is disabled with a warning."""

    for module in modules:
        module.enable_lookahead(True).set_lookahead_next(None)


def configure_default_lookahead(model) -> None:
    """Deprecated compatibility helper; no dense weights or module links are retained."""

    configure_lookahead_chain(module for module in model.modules() if isinstance(module, TorchLinear))
