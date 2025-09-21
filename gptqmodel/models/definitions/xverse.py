# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from . import LlamaQModel


class XverseQModel(LlamaQModel):
    require_pkgs_version = ["transformers<=4.38.2", "tokenizers<=0.15.2"]
