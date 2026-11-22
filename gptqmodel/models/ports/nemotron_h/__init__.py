# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 Nemotron-H classes, imported only on loading."""

from ..native.nemotron_h.configuration_nemotron_h import NemotronHConfig
from ..native.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM


__all__ = ["NemotronHConfig", "NemotronHForCausalLM"]
