# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native Transformers 5 Nemotron-H classes, imported only on loading."""

from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM


__all__ = ["NemotronHConfig", "NemotronHForCausalLM"]
