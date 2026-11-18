# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 EXAONE 4 classes and the released EXAONE 4.5 text-config alias.

The modeling is the verbatim 5.23 family under ``ports/native/exaone4``; the
EXAONE 4.5 config performs the same ``exaone4_5_text`` -> ``exaone4`` migration
for its nested language model.
"""

from ..native.exaone4.configuration_exaone4 import Exaone4Config
from ..native.exaone4.modeling_exaone4 import Exaone4ForCausalLM


class Exaone4TextConfig:
    """Load standalone text configs using the exact native architecture."""

    @classmethod
    def from_pretrained(cls, model_id_or_path, **kwargs):
        payload, kwargs = Exaone4Config.get_config_dict(model_id_or_path, **kwargs)
        if payload.get("model_type") not in {"exaone4", "exaone4_5_text"}:
            raise ValueError("EXAONE text loading requires exaone4 or exaone4_5_text")
        payload = dict(payload, model_type="exaone4")
        return Exaone4Config.from_dict(payload, **kwargs)


__all__ = ["Exaone4ForCausalLM", "Exaone4TextConfig"]
