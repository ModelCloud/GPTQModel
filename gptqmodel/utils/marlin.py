# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ..nn_modules.qlinear.marlin import MarlinQuantLinear
from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger
from .rocm import IS_ROCM

log = setup_logger()

# Validate marlin support
def _validate_marlin_device_support() -> bool:
    """
    Validates if the current device is compatible for Marlin.
    ref: https://github.com/IST-DASLab/marlin?tab=readme-ov-file#requirements

    Returns:
        bool: indicates if CUDA device is compatible for Marlin
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and not IS_ROCM


# Adapted from https://github.com/rib-2/marlin/tree/conversion
def _validate_marlin_compatibility(cfg: QuantizeConfig, throw_error: bool = False):
    validate, err = MarlinQuantLinear.validate(bits=cfg.bits, group_size=cfg.group_size, desc_act=cfg.desc_act, sym=cfg.sym, pack_dtype=cfg.pack_dtype, dynamic=cfg.dynamic)
    if throw_error and err is not None:
        raise ValueError(err)
    return err
