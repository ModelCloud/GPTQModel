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

from .backend import BACKEND
from .logger import setup_logger
from .python import gte_python_3_13_3, has_gil_control, has_gil_disabled, log_gil_requirements_for

log = setup_logger()

# TODO: datasets is not compatible with free threading
if has_gil_disabled():
    log.info("Python GIL is disabled and GPTQModel will auto enable multi-gpu quant acceleration for MoE models plus multi-cpu accelerated packing.")
    from .perplexity import Perplexity
else:
    if has_gil_control():
        log.warn(
            "Python >= 3.13T (free-threading) version detected but GIL is not disabled due to manual override or `regex` package compatibility which can be ignored. Please disable GIL via env `PYTHON_GIL=0`.")

    log.warn(
        "Python GIL is enabled: Multi-gpu quant acceleration for MoE models is sub-optimal and multi-core accelerated cpu packing is also disabled. We recommend Python >= 3.13.3t with Pytorch > 2.8 for mult-gpu quantization and multi-cpu packing with env `PYTHON_GIL=0`.")

    log_gil_requirements_for("utils/Perplexity")



from .vram import get_vram
