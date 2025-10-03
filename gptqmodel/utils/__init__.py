# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .backend import BACKEND
from .logger import setup_logger
from .python import gte_python_3_13_3, gte_python_3_14, has_gil_control, has_gil_disabled, log_gil_requirements_for
from .threads import AsyncManager, SerialWorker
from .vram import get_vram


log = setup_logger()

ASYNC_BG_QUEUE = AsyncManager(threads=4)
SERIAL_BG_QUEUE = SerialWorker()

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
