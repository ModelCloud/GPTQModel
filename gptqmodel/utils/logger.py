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

import logging
import sys
from typing import Callable

from colorlog import ColoredFormatter

# global static/shared logger instance
logger = None
last_logging_src = 1 # one for logger, 2 for progressbar

def update_logging_src(src: int):
    global last_logging_src
    last_logging_src = src

def setup_logger():
    global logger
    if logger is not None:
        return logger

    class CustomLogger(logging.Logger):
        def critical(self, msg, *args, **kwargs):
            op = super().critical
            self._process(op, msg, *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            op = super().warning
            self._process(op, msg, *args, **kwargs)

        def debug(self, msg, *args, **kwargs):
            op = super().debug
            self._process(op, msg, *args, **kwargs)

        def info(self, msg, *args, **kwargs):
            op = super().info
            self._process(op, msg, *args, **kwargs)

        def _process(self, op: Callable, msg, *args, **kwargs):
            global last_logging_src
            if last_logging_src == 2:
                print(" ", flush=True)
                last_logging_src = 1
            op(msg, *args, **kwargs)

    logging.setLoggerClass(CustomLogger)

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    # Create a colored formatter
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.flush = sys.stdout.flush
    logger.addHandler(handler)

    # fix warnings about warn() deprecated
    if hasattr(logger, "warning"):
        logger.warn = logger.warning

    return logger


