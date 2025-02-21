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
from enum import Enum
from typing import Callable

from colorlog import ColoredFormatter

# global static/shared logger instance
logger = None
last_logging_src = 1 # one for logger, 2 for progressbar

def update_logging_src(src: int):
    global last_logging_src
    last_logging_src = src

class LEVEL(str, Enum):
    CRITICAL = "CRITICAL"
    DEBUG = "DEBUG"
    WARN = "WARNING"
    INFO = "INFO"
    ERROR = "ERROR"

def setup_logger():
    global logger
    if logger is not None:
        return logger

    class CustomLogger(logging.Logger):
        history = set()
        history_limit = 1000

        def history_add(self, msg) -> bool:
            h = hash(msg) # TODO only msg is checked not level + msg
            if h in self.history:
                return False # add failed since it already exists

            if len(self.history) > self.history_limit:
                self.history.clear()

            self.history.add(h)

            return False

        class critical_cls:
            def __init__(self, logger):
                self.logger = logger

            def once(self, msg, *args, **kwargs):
                if self.logger.history_add(msg):
                    self(msg, *args, **kwargs)

            def __call__(self, msg, *args, **kwargs):
                self.logger._process(LEVEL.CRITICAL, msg, *args, **kwargs)

        class warn_cls:
            def __init__(self, logger):
                self.logger = logger

            def once(self, msg, *args, **kwargs):
                if self.logger.history_add(msg):
                    self(msg, *args, **kwargs)

            def __call__(self, msg, *args, **kwargs):
                self.logger._process(LEVEL.WARN, msg, *args, **kwargs)

        class debug_cls:
            def __init__(self, logger):
                self.logger = logger

            def once(self, msg, *args, **kwargs):
                if self.logger.history_add(msg):
                    self(msg, *args, **kwargs)

            def __call__(self, msg, *args, **kwargs):
                self.logger._process(LEVEL.DEBUG, msg, *args, **kwargs)

        class info_cls:
            def __init__(self, logger):
                self.logger = logger

            def once(self, msg, *args, **kwargs):
                if self.logger.history_add(msg):
                    self(msg, *args, **kwargs)

            def __call__(self, msg, *args, **kwargs):
                self.logger._process(LEVEL.INFO, msg, *args, **kwargs)

        class error_cls:
            def __init__(self, logger):
                self.logger = logger

            def once(self, msg, *args, **kwargs):
                if self.logger.history_add(msg):
                    self(msg, *args, **kwargs)

            def __call__(self, msg, *args, **kwargs):
                self.logger._process(LEVEL.ERROR, msg, *args, **kwargs)

        def __init__(self, name):
            super().__init__(name)
            self._critical = self.critical
            self._warning = self.warning
            self._debug = self.debug
            self._info = self.info
            self._error = self.error

            self.critical = self.critical_cls(logger=self)
            self.warn = self.warn_cls(logger=self)
            self.debug = self.debug_cls(logger=self)
            self.info = self.info_cls(logger=self)
            self.error = self.error_cls(logger=self)

        def _process(self, level: LEVEL, msg, *args, **kwargs):
            global last_logging_src
            if last_logging_src == 2:
                print(" ", flush=True)
                last_logging_src = 1

            if level == LEVEL.INFO:
                self._info(msg, *args, **kwargs)
            elif level == LEVEL.WARN:
                self._warning(msg, *args, **kwargs)
            elif level == LEVEL.ERROR:
                self._error(msg, *args, **kwargs)
            elif level == LEVEL.DEBUG:
                self._debug(msg, *args, **kwargs)
            elif level == LEVEL.CRITICAL:
                self._critical(msg, *args, **kwargs)

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

    return logger


