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

import datetime
import os
import sys
import time
from typing import Iterable
from warnings import warn

from gptqmodel.utils.logger import setup_logger, update_logging_src

logger = setup_logger()

class ProgressBarWarning(Warning):
    def __init__(self, msg, fp_write=None, *a, **k):
        if fp_write is not None:
            fp_write("\n" + self.__class__.__name__ + ": " + str(msg).rstrip() + '\n')
        else:
            super().__init__(msg, *a, **k)

class ProgressBar:
    def __init__(self,
                 iterable: Iterable=None,
                 total=None,
                 prefix:str = '',
                 bar_length:int =60,
                 fill:str = '█',
                 info:str = ""):

        # max info length over the life ot the pb
        self.max_info_length = len(info)

        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        elif total is not None and iterable is None:
            iterable = range(total)
        if total == float("inf"):
            # Infinite iterations, behave same as unknown
            total = None

        self.iterable = iterable
        self.total = total

        self.prefix = prefix
        self.bar_length = bar_length
        self.fill = fill
        self.info_text = info
        self.current_iteration = 0
        self.time = time.time()

    def info(self, info:str):
        if len(info) > self.max_info_length:
            self.max_info_length = len(info)

        self.info_text = info

    def progress(self, iteration:int = None):
        if not iteration:
            iteration = self.current_iteration

        columns, _ = terminal_size()
        bar_length = columns
        bar_length -= len(self.prefix) # +1 for space
        bar_length -= len(self.info_text)

        percent_num = iteration / float(len(self))
        percent = ("{0:.1f}").format(100 * (percent_num))
        log = f"{self.calc_time(iteration)} [{iteration}/{len(self)}] {percent}%"

        bar_length -= len(log)
        bar_length -= 5 # space + | chars

        # calculate padding
        if len(self.info_text) < self.max_info_length:
            padding = " " * (self.max_info_length - len(self.info_text))
        else:
            padding = ""

        bar_length -= len(padding)

        filled_length = int(bar_length * iteration // len(self))
        bar = self.fill * filled_length + '-' * (bar_length - filled_length)
        self.log(bar=bar, log=log, padding=padding, end='\n' if percent_num >= 1.0 else '')

    def calc_time(self, iteration):
        used_time = int(time.time() - self.time)
        formatted_time = str(datetime.timedelta(seconds=used_time))
        remaining = str(datetime.timedelta(seconds=int((used_time / max(iteration, 1)) * len(self))))
        return f"{formatted_time} / {remaining}"

    def log(self, bar:str, log:str, padding:str = "", end: str = ""):
        # print(f'\r{self.prefix} {self.info_text} |{bar}| {log}', end='', flush=True)
        if self.prefix:
            print(f'\r{self.prefix} {self.info_text}{padding} |{bar}| {log}', end=end, flush=True)
        else:
            print(f'\r{self.info_text}{padding} |{bar}| {log}', end=end, flush=True)

        update_logging_src(src=2)  # let logger now we logged

    def __bool__(self):
        if self.total is not None:
            return self.total > 0
        if self.iterable is None:
            raise TypeError('bool() undefined when iterable == total == None')
        return bool(self.iterable)

    def __len__(self):
        return (
            self.total if self.iterable is None
            else self.iterable.shape[0] if hasattr(self.iterable, "shape")
            else len(self.iterable) if hasattr(self.iterable, "__len__")
            else self.iterable.__length_hint__() if hasattr(self.iterable, "__length_hint__")
            else getattr(self, "total", None))

    # TODO FIXME: I have no cluse why the try/catch is catching nothing here
    def __reversed__(self):
        try:
            orig = self.iterable
        except AttributeError:
            raise TypeError("'progress' object is not reversible")
        else:
            self.iterable = reversed(self.iterable)
            return self.__iter__()
        finally:
            self.iterable = orig

    def __contains__(self, item):
        contains = getattr(self.iterable, '__contains__', None)
        return contains(item) if contains is not None else item in self.__iter__()

    def __enter__(self):
        return self

    # TODO FIXME: I don't understand the exception here. What are we catching? yield error?
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except AttributeError:
            # maybe eager thread cleanup upon external error
            if (exc_type, exc_value, traceback) == (None, None, None):
                raise
            warn("AttributeError ignored", ProgressBarWarning, stacklevel=2)

    def __del__(self):
        self.close()

    @property
    def _comparable(self):
        return abs(getattr(self, "pos", 1 << 31))

    def __hash__(self):
        return id(self)

    def __iter__(self):
        iterable = self.iterable

        for obj in iterable:
            self.current_iteration+=1
            self.progress()
            yield obj

        self.progress()
        return

    def close(self):
        pass
        #self.log(f"{self.fill * self.bar_length}", "100.0%", end="\n")

# copied from github.com/onsim/shutils
def terminal_size(fallback=(80, 24)):
    """Get the size of the terminal window.

    For each of the two dimensions, the environment variable, COLUMNS
    and LINES respectively, is checked. If the variable is defined and
    the value is a positive integer, it is used.

    When COLUMNS or LINES is not defined, which is the common case,
    the terminal connected to sys.__stdout__ is queried
    by invoking os.get_terminal_size.

    If the terminal size cannot be successfully queried, either because
    the system doesn't support querying, or because we are not
    connected to a terminal, the value given in fallback parameter
    is used. Fallback defaults to (80, 24) which is the default
    size used by many terminal emulators.

    The value returned is a named tuple of type os.terminal_size.
    """
    # columns, lines are the working values
    try:
        columns = int(os.environ['COLUMNS'])
    except (KeyError, ValueError):
        columns = 0

    try:
        lines = int(os.environ['LINES'])
    except (KeyError, ValueError):
        lines = 0

    # only query if necessary
    if columns <= 0 or lines <= 0:
        try:
            size = os.get_terminal_size(sys.__stdout__.fileno())
        except (AttributeError, ValueError, OSError):
            # stdout is None, closed, detached, or not a terminal, or
            # os.get_terminal_size() is unsupported
            size = os.terminal_size(fallback)
        if columns <= 0:
            columns = size.columns or fallback[0]
        if lines <= 0:
            lines = size.lines or fallback[1]

    return (columns, lines)

