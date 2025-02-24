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
import time
from enum import Enum
from typing import Iterable, Optional
from warnings import warn

from gptqmodel.utils.logger import setup_logger, update_last_pb_instance
from gptqmodel.utils.terminal import terminal_size, terminal_size

logger = setup_logger()

class ProgressBarWarning(Warning):
    def __init__(self, msg, fp_write=None, *a, **k):
        if fp_write is not None:
            fp_write("\n" + self.__class__.__name__ + ": " + str(msg).rstrip() + '\n')
        else:
            super().__init__(msg, *a, **k)

class RenderMode(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class ProgressBar:
    def __init__(self, iterable: Iterable):
        self._iterating = False # state: in init or active iteration
        self._skip_next_draw = False

        self._render_mode = RenderMode.AUTO

        self._title = ""
        self._subtitle = ""
        self._fill = '█'
        self.closed = False # active state

        # max info length over the life ot the pb
        self.max_title_len = 0
        self.max_subtitle_len = 0

        self.iterable = iterable

        self.bar_length = 0
        self.current_iter_step = 0
        self.time = time.time()

        self.ui_show_left_steps = True # show [1 of 100] on left side
        self.ui_show_left_steps_offset = 0

    def set(self,
            show_left_steps: Optional[bool] = None,
            left_steps_offset: Optional[int] = None,
            ):
        if show_left_steps is not None:
            self.ui_show_left_steps = show_left_steps

        if left_steps_offset is not None:
            self.ui_show_left_steps_offset = left_steps_offset
        return self

    def fill(self, fill = '█'):
        self._fill = fill
        return self

    def title(self, title:str):
        if self._iterating and self._render_mode != RenderMode.MANUAL:
            logger.warn("ProgressBar: Title should not be updated after iteration has started unless in `manual` render mode.")

        if len(title) > self.max_title_len:
            self.max_title_len = len(title)

        self._title = title
        return self

    def subtitle(self, subtitle: str):
        if self._iterating and self._render_mode != RenderMode.MANUAL:
            logger.warn("ProgressBar: Sub-title should not be updated after iteration has started unless in `manual` render mode.")

        if len(subtitle) > self.max_subtitle_len:
            self.max_subtitle_len = len(subtitle)

        self._subtitle = subtitle
        return self

    # set render mode
    def mode(self, mode: RenderMode):
        self._render_mode = mode

    def auto(self):
        self._render_mode = RenderMode.AUTO
        return self

    def manual(self ):
        self._render_mode = RenderMode.MANUAL
        return self

    def skip_next_draw(self):
        if not self._skip_next_draw:
            self._skip_next_draw = True

    def draw(self):
        columns, _ = terminal_size()
        bar_length = columns

        if self._title:
            bar_length -= self.max_title_len

        if self._subtitle:
            bar_length -= self.max_subtitle_len

        if self._title and self.subtitle:
            bar_length -= 1 # space between title and subtitle

        percent_num = self.step() / float(len(self))
        percent = ("{0:.1f}").format(100 * (percent_num))
        log = f"{self.calc_time(self.step())} [{self.step()}/{len(self)}] {percent}%"

        bar_length -= len(log)
        bar_length -= 5 # space + | chars

        if not self._title and not self._subtitle:
            bar_length += 1

        # generate: ui_left_steps
        if self.ui_show_left_steps:
            self.ui_show_left_steps_text = f"[{self.step()-self.ui_show_left_steps_offset} of {len(self)-self.ui_show_left_steps_offset}]"
            self.ui_show_left_steps_text_max_len = len(self.ui_show_left_steps_text)
            bar_length -= self.ui_show_left_steps_text_max_len

        padding = ""

        # calculate padding
        if self._title and len(self._title) < self.max_title_len:
            padding += " " * (self.max_title_len - len(self._title))

        # calculate padding
        if self._subtitle and len(self._subtitle) < self.max_subtitle_len:
            padding += " " * (self.max_subtitle_len - len(self._subtitle))

        bar_length -= len(padding)

        if bar_length < 0:
            bar_length = 0

        filled_length = int(bar_length * self.step() // len(self))
        bar = self._fill * filled_length + '-' * (bar_length - filled_length)
        self.log(bar=bar, log=log, padding=padding, end='') # '\n' if percent_num >= 1.0 else ''

    def calc_time(self, iteration):
        used_time = int(time.time() - self.time)
        formatted_time = str(datetime.timedelta(seconds=used_time))
        remaining = str(datetime.timedelta(seconds=int((used_time / max(iteration, 1)) * len(self))))
        return f"{formatted_time} / {remaining}"

    def log(self, bar:str, log:str, padding:str = "", end: str = ""):
        # print(f'\r{self.prefix} {self.info_text} |{bar}| {log}', end='', flush=True)
        out = ""
        if self._title:
            out += self._title + " "

        if self._subtitle:
            out += self._subtitle + " "

        out += padding

        if self.ui_show_left_steps:
            out += self.ui_show_left_steps_text + " "
        else:
            out += "|"

        out += f"{bar}| {log}"

        print(f'\r{out}', end=end, flush=True)

        # if self._title and self.subtitle:
        #     print(f'\r{self._title} {self._subtitle}{self.ui_show_left_steps_text}{padding} |{bar}| {log}', end=end, flush=True)
        # elif self._title:
        #     print(f'\r{self._title}{self.ui_show_left_steps_text}{padding} |{bar}| {log}', end=end, flush=True)
        # elif self._subtitle:
        #     print(f'\r{self._subtitle}{self.ui_show_left_steps_text}{padding} |{bar}| {log}', end=end, flush=True)
        # else:
        #     print(f'\r{self.ui_show_left_steps_text}|{bar}| {log}', end=end, flush=True)

        update_last_pb_instance(src=self)  # let logger now we logged

    def __bool__(self):
        if self.iterable is None:
            raise TypeError('bool() undefined when iterable == total == None')
        return bool(self.iterable)

    def __len__(self):
        return (
            self.iterable.shape[0] if hasattr(self.iterable, "shape")
            else len(self.iterable) if hasattr(self.iterable, "__len__")
            else self.iterable.__length_hint__() if hasattr(self.iterable, "__length_hint__")
            else getattr(self, "total", None))

    # TODO FIXME: I have no cluse why the try/catch is catching nothing here
    def __reversed__(self):
        try:
            original = self.iterable
        except AttributeError:
            raise TypeError("'progress' object is not reversible")
        else:
            self.iterable = reversed(self.iterable)
            return self.__iter__()
        finally:
            self.iterable = original

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

    def step(self) -> int:
        return self.current_iter_step

    def next(self):
        self.current_iter_step += 1
        return self

    def __iter__(self):
        iterable = self.iterable

        for obj in iterable:
            # updat running state
            if not self._iterating:
                self.iterating = True

            self.next()

            if self._render_mode == RenderMode.AUTO:
                self.draw()

            # if not self._skip_next_draw:
            #     if self._render_mode == RenderMode.AUTO:
            #         self.draw()
            # else:
            #     self._skip_next_draw = False
            yield obj

        # self.progress()
        self.close()
        return

    def close(self):
        self.closed = True
        #self.log(f"{self.fill * self.bar_length}", "100.0%", end="\n")


