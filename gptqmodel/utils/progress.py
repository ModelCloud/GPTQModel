import datetime
import sys
import time


class ProgressBar:
    def __init__(self, data, prefix='', length=40, fill='█', desc=""):
        self.list = []
        if isinstance(data, range):
            self.total = len(data)
        elif isinstance(data, list):
            self.list = data
            self.total = len(data)
        elif isinstance(data, int):
            self.total= data

        self.prefix = prefix
        self.length = length
        self.fill = fill
        self.description = desc
        self.current = 0
        self.time = time.time()

    def set_description(self, description):
        self.description = description

    def progress(self, iteration = None):
        if not iteration:
            iteration = self.current
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        self.log(bar, f"{self.calc_time(iteration)} [{iteration}/{self.total}] {percent}%")

    def calc_time(self, iteration):
        used_time = int(time.time() - self.time)
        formatted_time = str(datetime.timedelta(seconds=used_time))
        remaining = str(datetime.timedelta(seconds=int((used_time / iteration) * self.total)))
        return f"{formatted_time} / {remaining}"

    def log(self, bar, log):
        print(f'\r{self.prefix} {self.description} |{bar}| {log}', end='', flush=True)

    # fix TypeError: 'ProgressBar' object does not support the context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log(f"{'-' * self.length}", "100.0%")

    def __iter__(self):
        return self

    def __next__(self):
        if self.list:
            self.current += 1
            return self.list.pop(0)
        if self.current < self.total - 1:
            self.current += 1
            self.progress(self.current)
            return self.current
        else:
            raise StopIteration


