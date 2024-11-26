import datetime
import sys
import time


class ProgressBar:
    def __init__(self, total, prefix='', length=40, fill='█'):
        if isinstance(total, range):
            self.total = len(total)
        else:
            self.total= total
        self.prefix = prefix
        self.length = length
        self.fill = fill
        self.description = ''
        self.current = 0
        self.time = time.time()

    def set_description(self, description):
        self.description = description

    def print_bar(self, iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        self.write_log(bar, f"{self.calc_time(iteration)} [{iteration}/{self.total}] {percent}% Complete")

    def calc_time(self, iteration):
        used_time = int(time.time() - self.time)
        formatted_time = str(datetime.timedelta(seconds=used_time))
        remaining = str(datetime.timedelta(seconds=int((used_time / iteration) * (self.total))))
        return f"{formatted_time} / {remaining}"

    def update(self):
        for i in range(1, self.total + 1):
            self.print_bar(i)
        self.write_log(f"{'-' * self.length}","100.0% Complete")

    def write_log(self,bar, log):
        sys.stdout.write(f'\r{self.prefix} {self.description} |{bar}| {log}\n')
        sys.stdout.flush()

    # fix TypeError: 'ProgressBar' object does not support the context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write_log(f"{'-' * self.length}","100.0% Complete")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.total - 1:
            self.current += 1
            self.print_bar(self.current)
            return self.current
        else:
            raise StopIteration


