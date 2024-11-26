import sys


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

    def set_description(self, description):
        self.description = description

    def print_bar(self, iteration):
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        sys.stdout.write(f'\r{self.prefix} {self.description} |{bar}| {percent}% Complete')
        sys.stdout.flush()

    def update(self):
        for i in range(1, self.total + 1):
            self.print_bar(i)
        sys.stdout.write(f'\r{self.prefix} {self.description} |{"-" * self.length}| 100.0% Complete\n')
        sys.stdout.flush()

    # fix TypeError: 'ProgressBar' object does not support the context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write(f'\r{self.prefix} {self.description} |{"-" * self.length}| 100.0% Complete\n')
        sys.stdout.flush()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.total - 1:
            self.current += 1
            self.print_bar(self.current)
            return self.current
        else:
            raise StopIteration


