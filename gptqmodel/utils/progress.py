import datetime

import time
from warnings import warn

class ProgressBarWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """
    def __init__(self, msg, fp_write=None, *a, **k):
        if fp_write is not None:
            fp_write("\n" + self.__class__.__name__ + ": " + str(msg).rstrip() + '\n')
        else:
            super().__init__(msg, *a, **k)

class ProgressBar:
    def __init__(self, iterable=None, total=None, prefix='', bar_length=40, fill='█', desc=""):
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        if total == float("inf"):
            # Infinite iterations, behave same as unknown
            total = None

        self.iterable = iterable
        self.total = total

        self.prefix = prefix
        self.bar_length = bar_length
        self.fill = fill
        self.description = desc
        self.current = 0
        self.time = time.time()

    def set_description(self, description):
        self.description = description

    def progress(self, iteration = None):
        if not iteration:
            iteration = self.current
        percent = ("{0:.1f}").format(100 * (iteration / float(len(self))))
        filled_length = int(self.bar_length * iteration // len(self))
        bar = self.fill * filled_length + '-' * (self.bar_length - filled_length)
        self.log(bar, f"{self.calc_time(iteration)} [{iteration}/{len(self)}] {percent}%")

    def calc_time(self, iteration):
        used_time = int(time.time() - self.time)
        formatted_time = str(datetime.timedelta(seconds=used_time))
        remaining = str(datetime.timedelta(seconds=int((used_time / max(iteration, 1)) * len(self))))
        return f"{formatted_time} / {remaining}"

    def log(self, bar, log):
        print(f'\r{self.prefix} {self.description} |{bar}| {log}', end='', flush=True)

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

    def __reversed__(self):
        try:
            orig = self.iterable
        except AttributeError:
            raise TypeError("'tqdm' object is not reversible")
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
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        for obj in iterable:
            yield obj
        return

    def close(self):
        self.log(f"{'-' * self.bar_length}", "100.0%")


