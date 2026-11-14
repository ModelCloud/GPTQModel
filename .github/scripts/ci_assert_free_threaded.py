"""Fail a CI job that claims to run free-threaded but silently re-enabled the GIL.

A 3.14t build can re-enable the GIL when an extension that has not declared
free-threading support is imported; ``triton._C.libtriton`` is the known one.
The job then looks free-threaded, every ``PYTHON_GIL=0``-only test skips, and
the concurrency coverage is gone without a red build. This gate imports the
model stack and re-checks ``sys._is_gil_enabled()`` after each import.
"""

import importlib
import sys


# Imported in order. `torch` and `gptqmodel` are mandatory: a job that cannot
# import them is broken anyway, and the gate must not silently pass.
MANDATORY = ("torch", "gptqmodel")
OPTIONAL = ("triton", "gptqmodel.utils.p2p")


def _gil_enabled():
    gil_check = getattr(sys, "_is_gil_enabled", None)
    return None if gil_check is None else gil_check()


def main() -> int:
    print(f"python: {sys.version}")

    enabled = _gil_enabled()
    if enabled is None:
        print("FAIL: interpreter is not a free-threaded build (no sys._is_gil_enabled)")
        return 1
    if enabled:
        print("FAIL: the GIL is enabled before the model stack is imported; run with PYTHON_GIL=0")
        return 1

    for name in MANDATORY:
        importlib.import_module(name)
        if _gil_enabled():
            print(f"FAIL: importing {name} re-enabled the GIL; this lane cannot exercise GIL=0 code")
            return 1
        print(f"ok: GIL still disabled after importing {name}")

    for name in OPTIONAL:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            print(f"note: {name} is not importable here ({exc})")
            continue
        if _gil_enabled():
            print(f"FAIL: importing {name} re-enabled the GIL; this lane cannot exercise GIL=0 code")
            return 1
        print(f"ok: GIL still disabled after importing {name}")

    print("ok: GIL stays disabled after importing the model stack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
