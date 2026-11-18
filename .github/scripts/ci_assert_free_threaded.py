"""Verify the CI interpreter and model stack before using the installed package."""

import argparse
import importlib
import sys
import sysconfig
from pathlib import Path


PYTHON_VERSION = (3, 14, 10)
REQUIRED_MODULES = ("torch", "gptqmodel", "triton", "torchao")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--installed", action="store_true", help="Require imports from the active virtual environment")
    args = parser.parse_args()

    print(f"python: {sys.version}")
    if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
        raise SystemExit("FAIL: interpreter is not a free-threaded build")
    if sys._is_gil_enabled():
        raise SystemExit("FAIL: the GIL is enabled before the model stack is imported; run with PYTHON_GIL=0")
    if sys.version_info[:3] != PYTHON_VERSION:
        raise SystemExit(f"FAIL: expected CPython {'.'.join(map(str, PYTHON_VERSION))}t")
    if args.python_only:
        print("ok: the free-threaded interpreter starts with the GIL disabled")
        return 0

    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        if sys._is_gil_enabled():
            raise SystemExit(f"FAIL: importing {name} re-enabled the GIL")
        if args.installed and not Path(module.__file__).is_relative_to(Path(sys.prefix)):
            raise SystemExit(f"FAIL: expected {name} from the active environment, got {module.__file__}")
        print(f"ok: GIL still disabled after importing {name} from {module.__file__}")

    from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4  # noqa: F401
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, nvfp4_quantize  # noqa: F401

    if sys._is_gil_enabled():
        raise SystemExit("FAIL: importing the TorchAO NVFP4 helpers re-enabled the GIL")

    torch = sys.modules["torch"]
    constraint = Path(__file__).resolve().parents[1] / "ci" / "torch-cu132.txt"
    expected = next(line.removeprefix("torch==") for line in constraint.read_text().splitlines() if line.startswith("torch=="))
    if torch.__version__ != expected or torch.version.cuda != "13.2":
        raise SystemExit(f"FAIL: expected {expected} for CUDA 13.2, got {torch.__version__} for CUDA {torch.version.cuda}")
    print(f"ok: GIL stays disabled after importing the model stack using torch {torch.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
