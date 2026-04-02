from __future__ import annotations

import os

from packaging import version

TRUTHFUL = {"1", "true", "yes", "on", "y"}

MINIMUM_BITBLAS_VERSION = "0.1.0.post1"


def env_flag(name: str, default: str | bool | None = "0") -> bool:
    """Return ``True`` when an env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        if default is None:
            return False
        if isinstance(default, bool):
            return default
        value = default
    return str(value).strip().lower() in TRUTHFUL


__all__ = ["env_flag"]


def _is_bitblas_available() -> bool:
    # Allow disabling BitBLAS probing in environments where TVM import is unstable.
    if env_flag("GPTQMODEL_DISABLE_BITBLAS", default="0"):
        return False

    try:
        import bitblas
    except Exception as exc:
        error_text = str(exc)
        if "libcu" not in error_text:
            print("BitBLAS import failed: %s", exc)
            return False
        # if not _load_cuda_libraries():
        #     print("CUDA libraries missing, BitBLAS import failed: %s", exc)
        #     return False
        try:
            import bitblas
        except Exception as retry_exc:
            print("BitBLAS import retry failed: %s", retry_exc)
            return False
    parsed_version = version.parse(bitblas.__version__)
    minimum_version = version.parse(MINIMUM_BITBLAS_VERSION)
    if parsed_version < minimum_version:
        print(
            "BitBLAS version %s below minimum required %s",
            bitblas.__version__,
            MINIMUM_BITBLAS_VERSION,
        )
        return False
    return True


if __name__ == '__main__':
    print(_is_bitblas_available())
