# ruff: noqa: I001
import os
import random
import sys
from pathlib import Path


_MAX_PYTEST_CPU_THREADS = 16
_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pytest_cpu_thread_limit() -> int:
    requested = []
    for name in _NATIVE_THREAD_ENV_VARS:
        raw_value = os.environ.get(name)
        if raw_value is None:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value > 0:
            requested.append(value)
    return min(_MAX_PYTEST_CPU_THREADS, *requested) if requested else _MAX_PYTEST_CPU_THREADS


# Apply the environment limit before importing NumPy or Torch so native runtimes
# observe it during initialization. Use one effective limit to prevent a lower
# explicit request for one runtime from being defeated by another runtime.
_PYTEST_CPU_THREAD_LIMIT = _pytest_cpu_thread_limit()
for _thread_env_name in _NATIVE_THREAD_ENV_VARS:
    os.environ[_thread_env_name] = str(_PYTEST_CPU_THREAD_LIMIT)

import numpy
import pytest
import torch
from threadpoolctl import threadpool_limits


_PYTEST_THREADPOOL_LIMITER = threadpool_limits(limits=_PYTEST_CPU_THREAD_LIMIT)
_PYTEST_THREADPOOL_LIMITER.__enter__()
torch.set_num_threads(_PYTEST_CPU_THREAD_LIMIT)


def pytest_addoption(parser):
    parser.addoption(
        "--devin-api-token",
        default=os.environ.get("DEVIN_API_TOKEN"),
        help="Devin API token for GPU allocator integration tests",
    )
    parser.addoption(
        "--devin-org-id",
        default=os.environ.get("DEVIN_ORG_ID"),
        help="Devin organization id for GPU allocator integration tests",
    )
    parser.addoption(
        "--devin-api-base",
        default=os.environ.get("DEVIN_API_BASE", "https://api.devin.ai"),
        help="Devin API base URL (default: https://api.devin.ai)",
    )


def pytest_unconfigure(config):
    del config
    _PYTEST_THREADPOOL_LIMITER.__exit__(None, None, None)


# Reduce logbar progress noise for pytest runs unless a caller explicitly
# overrides the environment. Keep the library default unchanged in LogBar
# itself; this is only a test harness preference.
# os.environ.setdefault("LOGBAR_ANIMATION", "0")
# os.environ.setdefault("LOGBAR_PROGRESS_OUTPUT_INTERVAL", "10")

# Keep unit tests deterministic across CI runs.
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)

_TESTS_DIR = Path(__file__).resolve().parent
_MODELS_TESTS_DIR = _TESTS_DIR / "models"
_REPO_ROOT = _TESTS_DIR.parent

# The suite mixes two helper import styles:
# - `from models.model_test import ModelTest`
# - `from ovis.image_to_test_dataset import ...`
# Add both helper directories, plus the repo root for `tests.*` imports.
for path in (_REPO_ROOT, _TESTS_DIR, _MODELS_TESTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(scope="session")
def devin_api_token(pytestconfig):
    token = pytestconfig.getoption("--devin-api-token")
    if not token:
        pytest.skip("--devin-api-token or DEVIN_API_TOKEN required")
    return token


@pytest.fixture(scope="session")
def devin_org_id(pytestconfig):
    org_id = pytestconfig.getoption("--devin-org-id")
    if not org_id:
        pytest.skip("--devin-org-id or DEVIN_ORG_ID required")
    return org_id


@pytest.fixture(scope="session")
def devin_api_base(pytestconfig):
    return pytestconfig.getoption("--devin-api-base")
