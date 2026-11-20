import os
import random
import sys
from pathlib import Path


# OpenMP, MKL and the other BLAS runtimes read their thread-count variables
# when they are loaded, so these assignments have to happen before numpy and
# torch are imported. A container can advertise hundreds of CPUs while sharing
# a much smaller physical host; without the cap every pytest process sizes its
# intra-op pool from that advertised count and then spin-waits in the barrier.
# `setdefault` preserves a caller's explicit values, and an explicit
# OMP_NUM_THREADS also wins as the torch pool size below.
_DEFAULT_TEST_THREADS = min(8, os.cpu_count() or 1)
try:
    _TEST_THREADS = int(os.environ.get("OMP_NUM_THREADS", _DEFAULT_TEST_THREADS))
except ValueError:
    _TEST_THREADS = _DEFAULT_TEST_THREADS
if _TEST_THREADS < 1:
    _TEST_THREADS = _DEFAULT_TEST_THREADS
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, str(_TEST_THREADS))
os.environ.setdefault("OMP_NESTED", "FALSE")

import numpy
import pytest
import torch


# The environment variables are not authoritative for torch's own pool once
# the runtime has loaded, so set the intra-op count explicitly. Inter-op
# threads cannot be changed after the first parallel region, so that call is
# best-effort; the intra-op cap above is always applied.
torch.set_num_threads(_TEST_THREADS)
try:
    torch.set_num_interop_threads(max(1, min(2, _TEST_THREADS)))
except RuntimeError:
    pass


# Reduce logbar progress noise for pytest runs unless a caller explicitly
# overrides the environment. Keep the library default unchanged in LogBar
# itself; this is only a test harness preference.
#os.environ.setdefault("LOGBAR_ANIMATION", "0")
#os.environ.setdefault("LOGBAR_PROGRESS_OUTPUT_INTERVAL", "10")

# Keep unit tests deterministic across CI runs.
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)


@pytest.fixture(autouse=True)
def _reset_kernel_config_between_tests():
    """Keep the process-wide kernel policy from leaking between tests."""

    from gptqmodel.utils.kernel import reset_kernel_config

    reset_kernel_config()
    try:
        yield
    finally:
        reset_kernel_config()

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


_OPT_IN_MARKERS = (
    "exl3_cuda_e2e",
    "gptq_cuda_e2e",
    "gptq_cuda_eval_only",
    "checkpoint_recovery",
    "meta_coverage",
)


def pytest_collection_modifyitems(config, items):
    """Apply opt-in test selection, then order native-TTNN hardware tests.

    Opt-in checks quantize and evaluate a real model, run the multi-minute
    subprocess checkpoint-recovery matrix, or build every registered model on
    ``meta`` to check routed-expert coverage, so they are selected by marker
    rather than by an environment variable. Any other ``-m`` expression still
    leaves them skipped; only naming the marker runs them.

    TT-Metal allows one runtime owner per chip, and initializing the Torch/XLA
    PJRT plugin is irreversible within a process: once it has claimed the
    devices, a native ``ttnn.MeshDevice`` cannot be opened again.  Both suites
    are individually valid and gated behind explicit environment flags, so the
    only reliable way to run them together is to take the native path first.
    This is an ordering contract, not a skip: whichever backend the user
    enabled still runs.
    """
    markexpr = getattr(config.option, "markexpr", "") or ""
    for item in items:
        for marker_name in _OPT_IN_MARKERS:
            if item.get_closest_marker(marker_name) and marker_name not in markexpr:
                item.add_marker(
                    pytest.mark.skip(reason=f"opt-in test; run with `-m {marker_name}` to enable it")
                )
                break

    if os.environ.get("GPTQMODEL_TEST_TTNN") != "1":
        return

    def _priority(item):
        name = item.nodeid
        if "ttnn" in name and "actual" in name:
            return 0
        if "ttxla" in name or "tt_xla" in name:
            return 2
        return 1

    items.sort(key=_priority)
