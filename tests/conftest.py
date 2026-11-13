import os
import random
import sys
from pathlib import Path

import numpy
import torch


# Reduce logbar progress noise for pytest runs unless a caller explicitly
# overrides the environment. Keep the library default unchanged in LogBar
# itself; this is only a test harness preference.
#os.environ.setdefault("LOGBAR_ANIMATION", "0")
#os.environ.setdefault("LOGBAR_PROGRESS_OUTPUT_INTERVAL", "10")

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


def pytest_collection_modifyitems(config, items):
    """Run native-TTNN hardware tests before the Torch/XLA hardware tests.

    TT-Metal allows one runtime owner per chip, and initializing the Torch/XLA
    PJRT plugin is irreversible within a process: once it has claimed the
    devices, a native ``ttnn.MeshDevice`` cannot be opened again.  Both suites
    are individually valid and gated behind explicit environment flags, so the
    only reliable way to run them together is to take the native path first.
    This is an ordering contract, not a skip: whichever backend the user
    enabled still runs.
    """

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
