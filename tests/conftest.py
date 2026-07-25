import os
import random
import sys
from pathlib import Path

import numpy
import pytest
import torch


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
