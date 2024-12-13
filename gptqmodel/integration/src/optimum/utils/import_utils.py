# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
import unittest
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from optimum.utils import (is_accelerate_available, is_auto_gptq_available, is_diffusers_available,
                           is_sentence_transformers_available, is_timm_available)

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import utilities."""

import importlib.util
import inspect
import sys
from collections import OrderedDict
from contextlib import contextmanager
from typing import Tuple, Union

import numpy as np
from packaging import version
from transformers.utils import is_torch_available


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


TORCH_MINIMUM_VERSION = version.parse("1.11.0")
TRANSFORMERS_MINIMUM_VERSION = version.parse("4.25.0")
DIFFUSERS_MINIMUM_VERSION = version.parse("0.22.0")
AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0
GPTQMODEL_MINIMUM_VERSION = version.parse("1.3.99") # Allows 1.4.0.dev0


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = version.parse("1.4.0")


_onnx_available = _is_package_available("onnx")

# importlib.metadata.version seem to not be robust with the ONNX Runtime extensions (`onnxruntime-gpu`, etc.)
_onnxruntime_available = importlib.util.find_spec("onnxruntime") is not None

_pydantic_available = _is_package_available("pydantic")
_accelerate_available = _is_package_available("accelerate")
_diffusers_available = _is_package_available("diffusers")
_auto_gptq_available = _is_package_available("auto_gptq")
_gptqmodel_available = _is_package_available("gptqmodel")
_timm_available = _is_package_available("timm")
_sentence_transformers_available = _is_package_available("sentence_transformers")
_datasets_available = _is_package_available("datasets")

torch_version = None
if is_torch_available():
    torch_version = version.parse(importlib_metadata.version("torch"))

_is_torch_onnx_support_available = is_torch_available() and (
    TORCH_MINIMUM_VERSION.major,
    TORCH_MINIMUM_VERSION.minor,
) <= (
                                       torch_version.major,
                                       torch_version.minor,
                                   )


_diffusers_version = None
if _diffusers_available:
    try:
        _diffusers_version = importlib_metadata.version("diffusers")
    except importlib_metadata.PackageNotFoundError:
        _diffusers_available = False


def is_torch_onnx_support_available():
    return _is_torch_onnx_support_available


def is_onnx_available():
    return _onnx_available


def is_onnxruntime_available():
    try:
        # Try to import the source file of onnxruntime - if you run the tests from `tests` the function gets
        # confused since there a folder named `onnxruntime` in `tests`. Therefore, `_onnxruntime_available`
        # will be set to `True` even if not installed.
        mod = importlib.import_module("onnxruntime")
        inspect.getsourcefile(mod)
    except Exception:
        return False
    return _onnxruntime_available


def is_pydantic_available():
    return _pydantic_available


def is_accelerate_available():
    return _accelerate_available


def is_diffusers_available():
    return _diffusers_available


def is_timm_available():
    return _timm_available


def is_sentence_transformers_available():
    return _sentence_transformers_available


def is_datasets_available():
    return _datasets_available


def is_auto_gptq_available():
    if _auto_gptq_available:
        v = version.parse(importlib_metadata.version("auto_gptq"))
        if v >= AUTOGPTQ_MINIMUM_VERSION:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {v}, but only version >= {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


def is_gptqmodel_available():
    if _gptqmodel_available:
        v = version.parse(importlib_metadata.version("gptqmodel"))
        if v >= GPTQMODEL_MINIMUM_VERSION:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of gptqmodel. Found version {v}, but only version >= {GPTQMODEL_MINIMUM_VERSION} are supported"
            )


@contextmanager
def check_if_pytorch_greater(target_version: str, message: str):
    r"""
    A context manager that does nothing except checking if the PyTorch version is greater than `pt_version`
    """
    import torch

    if not version.parse(torch.__version__) >= version.parse(target_version):
        raise ImportError(
            f"Found an incompatible version of PyTorch. Found version {torch.__version__}, but only {target_version} and above are supported. {message}"
        )
    try:
        yield
    finally:
        pass


def check_if_transformers_greater(target_version: Union[str, version.Version]) -> bool:
    """
    Checks whether the current install of transformers is greater than or equal to the target version.

    Args:
        target_version (`Union[str, packaging.version.Version]`): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    import transformers

    if isinstance(target_version, str):
        target_version = version.parse(target_version)

    return version.parse(transformers.__version__) >= target_version


def check_if_diffusers_greater(target_version: str) -> bool:
    """
    Checks whether the current install of diffusers is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if not _diffusers_available:
        return False

    return version.parse(_diffusers_version) >= version.parse(target_version)


def check_if_torch_greater(target_version: str) -> bool:
    """
    Checks whether the current install of torch is greater than or equal to the target version.

    Args:
        target_version (str): version used as the reference for comparison.

    Returns:
        bool: whether the check is True or not.
    """
    if not is_torch_available():
        return False

    return torch_version >= version.parse(target_version)


@contextmanager
def require_numpy_strictly_lower(package_version: str, message: str):
    if not version.parse(np.__version__) < version.parse(package_version):
        raise ImportError(
            f"Found an incompatible version of numpy. Found version {np.__version__}, but expected numpy<{version}. {message}"
        )
    try:
        yield
    finally:
        pass


DIFFUSERS_IMPORT_ERROR = """
{0} requires the diffusers library but it was not found in your environment. You can install it with pip: `pip install
diffusers`. Please note that you may need to restart your runtime after installation.
"""

TRANSFORMERS_IMPORT_ERROR = """requires the transformers>={0} library but it was not found in your environment. You can install it with pip: `pip install
-U transformers`. Please note that you may need to restart your runtime after installation.
"""

DATASETS_IMPORT_ERROR = """
{0} requires the datasets library but it was not found in your environment. You can install it with pip:
`pip install datasets`. Please note that you may need to restart your runtime after installation.
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("diffusers", (is_diffusers_available, DIFFUSERS_IMPORT_ERROR)),
        (
            "transformers_431",
            (lambda: check_if_transformers_greater("4.31"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.31")),
        ),
        (
            "transformers_432",
            (lambda: check_if_transformers_greater("4.32"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.32")),
        ),
        (
            "transformers_434",
            (lambda: check_if_transformers_greater("4.34"), "{0} " + TRANSFORMERS_IMPORT_ERROR.format("4.34")),
        ),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


# Copied from: https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/utils/import_utils.py#L1041
class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)
# Used to test the hub
USER = "__DUMMY_OPTIMUM_USER__"


def flatten_dict(dictionary: Dict):
    """
    Flatten a nested dictionaries as a flat dictionary.
    """
    items = []
    for k, v in dictionary.items():
        new_key = k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v).items())
        else:
            items.append((new_key, v))
    return dict(items)


def require_accelerate(test_case):
    """
    Decorator marking a test that requires accelerate. These tests are skipped when accelerate isn't installed.
    """
    return unittest.skipUnless(is_accelerate_available(), "test requires accelerate")(test_case)


def require_gptq(test_case):
    """
    Decorator marking a test that requires gptqmodel or auto-gptq. These tests are skipped when gptqmodel and auto-gptq are not installed.
    """
    return unittest.skipUnless(is_auto_gptq_available() or is_gptqmodel_available(), "test requires auto-gptq")(
        test_case
    )


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)


def require_ort_rocm(test_case):
    """Decorator marking a test that requires ROCMExecutionProvider for ONNX Runtime."""
    import onnxruntime as ort

    providers = ort.get_available_providers()

    return unittest.skipUnless("ROCMExecutionProvider" == providers[0], "test requires ROCMExecutionProvider")(
        test_case
    )


def require_hf_token(test_case):
    """
    Decorator marking a test that requires huggingface hub token.
    """
    # is HF_AUTH_TOKEN used instead of HF_TOKEN to avoid huggigface_hub picking it up ?
    hf_token = os.environ.get("HF_AUTH_TOKEN", None)
    if hf_token is None:
        return unittest.skip("test requires hf token as `HF_AUTH_TOKEN` environment variable")(test_case)
    else:
        return test_case


def require_sigopt_token_and_project(test_case):
    """
    Decorator marking a test that requires sigopt API token.
    """
    sigopt_api_token = os.environ.get("SIGOPT_API_TOKEN", None)
    has_sigopt_project = os.environ.get("SIGOPT_PROJECT", None)
    if sigopt_api_token is None or has_sigopt_project is None:
        return unittest.skip("test requires an environment variable `SIGOPT_API_TOKEN` and `SIGOPT_PROJECT`")(
            test_case
        )
    else:
        return test_case


def is_ort_training_available():
    is_ort_train_available = importlib.util.find_spec("onnxruntime.training") is not None

    if importlib.util.find_spec("torch_ort") is not None:
        try:
            is_torch_ort_configured = True
            subprocess.run([sys.executable, "-m", "torch_ort.configure"], shell=False, check=True)
        except subprocess.CalledProcessError:
            is_torch_ort_configured = False

    return is_ort_train_available and is_torch_ort_configured


def require_ort_training(test_case):
    """
    Decorator marking a test that requires onnxruntime-training and torch_ort correctly installed and configured.
    These tests are skipped otherwise.
    """
    return unittest.skipUnless(
        is_ort_training_available(),
        "test requires torch_ort correctly installed and configured",
    )(test_case)


def require_diffusers(test_case):
    return unittest.skipUnless(is_diffusers_available(), "test requires diffusers")(test_case)


def require_timm(test_case):
    return unittest.skipUnless(is_timm_available(), "test requires timm")(test_case)


def require_sentence_transformers(test_case):
    return unittest.skipUnless(is_sentence_transformers_available(), "test requires sentence-transformers")(test_case)


def require_datasets(test_case):
    return unittest.skipUnless(is_datasets_available(), "test requires datasets")(test_case)


def grid_parameters(
        parameters: Dict[str, Iterable[Any]],
        yield_dict: bool = False,
        add_test_name: bool = True,
        filter_params_func: Optional[Callable[[Tuple], Tuple]] = None,
) -> Iterable:
    """
    Generates an iterable over the grid of all combinations of parameters.

    Args:
        `parameters` (`Dict[str, Iterable[Any]]`):
            Dictionary of multiple values to generate a grid from.
        `yield_dict` (`bool`, defaults to `False`):
            If True, a dictionary with all keys, and sampled values will be returned. Otherwise, return sampled values as a list.
        `add_test_name` (`bool`, defaults to `True`):
            Whether to add the test name in the yielded list or dictionary.
        filter_params_func (`Optional[Callable[[Tuple], Tuple]]`, defaults to `None`):
            A function that can modify or exclude the current set of parameters. The function should take a tuple of the
            parameters and return the same. If a parameter set is to be excluded, the function should return an empty tuple.
    """
    for params in itertools.product(*parameters.values()):
        if filter_params_func is not None:
            params = filter_params_func(list(params))
            if params is None:
                continue

        test_name = "_".join([str(param) for param in params])
        if yield_dict is True:
            res_dict = {}
            for i, key in enumerate(parameters.keys()):
                res_dict[key] = params[i]
            if add_test_name is True:
                res_dict["test_name"] = test_name
            yield res_dict
        else:
            returned_list = [test_name] + list(params) if add_test_name is True else list(params)
            yield returned_list


def remove_directory(dirpath):
    """
    Remove a directory and its content.
    This is a cross-platform solution to remove a directory and its content that avoids the use of `shutil.rmtree` on Windows.
    Reference: https://github.com/python/cpython/issues/107408
    """
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        if os.name == "nt":
            os.system(f"rmdir /S /Q {dirpath}")
        else:
            shutil.rmtree(dirpath)