# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD, QUANT_CONFIG_FILENAME  # noqa: E402


PROMPT = "The capital city of France is named"
PRETRAINED_MODEL_ID = "/monster/data/model/Llama-3.2-1B"
# Historical local alternative used during AWQ bring-up:
# "/monster/data/model/Qwen2.5-0.5B-Instruct/"
CALIBRATION_DATASET_PATH = "/monster/data/model/dataset/c4-train.00000-of-01024.json.gz"
AWQ_GROUP_SIZE = 128


_TOKENIZER = None
_CALIBRATION_DATASETS: dict[int, object] = {}


def awq_sample_count() -> int:
    requested_samples = os.getenv("GPTQMODEL_AWQ_CALIB_SAMPLES")
    if requested_samples is not None:
        return max(1, int(requested_samples))

    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        except Exception:
            pass

    # if total_mem_gb >= 80:
    #     sample_count = 1024
    # elif total_mem_gb >= 48:
    #     sample_count = 512
    # else:
    #     sample_count = 192
    return 512


def get_awq_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_ID, use_fast=True)
    return _TOKENIZER


def get_awq_calibration_dataset():
    sample_count = awq_sample_count()
    dataset = _CALIBRATION_DATASETS.get(sample_count)
    if dataset is None:
        traindata = load_dataset("json", data_files=CALIBRATION_DATASET_PATH, split="train")
        dataset = traindata.select(range(sample_count))
        _CALIBRATION_DATASETS[sample_count] = dataset
    return dataset


def save_quantized_awq_artifact(
        checkpoint_format: FORMAT,
        artifact_root: str,
        *,
        group_size: int = AWQ_GROUP_SIZE,
) -> dict:
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=group_size,
        quant_method=METHOD.AWQ,
        format=checkpoint_format,
    )

    model = GPTQModel.load(
        PRETRAINED_MODEL_ID,
        quantize_config=quantize_config,
    )
    model.quantize(get_awq_calibration_dataset(), batch_size=1, calibration_concat_size=0)
    model.save(artifact_root)

    with open(Path(artifact_root) / QUANT_CONFIG_FILENAME, "r", encoding="utf-8") as config_file:
        file_dict = json.loads(config_file.read())
        assert model.quantize_config.to_dict() == file_dict
        # Exclude `offload_to_disk_path`, which is a random value.
        file_dict["meta"].pop("offload_to_disk_path")
        logging.info("Saved config file: %s", file_dict)

    del model
    return file_dict


def assert_awq_linear_backend(model, backend: BACKEND) -> None:
    if backend == BACKEND.GEMM:
        linear_cls = AwqGEMMLinear
    elif backend == BACKEND.MACHETE:
        linear_cls = AwqMacheteLinear
    elif backend == BACKEND.MARLIN:
        linear_cls = AwqMarlinLinear
    elif backend == BACKEND.GEMV:
        linear_cls = AwqGEMVLinear
    elif backend == BACKEND.GEMV_FAST:
        linear_cls = AwqGEMVFastLinear
    else:
        raise ValueError(f"unknown backend: {backend}")

    assert any(isinstance(module, linear_cls) for _, module in model.named_modules())


def assert_loaded_quantize_config_matches(model, expected_config: dict) -> None:
    actual_quantize_config = model.quantize_config.to_dict()
    actual_quantize_config["meta"].pop("offload_to_disk_path")
    assert actual_quantize_config == expected_config


def assert_generation_mentions_paris_or_city(result: str, *, extra_terms: tuple[str, ...] = ()) -> None:
    accepted_terms = {"paris", "city", *extra_terms}
    if not any(term in result.lower() for term in accepted_terms):
        raise AssertionError(f"expected one of {sorted(accepted_terms)} in generation: {result}")


def run_quantized_awq_generation_test(checkpoint_format: FORMAT, backend: BACKEND, *, group_size: int = AWQ_GROUP_SIZE):
    format_name = getattr(checkpoint_format, "value", str(checkpoint_format)).lower()
    with tempfile.TemporaryDirectory(prefix=f"awq_{format_name}_") as artifact_root:
        quantize_config_dict = save_quantized_awq_artifact(
            checkpoint_format,
            artifact_root,
            group_size=group_size,
        )
        model = GPTQModel.load(
            artifact_root,
            backend=backend,
            device="cuda",
        )

        assert_loaded_quantize_config_matches(model, quantize_config_dict)
        assert_awq_linear_backend(model, backend)

        result = ModelTest.generate_stable_with_limit(
            model,
            get_awq_tokenizer(),
            PROMPT,
            max_new_tokens=100,
        )
        print(f"BACKEND: {backend}, Result: {result}")
        assert_generation_mentions_paris_or_city(result)

        del model


def run_inference_only_generation_test(
    model_id: str,
    *,
    backend: BACKEND,
    max_new_tokens: int,
    extra_terms: tuple[str, ...] = (),
) -> None:
    model = GPTQModel.load(
        model_id,
        backend=backend,
        device="cuda",
    )

    result = ModelTest.generate_stable_with_limit(
        model,
        model.tokenizer,
        PROMPT,
        max_new_tokens=max_new_tokens,
    )
    print(f"BACKEND: {backend}, Result: {result}")
    assert_generation_mentions_paris_or_city(result, extra_terms=extra_terms)

    del model
