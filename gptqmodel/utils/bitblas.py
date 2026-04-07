# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from contextlib import nullcontext

import torch

from ..nn_modules.qlinear.bitblas import BitBLASLinear
from ..nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from ..quantization import FORMAT, METHOD, QuantizeConfig
from ..quantization.config import resolve_quant_format
from ..utils.logger import setup_logger
from .model import load_checkpoint_in_model_then_tie_weights
from .safe import THREADPOOLCTL
from .torch import torch_empty_cache


log = setup_logger()


def _select_bitblas_kernel_class(qcfg: QuantizeConfig):
    if qcfg.quant_method == METHOD.AWQ:
        return AWQBitBlasKernel
    return BitBLASLinear


def _should_enable_bitblas_tuning(repack: bool) -> bool:
    """Keep GPTQ repacks responsive unless tuning is explicitly requested."""
    raw = os.getenv("BITBLAS_ENABLE_TUNING")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return not repack


def prepare_model_for_bitblas_load(
        model,
        qcfg: QuantizeConfig,
        quant_linear_class,
        dtype,
        model_save_name,
        device_map,
        sym: bool,
        desc_act: bool,
        load_checkpoint_in_model: bool,
):
    # The model (e.g. model.safetensors) is already serialized in the BitBLAS format, load it directly.
    if resolve_quant_format(qcfg.format, qcfg.method) == FORMAT.BITBLAS:
        # if the checkpoint is already in bitblas format, we can load it directly.
        log.info(f"Loading a {qcfg.quant_method.upper()} model, detected BitBLAS serialized format at {model_save_name}.")
        model = convert_to_bitblas(model, quant_linear_class, qcfg, sym, desc_act, repack=False, dtype=dtype)
        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=dtype,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )
    else:
        # Loading the GPTQ checkpoint to do the conversion.
        # TODO: Avoid loading the model with wrong QuantLinear, and directly use
        # BitBLAS ones. The repacking can be done directly on the safetensors, just
        # as for AWQ checkpoints.
        if load_checkpoint_in_model:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )
        # Convert model to bitblas, repacking weights into BitBLAS format.
        model = convert_to_bitblas(model, quant_linear_class, qcfg, sym, desc_act, repack=True, dtype=dtype)
    return model


@torch.inference_mode()
def convert_to_bitblas(
    model,
    model_quantlinear,
    qcfg: QuantizeConfig,
    sym: bool,
    desc_act: bool,
    repack: bool,
    dtype: torch.dtype = torch.float16,
):
    """
    Converts GPTQ-packed weights to the Bitblas format.

    Arguments:
        repack (`bool`):
            Whether to repack the qweights from `model` into the BitBLAS's QuantLinear layers.
    """
    if repack:
        message = "Repacking weights to be compatible with BitBLAS kernel..."
    else:
        # TODO: load directly BitBLAS QuantLinear.
        message = "Overriding QuantLinear layers to use BitBLAS's QuantLinear..."

    bitblas_quantlinear = _select_bitblas_kernel_class(qcfg)

    # TODO: need to benchmark to see multiple threads help with bitblas/tvm compilation and runtime
    threadpool_limits = (
        THREADPOOLCTL.threadpool_limits
        if THREADPOOLCTL is not None
        else (lambda *args, **kwargs: nullcontext())
    )

    enable_tuning = _should_enable_bitblas_tuning(repack)

    with threadpool_limits(limits=1):
        os.environ["NUMEXPR_MAX_THREADS"] = "1"

        # Note that due to tvm compilation of per layer modules shapes, the first layer loop is
        # relatively much slower if caching is not available. estimate time remaining is highly inaccurate
        for name, module in log.pb(list(model.named_modules())).title(message):
            if not isinstance(module, model_quantlinear):
                continue

            parent_name, _, layer_name = name.rpartition(".")

            # We could use `torch.count_nonzero(module.bias) > 0` here to discard zero bias, but this has issues when loading weights
            # from checkpoints holding zero bias.
            with torch.device("meta"):
                bitblas_module = bitblas_quantlinear(
                    bits=qcfg.runtime_bits,
                    group_size=qcfg.group_size,
                    sym=sym,
                    desc_act=desc_act,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    pack_dtype=qcfg.pack_dtype,
                    bias=module.bias is not None,
                    dtype=dtype,
                    enable_tuning=enable_tuning,
                    adapter=qcfg.adapter,
                    name=name,
                )

            # convert to bitblas format
            if repack:
                if qcfg.quant_method == METHOD.AWQ:
                    bitblas_module.repack_from_awq(module)
                else:
                    bitblas_module.repack_from_gptq(module)

            # Save to parent.
            parent_module = model if parent_name == "" else model.get_submodule(parent_name)
            setattr(parent_module, layer_name, bitblas_module)

            # Free cuda memory.
            del module
            torch_empty_cache()

    # Set quantization config to be BitBLAS.
    qcfg.runtime_format = FORMAT.BITBLAS

    return model
