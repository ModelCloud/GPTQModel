from __future__ import annotations
import logging
from typing import Dict
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from ..nn_modules.qlinear.qlinear_qbits import QBitsQuantLinear, qbits_dtype
from ..utils.device import check_cuda
from ..utils.model import (auto_dtype_from_config)
from ._const import SUPPORTED_MODELS


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.propagate = False
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelLoader():
    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    model_loader = AutoModelForCausalLM

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            trust_remote_code: bool = False,
            use_liger_kernel: bool = False,
            torch_dtype: [str | torch.dtype] = "auto",
            require_trust_remote_code=None,
            **model_init_kwargs,
    ):
        """load un-quantized pretrained model to cpu"""
        got_cuda = check_cuda(raise_exception=False)

        if not got_cuda:
            try:
                pass
            except Exception as e:
                raise ValueError(
                    f"QBits is not available: {e}. Please install with `pip install -U intel-extension-for-transformers`."
                )

            model_init_kwargs["device"] = "cpu"
            torch_dtype = qbits_dtype()

        if require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_name_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        # allow models to define optional notes that output messages to users that want to use this model
        notes = cls.info.get("notes")
        if notes:
            logger.info(notes)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        model_init_kwargs["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        if use_liger_kernel:
            from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

            apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(config.model_type, None)
            if apply_fn is None:
                raise ValueError(f"apply_fn is not defined for model type {config.model_type}")

            apply_fn()

        if torch_dtype == "auto":
            torch_dtype = auto_dtype_from_config(config)
        elif not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"torch_dtype value of `{torch_dtype}` is not a torch.dtype instance.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        if model_init_kwargs.get("cpu") != "cpu":
            torch.cuda.empty_cache()

        model = cls.model_loader.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return model

