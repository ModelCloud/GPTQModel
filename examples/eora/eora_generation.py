# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
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

# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# -- end do not touch


# from models.model_test import ModelTest  # noqa: E402
from eora_calibration_data_construction import construct_c4, construct_mmlu
from gptqmodel import GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import Lora
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402

## meta-llama/Llama-3.2-1B
## meta-llama/Llama-3.2-3B
## meta-llama/Meta-Llama-3-8B
## meta-llama/Llama-3.1-8B
## meta-llama/Meta-Llama-3-70B

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Full-preicision model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--bits', type=int, default=4
    )
    parser.add_argument(
        '--group_size', type=int, default=128
    )
    parser.add_argument(
        '--quant_save_path', type=str, default=None
    )
    parser.add_argument(
        '--eora_dataset', type=str, choices=['c4','mmlu'],
        help='calibration set for eora'
    )
    parser.add_argument(
        '--eora_save_path',type=str, default=None
    )
    parser.add_argument(
        '--eora_rank', type=int, default=64
    )

    args = parser.parse_args()

    NATIVE_MODEL_ID = args.model

    bits = args.bits
    group_size = args.group_size
    desc_act = True
    rank = args.eora_rank
    batch_size = 1
    calibration_dataset_concat_size = 0  # disable
    auto_gc = False

    if args.quant_save_path is not None:
        quant_path = args.quant_save_path
    else:
        raise AssertionError('Please provide a save path for the quantized model')

    if args.eora_save_path is not None:
        eora_path = args.eora_save_path
    else:
        raise AssertionError('Please provide a save path for EoRA')



    ## C4 for quant
    calibration_dataset = construct_c4()

    eora = Lora(
        # for quant, path is save path. for load, it is loading path
        path=os.path.join(quant_path, eora_path),
        rank=rank,
    )

    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,  # bitblas only supports DESC_ACT=False
        adapter=eora
    )

    model = GPTQModel.load(
        model_id_or_path=NATIVE_MODEL_ID,
        quantize_config=quant_config,
    )

    if args.eora_dataset == "c4":
        model.quantize(
            calibration_dataset=calibration_dataset,
            batch_size=batch_size,
            auto_gc=auto_gc,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
        )  #
    else:

        eora_calibration_dataset = construct_mmlu()

        model.quantize(
            calibration_dataset=calibration_dataset,
            batch_size=batch_size,
            auto_gc=auto_gc,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            adapter_calibration_dataset=eora_calibration_dataset
        )  #


    # EoRA adapter is saved according to Lora.path property
    # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
    # You can also pass `eora_path` to `model.save()` to override this save path
    model.save(quant_path)

    del model
    torch_empty_cache()


