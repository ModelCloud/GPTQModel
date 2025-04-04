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


from eora_calibration_data_construction import construct_ARC, construct_c4, construct_mmlu
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='Full-preicision model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['c4','arc','mmlu'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--quantized_model',type=str,
        help='Quantized model to load'
    )
    parser.add_argument(
        '--eora_save_path',type=str
    )
    parser.add_argument(
        '--eora_rank', type=int
    )

    args = parser.parse_args()

    eora = Lora(
        # for eora generation, path is adapter save path; for load, it is loading path
        path=os.path.join(args.eora_save_path),
        rank=args.eora_rank,
    )

    if args.dataset == "c4":
        calibration_dataset = construct_c4()
    elif args.dataset == "arc":
        calibration_dataset = construct_ARC()
    elif args.dataset == "mmlu":
        calibration_dataset = construct_mmlu()
    else:
        raise NotImplementedError


    # eora generation and save in one step
    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=args.model,
        quantized_model_id_or_path=args.quantized_model,
        calibration_dataset=calibration_dataset,
        calibration_dataset_concat_size=0,
        auto_gc=False)

