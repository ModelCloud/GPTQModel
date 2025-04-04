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

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--quantized_model', type=str,
        help='Quantized model to load; pass'
    )
    parser.add_argument(
        '--eora',type=str, default=None
    )
    parser.add_argument(
        '--eora_rank', type=int
    )


    args = parser.parse_args()

    if args.eora:
        eora = Lora(
            # for eora generation, path is adapter save path; for load, it is loading path
            path=args.eora,
            rank=args.eora_rank,
        )
    else:
        raise AssertionError("Please provide EoRA weight")


    model = GPTQModel.load(
        model_id_or_path=args.quantized_model,
        backend=BACKEND.TORCH,
        adapter=eora,
    )

    tokens = model.generate("Capital of France is")[0]
    result = model.tokenizer.decode(tokens)
    print(f"Result: {result}")


