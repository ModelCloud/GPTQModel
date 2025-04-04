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

from typing import Optional  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402


def bench(path: str, backend: BACKEND, adapter: Optional[Lora], task):
    # test post-quant inference
    model = GPTQModel.load(
        model_id_or_path=path,
        backend=backend,
        adapter=adapter,
    )

    # torch can benefit from optimization
    if backend == BACKEND.TORCH:
        model.optimize()

    if task == "all":
        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.LM_EVAL.MMLU]
        )
    elif task == "arc":
        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
        )
    elif task == "mmlu":
        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.MMLU]
        )

    del model
    torch_empty_cache()

    return bench_result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--quantized_model', type=str,
        help='Quantized model to load; pass'
    )
    parser.add_argument(
        '--eora_save_path',type=str, default=None
    )
    parser.add_argument(
        '--eora_rank', type=int
    )
    parser.add_argument(
        '--eval_task', type=str, default='all', choices=['mmlu','arc','all']
    )

    args = parser.parse_args()

    if args.eora_save_path:
        eora = Lora(
            # for eora generation, path is adapter save path; for load, it is loading path
            path=args.eora_save_path,
            rank=args.eora_rank,
        )

    if args.eora_save_path:
        eora_bench = bench(path=args.quantized_model, backend=BACKEND.TORCH, adapter=eora, task=args.eval_task)  # inference using eora (lora)
        print('--------Eval EoRA Result---------')
        print(make_table(eora_bench))
        if "groups" in eora_bench:
            print(make_table(eora_bench, "groups"))

    else:
        base_bench = bench(path=args.quantized_model, backend=BACKEND.TORCH, adapter=None, task=args.eval_task)  # inference using qweights only

        print('--------Eval Base Result---------')
        print(make_table(base_bench))
        if "groups" in base_bench:
            print(make_table(base_bench, "groups"))
