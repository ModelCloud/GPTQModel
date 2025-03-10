# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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

import json
import os
from enum import Enum
from typing import Optional

from .evalplus import patch_evalplus


class EVAL:
    class LM_EVAL(str, Enum):
        ARC_CHALLENGE = "arc_challenge"
        GSM8K_COT = "gsm8k_cot"
        GSM8K_PLATINUM_COT = "gsm8k_platinum_cot"
        HELLASWAG = "hellaswag"
        MMLU = "mmlu"
        GPQA = "gpqa"

    class EVALPLUS(str, Enum):
        HUMAN = "humaneval"
        MBPP = "mbpp"

    class MMLUPRO(str, Enum):
        BIOLOGY = "biology"
        BUSINESS = "business"
        CHEMISTRY = "chemistry"
        COMPUTER_SCIENCE = "computer science"
        ECONOMICS = "economics"
        ENGINEERING = "engineering"
        HEALTH = "health"
        HISTORY = "history"
        LAW = "law"
        MATH = "math"
        OTHER = "other"
        PHILOSOPHY = "philosophy"
        PHYSICS = "physics"
        PSYCHOLOGY = "psychology"

    @classmethod
    def get_task_enums(cls):
        task_lists = []
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, type) and issubclass(attr, Enum):
                task_lists.extend(list(attr))
        return task_lists

    @classmethod
    def get_full_name(cls, member):
        return f"{cls.__name__}.{member.__class__.__name__}.{member.name}"

    @classmethod
    def get_all_tasks_string(cls):
        full_names = []
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, type) and issubclass(attr, Enum):
                full_names.extend(cls.get_full_name(member) for member in attr)
        return ', '.join(full_names)

def evalplus(
        model,
        dataset: str,
        batch: int = 1,
        trust_remote_code: bool = False,
        output_file: Optional[str] = None,
        backend: str = 'gptqmodel'
):
    patch_evalplus(model)

    try:
        from evalplus.evaluate import evaluate
    except BaseException:
        raise ValueError("evalplus is not installed. Please install via `pip install gptqmodel[evalplus]`.")

    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"

    evaluate(dataset=dataset, model=model, backend=backend, bs=batch, trust_remote_code=trust_remote_code, output_file=output_file,
             greedy=True)

    if output_file is None:
        output_file = model.strip("./").replace("/", "--") + "_gptqmodel_temp_0.0_eval_results.json"
        output_file = os.path.join("evalplus_results", dataset, output_file)

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"No such file: {output_file}")

    try:
        with open(output_file, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON: {output_file}")

    try:
        pass_at_k = data["pass_at_k"]
        base = float(pass_at_k["base"]["pass@1"])
        plus = float(pass_at_k["plus"]["pass@1"])

        base_formatted = format(base, ".3f")
        plus_formatted = format(plus, ".3f")
    except KeyError as e:
        raise ValueError(f"Required key not found in JSON: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Data format error: {str(e)}")

    return base_formatted, plus_formatted, output_file


def evalplus_make_table(results):
    print("|    Tasks    | base tests | base + extra tests |")
    print("|-------------|------------|--------------------|")
    for task, metrics in results.items():
        print(f"| {task} | {metrics['base tests']} | {metrics['base + extra tests']} |")

