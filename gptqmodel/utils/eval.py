# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
from enum import Enum


try:
    from enum import EnumType
except ImportError:
    EnumType = type(Enum)
from typing import Dict, List, Optional, Type, Union

from .evalplus import patch_evalplus


class EVAL:
    class LM_EVAL(str, Enum):
        ARC_CHALLENGE = "arc_challenge"
        GSM8K_COT = "gsm8k_cot"
        GSM8K_PLATINUM_COT = "gsm8k_platinum_cot"
        HELLASWAG = "hellaswag"
        MMLU = "mmlu"
        GPQA = "gpqa"
        ARC_EASY = "arc_easy"
        BOOLQ = "boolq"
        OPENBOOKQA = "openbookqa"

    class EVALPLUS(str, Enum):
        HUMAN = "humaneval"
        MBPP = "mbpp"

    class MMLU_PRO(str, Enum):
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
    def get_tasks_for_framework(cls, framework: Union[str, Type[Enum]]) -> list:
        if isinstance(framework, EnumType):
            framework = framework.__name__

        if not hasattr(cls, framework):
            raise ValueError(f"No such EVAL framework: `{framework}`")

        enum_class = getattr(cls, framework)
        return list(enum_class)

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

    @classmethod
    def get_task_groups_from_tasks(cls, tasks: Union[str, List[str]]) -> Dict[Type[Enum], List[str]]:
        """Group tasks by their evaluation framework.

        Args:
            tasks: Either a single task name or list of task names

        Returns:
            Dictionary mapping framework enum classes to lists of tasks
            Example: {EVAL.LM_EVAL: ["arc_challenge", "hellaswag"], EVAL.EVALPLUS: ["humaneval"]}

        Raises:
            ValueError: If any task doesn't match a known framework
        """
        if isinstance(tasks, str):
            tasks = [tasks]

        # Create a mapping of task values to their enum classes
        task_to_framework = {}

        # Populate the mapping for all frameworks
        for framework in [cls.LM_EVAL, cls.EVALPLUS, cls.MMLU_PRO]:
            for task in framework:
                task_to_framework[task.value] = framework

        # Group tasks by their framework
        task_groups = {}
        unknown_tasks = []

        for task in tasks:
            if task in task_to_framework:
                framework = task_to_framework[task]
                if framework not in task_groups:
                    task_groups[framework] = []
                task_groups[framework].append(task)
            else:
                unknown_tasks.append(task)

        if unknown_tasks:
            raise ValueError(f"Unknown tasks: {unknown_tasks}")

        return task_groups


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

