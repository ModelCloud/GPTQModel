import json
import os
from enum import Enum
from typing import List, Optional, Union


class EVAL:
    class LM_EVAL(Enum):
        ARC_CHALLENGE = "arc_challenge"
        MMLU = "mmlu"
        HELLASWAG = "hellaswag"
        GSM8K_COT = "gsm8k_cot"

    class EVALPLUS(Enum):
        HUMAN = "humaneval"
        MBPP = "mbpp"

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
        model: str,
        dataset: str,
        batch: int = 1,
        trust_remote_code: bool = False,
        output_file: Optional[str] = None,
):
    try:
        from evalplus.evaluate import evaluate
    except BaseException:
        raise ValueError("evalplus is not installed. Please install via `pip install gptqmodel[evalplus]`.")

    assert dataset in ["humaneval", "mbpp"], f"Invalid dataset {dataset}"

    evaluate(dataset=dataset, model=model, backend="gptqmodel", bs=batch, trust_remote_code=trust_remote_code, output_file=output_file,
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


def lm_eval(
        model,
        model_args: str = "",
        model_name: Optional[str] = "hf",
        tasks: Optional[List[Union[str, dict, object]]] = None,
        num_fewshot: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = 32,
        max_batch_size: Optional[int] = 64,
        use_cache: Optional[str] = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        delete_requests_cache: bool = False,
        limit: Optional[Union[int, float]] = None,
        bootstrap_iters: int = 100000,
        check_integrity: bool = False,
        write_out: bool = False,
        log_samples: bool = True,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_kwargs: Optional[str] = None,
        verbosity: str = "INFO",
        predict_only: bool = False,
        random_seed: int = 0,
        numpy_random_seed: int = 1234,
        torch_random_seed: int = 1234,
        fewshot_random_seed: int = 1234,
        output_path: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        show_config: bool = False,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
):
    try:
        from lm_eval import simple_evaluate
        from lm_eval.loggers import EvaluationTracker, WandbLogger
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import handle_non_serializable
        from transformers import PreTrainedModel as PreTrainedModelType
    except BaseException:
        raise ValueError("lm_eval is not installed. Please install via `pip install gptqmodel[eval]`.")

    if model_name == "hf" and isinstance(model, PreTrainedModelType):
        model_name = HFLM(
            pretrained=model,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            trust_remote_code=trust_remote_code,
        )
    evaluation_tracker = None
    if output_path is not None:
        evaluation_tracker = EvaluationTracker(output_path=output_path)
    results = simple_evaluate(
        model=model_name,
        model_args=model_args,
        tasks=tasks,
        device=device,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        use_cache=use_cache,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        delete_requests_cache=delete_requests_cache,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        check_integrity=check_integrity,
        write_out=write_out,
        log_samples=log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        verbosity=verbosity,
        predict_only=predict_only,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
    )

    if results is not None:
        if log_samples:
            samples = results.pop("samples")

        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if show_config:
            print(dumped)

        # Add W&B logging
        if wandb_project is not None:
            wandb_logger = WandbLogger(
                project=wandb_project, job_type="eval", name=wandb_name
            )
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            if log_samples:
                wandb_logger.log_eval_samples(samples=samples)

        if evaluation_tracker is not None:
            evaluation_tracker.save_results_aggregated(
                results=results, samples=samples if log_samples else None
            )

            if log_samples:
                for task_name, config in results["configs"].items():
                    evaluation_tracker.save_results_samples(
                        task_name=task_name, samples=samples[task_name]
                    )

            if (evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub):
                evaluation_tracker.recreate_metadata_card()

        return results
    else:
        raise ValueError('lm_eval run fail, check your code!!!')



