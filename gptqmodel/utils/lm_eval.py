import json
from typing import List, Optional, Union

try:
    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker, WandbLogger
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable
except:
    raise ValueError("lm_eval is not installed. Please install via `pip install gptqmodel[eval]`.")


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
        evaluation_tracker: Optional[EvaluationTracker] = None,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        gen_kwargs: Optional[str] = None,
        task_manager: Optional[TaskManager] = None,
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
):
    if model_name == "hf":
        model_name = HFLM(
            pretrained=model,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            trust_remote_code=trust_remote_code,
        )
    # evaluation_tracker need model_args cannot be None
    if evaluation_tracker is None and output_path is not None:
        evaluation_tracker = EvaluationTracker(output_path=output_path)
    results = simple_evaluate(
        model=model_name,
        model_args=model_args,
        tasks=tasks,
        device=str(model.device),
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
        task_manager=task_manager,
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