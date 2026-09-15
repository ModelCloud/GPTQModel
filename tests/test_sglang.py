# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.sglang import (
    SGLANG_AVAILABLE,
    SGLANG_INSTALL_HINT,
    sglang_generate,
)
from gptqmodel.utils.torch import torch_empty_cache


pytestmark = [pytest.mark.model, pytest.mark.slow]

_DEFAULT_MODEL = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
_DEFAULT_SHARDED_MODEL = (
    "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-sharded"
)
_DEFAULT_AWQ_MODEL = "/monster/data/model/AWQ-Llama-3.2-1B-g128-gemm"
_PROMPT = "The capital city of France is named"


def _live_descendant_pids() -> set[int]:
    """Return descendant process IDs when psutil is available."""
    try:
        import psutil
    except ImportError:
        return set()
    try:
        return {child.pid for child in psutil.Process().children(recursive=True)}
    except (psutil.Error, OSError):
        return set()


def _runtime_pids(runtime: object) -> set[int]:
    pids = set()
    get_all_child_pids = getattr(runtime, "get_all_child_pids", None)
    if callable(get_all_child_pids):
        try:
            pids.update(
                pid for pid in get_all_child_pids() if isinstance(pid, int) and pid > 0
            )
        except (AttributeError, RuntimeError):
            pass
    for name in ("pid", "process_id", "runtime_pid"):
        value = getattr(runtime, name, None)
        if isinstance(value, int) and value > 0:
            pids.add(value)
    for name in ("process", "server_process", "manager"):
        process = getattr(runtime, name, None)
        value = getattr(process, "pid", None)
        if isinstance(value, int) and value > 0:
            pids.add(value)
    return pids


def _assert_pids_stopped(pids: set[int]) -> None:
    if not pids:
        return
    try:
        import psutil
    except ImportError:
        return
    deadline = time.monotonic() + 10
    alive = set(pids)
    while time.monotonic() < deadline:
        alive = set()
        for pid in pids:
            try:
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    if process.status() != psutil.STATUS_ZOMBIE:
                        alive.add(pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
        if not alive:
            return
        time.sleep(0.1)
    pytest.fail(
        f"SGLang runtime processes did not exit after shutdown: {sorted(alive)}"
    )


def _shutdown_model(model: object) -> None:
    """Shut down the public wrapper or its in-process SGLang runtime."""
    shutdown = getattr(model, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    runtime = getattr(model, "model", None)
    shutdown = getattr(runtime, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _tokenizer_and_ids(model: object, prompt: str) -> tuple[str, list[int]]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AssertionError("SGLang model must expose its tokenizer")

    render = prompt
    if getattr(tokenizer, "chat_template", None) and hasattr(
        tokenizer, "apply_chat_template"
    ):
        render = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer(render, return_tensors="pt")
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise AssertionError(
            f"Expected one tokenized prompt, got shape {tuple(input_ids.shape)}"
        )
    ids = input_ids[0].detach().cpu().tolist()
    encoded_again = tokenizer(render, return_tensors="pt")
    input_ids_again = (
        encoded_again["input_ids"]
        if isinstance(encoded_again, dict)
        else encoded_again.input_ids
    )
    assert input_ids_again[0].detach().cpu().tolist() == ids
    assert ids
    return render, ids


def _assert_deterministic_generation(model_path: str) -> None:
    if not Path(model_path).exists():
        pytest.skip(f"missing local model path: {model_path}")

    before = _live_descendant_pids()
    model = GPTQModel.load(model_path, device="cuda:0", backend=BACKEND.SGLANG)
    runtime_pids = _runtime_pids(getattr(model, "model", None))
    try:
        assert model._runtime_generate is sglang_generate
        rendered_prompt, input_ids = _tokenizer_and_ids(model, _PROMPT)
        text_output = model.generate(
            prompts=rendered_prompt,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1,
        )
        ids_output = model.generate(
            input_ids=input_ids,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1,
        )
        assert isinstance(text_output, str)
        assert isinstance(ids_output, str)
        assert text_output == ids_output
    finally:
        runtime = getattr(model, "model", None)
        runtime_pids.update(_runtime_pids(runtime))
        try:
            _shutdown_model(model)
        finally:
            del model
            _assert_pids_stopped(runtime_pids | (_live_descendant_pids() - before))


class TestLoadSglang:
    @classmethod
    def setup_class(cls):
        if importlib.util.find_spec("sglang") is None or not SGLANG_AVAILABLE:
            pytest.skip(SGLANG_INSTALL_HINT)
        if importlib.util.find_spec("flashinfer") is None:
            pytest.skip(
                "flashinfer is required by SGLang integration; install gptqmodel['sglang']"
            )
        if not torch.cuda.is_available():
            pytest.skip(
                "SGLang integration requires CUDA/HIP; no CUDA device is available"
            )

        cls.model_id = os.environ.get("GPTQMODEL_SGLANG_MODEL", _DEFAULT_MODEL)
        cls.sharded_model_id = os.environ.get(
            "GPTQMODEL_SGLANG_SHARDED_MODEL", _DEFAULT_SHARDED_MODEL
        )
        cls.awq_model_id = _DEFAULT_AWQ_MODEL

    def test_load_sglang_gptq(self):
        _assert_deterministic_generation(self.model_id)

    def test_load_sglang_gptq_sharded(self):
        _assert_deterministic_generation(self.sharded_model_id)

    def test_load_sglang_awq(self):
        if not Path(self.awq_model_id).exists():
            pytest.skip(
                f"missing local AWQ model path: {self.awq_model_id}"
            )
        _assert_deterministic_generation(self.awq_model_id)

    def test_hf_sglang_deterministic_first_token(self):
        if not Path(self.model_id).exists():
            pytest.skip(f"missing local model path: {self.model_id}")

        hf_model = GPTQModel.load(
            self.model_id,
            device="cuda:0",
            backend=BACKEND.GPTQ_TORCH,
        )
        try:
            hf_rendered_prompt, hf_input_ids = _tokenizer_and_ids(hf_model, _PROMPT)
            input_tensor = torch.tensor(
                [hf_input_ids],
                dtype=torch.long,
                device=hf_model.model.device,
            )
            hf_tokens = hf_model.generate(
                input_tensor,
                do_sample=False,
                max_new_tokens=1,
            )
            hf_new_token_ids = hf_tokens[0, len(hf_input_ids) :].detach().cpu().tolist()
            hf_text = hf_model.tokenizer.decode(
                hf_new_token_ids,
                skip_special_tokens=True,
            )
        finally:
            del hf_model
            torch_empty_cache()

        before = _live_descendant_pids()
        sglang_model = GPTQModel.load(
            self.model_id,
            device="cuda:0",
            backend=BACKEND.SGLANG,
        )
        runtime_pids = _runtime_pids(getattr(sglang_model, "model", None))
        try:
            sglang_rendered_prompt, sglang_input_ids = _tokenizer_and_ids(
                sglang_model,
                _PROMPT,
            )
            assert sglang_rendered_prompt == hf_rendered_prompt
            assert sglang_input_ids == hf_input_ids
            sglang_text = sglang_model.generate(
                input_ids=sglang_input_ids,
                do_sample=False,
                max_new_tokens=1,
            )
            assert sglang_text == hf_text
        finally:
            runtime = getattr(sglang_model, "model", None)
            runtime_pids.update(_runtime_pids(runtime))
            _shutdown_model(sglang_model)
            del sglang_model
            _assert_pids_stopped(runtime_pids | (_live_descendant_pids() - before))

    def test_evalution_sglang_smoke(self):
        if not Path(self.model_id).exists():
            pytest.skip(f"missing local model path: {self.model_id}")
        try:
            from tests.eval import evaluate, get_eval_task_metrics, import_evalution

            import_evalution()
        except ValueError as exc:
            pytest.skip(str(exc))

        result = evaluate(
            model_or_id_or_path=self.model_id,
            tasks=["arc_challenge"],
            batch_size=1,
            llm_backend="sglang",
            output_path=None,
            gen_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 1},
            model_args={
                "device": "cuda",
                "tp_size": 1,
                "mem_fraction_static": 0.8,
                "seed": 0,
            },
            suite_kwargs={"max_rows": 1},
        )
        assert get_eval_task_metrics(result, "arc_challenge")
        execution = result.get("engine", {}).get("execution", {})
        assert execution.get("generation_backend") == "sglang.generate"
