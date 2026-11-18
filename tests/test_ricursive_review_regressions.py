# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for accumulation, masks, tying, shared calls, and real TP."""

import copy
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from ricursive_test_utils import make_model


@pytest.mark.parametrize("return_dict", [True, False])
def test_accumulated_loss_matches_whole_batch_gradients(return_dict):
    model = make_model().train()
    ids = torch.tensor([[3, 5, 6, 8], [2, 3, 4, 5]])
    labels = ids.clone()
    labels[1, 2:] = -100
    whole = model(ids, labels=labels, use_cache=False, return_dict=return_dict)[0]
    whole.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    total = (labels[:, 1:] != -100).sum()
    losses = []
    for index in range(2):
        loss = model(
            ids[index : index + 1],
            labels=labels[index : index + 1],
            use_cache=False,
            num_items_in_batch=total,
            return_dict=return_dict,
            logits_to_keep=1,
        )[0]
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(sum(losses), whole.detach(), atol=1e-6, rtol=1e-6)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=2e-5, msg=name)


def test_accumulation_with_empty_label_microbatch_matches_full_gradient():
    model = make_model().train()
    ids = torch.tensor([[3, 4, 5, 6], [2, 3, 4, 5]])
    labels = ids.clone()
    labels[0] = -100
    token_count = int((labels[:, 1:] != -100).sum())

    full_logits = model(ids, use_cache=False).logits.float()
    expected_loss = (
        torch.nn.functional.cross_entropy(
            full_logits[:, :-1, :].reshape(-1, full_logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="sum",
        )
        / token_count
    )
    expected_loss.backward()
    expected_gradient = model.lm_head.weight.grad.clone()
    model.zero_grad()

    losses = [
        model(
            ids[index : index + 1],
            labels=labels[index : index + 1],
            num_items_in_batch=token_count,
            logits_to_keep=1,
            return_dict=False,
            use_cache=False,
        )[0]
        for index in range(2)
    ]
    assert losses[0].item() == 0.0
    assert bool(torch.isfinite(losses[1]))
    actual_loss = sum(losses)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-6)
    actual_loss.backward()
    torch.testing.assert_close(model.lm_head.weight.grad, expected_gradient, atol=2e-6, rtol=2e-5)


def test_trainer_accumulation_preserves_optimizer_update(tmp_path):
    try:
        from transformers import Trainer, TrainingArguments
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Trainer dependencies unavailable: {exc}")
    initial = make_model(use_cache=False)
    ids = torch.tensor([[3, 5, 6, 8], [2, 3, 4, 5]])
    labels = ids.clone()
    labels[1, 2:] = -100
    dataset = [{"input_ids": ids[i], "labels": labels[i]} for i in range(2)]
    weights = []
    for batch, accumulation in [(2, 1), (1, 2)]:
        model = copy.deepcopy(initial)
        args = TrainingArguments(
            output_dir=str(tmp_path / str(batch)),
            use_cpu=True,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accumulation,
            max_steps=1,
            max_grad_norm=0,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
            dataloader_pin_memory=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
        )
        assert trainer.model_accepts_loss_kwargs
        trainer.train()
        weights.append({name: p.detach().clone() for name, p in model.named_parameters()})
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name], weights[1][name], atol=2e-6, rtol=2e-5, msg=name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_additive_mask_preserves_empty_rows_and_forbidden_keys(dtype):
    model = make_model().to(dtype=dtype)
    attn = model.model.layers[0].self_attn
    x = torch.randn(1, 6, model.config.hidden_size, dtype=dtype)
    positions = torch.arange(6)
    allowed = torch.ones(1, 1, 6, 6, dtype=torch.bool).tril()
    allowed[..., 2, :] = False
    allowed[..., 4, :] = False
    additive = torch.zeros(allowed.shape, dtype=dtype).masked_fill(~allowed, float("-inf"))
    options = {"position_ids": positions[None], "output_attentions": True}
    expected, expected_weights = attn(x, attention_mask=allowed, **options)
    actual, weights = attn(x, attention_mask=additive, **options)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(weights, expected_weights, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(actual[:, [2, 4]]) == 0
    assert torch.count_nonzero(weights.masked_select(~allowed)) == 0


@pytest.mark.parametrize("second_request", ["keep_three", "full", "labels"])
def test_shared_model_concurrent_calls_keep_independent_logits(second_request):
    model = make_model()
    ids = torch.tensor([[3, 5, 6, 8]])
    requests = [
        {"logits_to_keep": 1},
        {
            "keep_three": {"logits_to_keep": 3},
            "full": {},
            "labels": {"labels": ids},
        }[second_request],
    ]

    def run(options):
        with torch.no_grad():
            return model(ids, use_cache=False, **options)

    reference = [run(options) for options in requests]
    barrier = threading.Barrier(2, timeout=30)

    def synchronize_calls(*_):
        barrier.wait()

    handle = model.lm_head.register_forward_pre_hook(synchronize_calls)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            actual = list(executor.map(run, requests))
    finally:
        handle.remove()
    assert not model.lm_head._forward_pre_hooks
    for out, expected in zip(actual, reference):
        torch.testing.assert_close(out.logits, expected.logits)
        if expected.loss is not None:
            torch.testing.assert_close(out.loss, expected.loss)


@pytest.mark.parametrize("via_config", [False, True])
@torch.no_grad()
def test_tied_embeddings_construct_resize_and_roundtrip(tmp_path, via_config):
    if via_config:
        untied = make_model()
        config = copy.deepcopy(untied.config)
        config.tie_word_embeddings = True
        model = type(untied)(config).eval()
    else:
        model = make_model(tie_word_embeddings=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    model.resize_token_embeddings(70)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    ids = torch.tensor([[3, 65, 69]])
    expected = model(ids, use_cache=False).logits
    model.save_pretrained(tmp_path)
    restored = type(model).from_pretrained(tmp_path).eval()
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    torch.testing.assert_close(restored(ids, use_cache=False).logits, expected)


@pytest.mark.parametrize(
    "runtime,dtype", [("cpu", "float32"), ("cuda", "float32"), ("cuda", "float16"), ("cuda", "bfloat16")]
)
def test_real_tensor_parallel_matches_unsharded_model(runtime, dtype):
    # Portable Torch ops: cuda is the execution namespace for CUDA and ROCm.
    if runtime == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2):
        pytest.skip("Two CUDA/ROCm runtime devices required")
    env = dict(os.environ)
    if runtime == "cpu":
        # Transformers otherwise selects the available accelerator during loading.
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("ricursive_tp_worker.py")),
            runtime,
            dtype,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('{"dtype"'):
            print(line)
