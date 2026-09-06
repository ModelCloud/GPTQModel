import pytest
import torch

from gptqmodel.quantization.qvq_rank8_capture import (
    Rank8Capture,
    Rank8Document,
    capture_rank8_calibration,
)


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 16)
        self.projection = torch.nn.Linear(16, 16)

    def forward(self, input_ids, attention_mask=None):
        return self.projection(self.embedding(input_ids))


def request(**kwargs):
    return Rank8Capture(
        ("projection",),
        (Rank8Document("train", {"input_ids": torch.tensor([[1, 2, 3, 0]]),
                                 "attention_mask": torch.tensor([[1, 1, 1, 0]])}),),
        (Rank8Document("heldout", {"input_ids": torch.tensor([[4, 5, 6]])}),),
        **kwargs,
    )


def test_capture_preserves_documents_padding_and_dense_inputs():
    teacher = Teacher().eval()
    original = {k: v.clone() for k, v in teacher.state_dict().items()}
    captured = capture_rank8_calibration(teacher, request(rows_per_document=2))["projection"]
    torch.testing.assert_close(captured.train_inputs, teacher.embedding(torch.tensor([1, 2])))
    torch.testing.assert_close(captured.heldout_inputs, teacher.embedding(torch.tensor([4, 5])))
    assert captured.train_document_ids == ("train",)
    assert captured.heldout_document_ids == ("heldout",)
    assert not teacher.projection._forward_pre_hooks
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_capture_failure_removes_hooks_and_enforces_budget():
    teacher = Teacher().eval()
    with pytest.raises(ValueError, match="max_bytes"):
        capture_rank8_calibration(teacher, request(max_bytes=1))
    assert not teacher.projection._forward_pre_hooks
    # A new capture must succeed after the rejected collection.
    assert capture_rank8_calibration(teacher, request())["projection"].train_inputs.shape == (3, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        capture_rank8_calibration(teacher, request())


def test_capture_rejects_relabelled_overlap_and_evaluation_source():
    with pytest.raises(ValueError, match="calibration"):
        request(source_kind="evaluation")
    docs = request()
    with pytest.raises(ValueError, match="overlap"):
        Rank8Capture(docs.module_names, docs.train,
                     (Rank8Document("different-name", docs.train[0].inputs),))
    docs.heldout[0].inputs.clear()
    docs.heldout[0].inputs.update(docs.train[0].inputs)
    with pytest.raises(ValueError, match="overlap"):
        capture_rank8_calibration(Teacher().eval(), docs)


def test_capture_raises_on_missing_or_repeated_module_execution():
    teacher = Teacher().eval()
    teacher.extra = torch.nn.Linear(16, 16).eval()
    docs = request()
    with pytest.raises(ValueError, match="did not execute"):
        capture_rank8_calibration(teacher, Rank8Capture(("extra",), docs.train, docs.heldout))
    assert not teacher.extra._forward_pre_hooks
    teacher.embedding.register_forward_hook(lambda module, args, output: teacher.projection(output))
    with pytest.raises(ValueError, match="more than once"):
        capture_rank8_calibration(teacher, docs)
    assert not teacher.projection._forward_pre_hooks


def test_processor_rejects_unconsumed_rank8_capture():
    from test_qvq_lifecycle import _processor

    from gptqmodel.quantization import FORMAT, QVQConfig

    processor = _processor(qcfg=QVQConfig(
        bits=2, format=FORMAT.QVQ_V2B2_P32, device="cpu", offload_to_disk=False,
    ))
    captured = capture_rank8_calibration(Teacher().eval(), request())
    processor.set_rank8_calibration("projection", captured["projection"])
    with pytest.raises(ValueError, match="were not quantized: projection"):
        processor.finalize(None)


def test_captured_inputs_finish_quantization_and_bind_original_teacher():
    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.quantization.qvq_rank8 import finish_rank8_quantization

    teacher = Teacher().eval()
    captured = capture_rank8_calibration(teacher, request())["projection"]
    x = captured.train_inputs
    result = quantize_qvq_linear(
        teacher.projection.weight.detach(), x.T @ x / x.shape[0],
        bits=2, bank_count=2, v2b2_p32=True, bias=teacher.projection.bias,
        rank8_calibration=captured,
    )
    assert result.rank8_fit_report["teacher_hash"] == captured.teacher_hash
    assert result.rank8_fit_report["train_document_ids"] == ["train"]
    assert result.rank8_fit_report["heldout_document_ids"] == ["heldout"]
    with torch.no_grad():
        teacher.projection.weight.add_(1)
    with pytest.raises(ValueError, match="teacher state changed"):
        finish_rank8_quantization(
            result, teacher.projection.weight, teacher.projection.bias, captured,
            bits=2, codebook_version="pgc16-v1",
        )


def test_public_quantize_captures_and_fits_in_one_job(tmp_path, monkeypatch):
    from test_qvq_activation_e2e import _build_tiny_llama_fixture, _calibration_dataset
    from transformers import LlamaForCausalLM

    from gptqmodel import GPTQModel
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization import QVQConfig, qvq_rank8
    from gptqmodel.utils.backend import BACKEND

    tokenizer = _build_tiny_llama_fixture(tmp_path)
    dense = LlamaForCausalLM.from_pretrained(tmp_path).float().eval()
    wrapper = LlamaQModel(
        dense, quantized=False, tokenizer=tokenizer, model_local_path=str(tmp_path),
        quantize_config=QVQConfig(
            bits=2, format="v2b2-g32", rounding="block_ldlq", device=torch.device("cpu"), offload_to_disk=False,
        ),
    )
    # Synthetic pipeline fixture: distinct documents with a shared token
    # vocabulary. This tests artifact plumbing, not held-out model quality.
    train = Rank8Document("train", {"input_ids": torch.arange(1, 33).reshape(1, -1)})
    heldout = Rank8Document("heldout", {"input_ids": torch.arange(32, 0, -1).reshape(1, -1)})
    capture = Rank8Capture(
        ("model.layers.0.self_attn.q_proj",), (train,), (heldout,),
    )
    reports = []
    finish = qvq_rank8.finish_rank8_quantization

    def record_finish(*args, **kwargs):
        result = finish(*args, **kwargs)
        reports.append(result.rank8_fit_report)
        return result

    monkeypatch.setattr(qvq_rank8, "finish_rank8_quantization", record_finish)
    wrapper.quantize(
        _calibration_dataset(tokenizer), backend=BACKEND.QVQ, calibration_data_min_length=1,
        rank8_capture=capture,
    )
    child = wrapper.model.model.layers[0].self_attn.q_proj
    assert isinstance(child, QVQLinear)
    assert wrapper.quantized
    assert len(reports) == 1
    assert reports[0]["train_document_ids"] == ["train"]
    assert reports[0]["heldout_document_ids"] == ["heldout"]
    assert reports[0]["validated"]
    output = tmp_path / "quantized"
    wrapper.save(output)
    reloaded = GPTQModel.load(str(output), backend=BACKEND.QVQ, device_map={"": "cpu"}, dtype=torch.float32)
    loaded_child = reloaded.model.model.layers[0].self_attn.q_proj
    torch.testing.assert_close(loaded_child.rank8_A, child.rank8_A, rtol=0, atol=0)
    torch.testing.assert_close(loaded_child.rank8_B, child.rank8_B, rtol=0, atol=0)
    qvq_rank8.prepare_rank8(child, qvq_rank8.P32WindowConfig(recovery_mode="on"))
    qvq_rank8.prepare_rank8(loaded_child, qvq_rank8.P32WindowConfig(recovery_mode="on"))
    x = torch.randn(3, 64)
    torch.testing.assert_close(loaded_child(x), child(x), rtol=0, atol=0)
