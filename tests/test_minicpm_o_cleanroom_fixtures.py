# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/minicpm_o/SPEC.md; see NOTICE.
"""Golden-fixture checks for the clean-room MiniCPM-o 2.6 TTS codec and sampling components.

Fixtures and their tolerances are described in tests/fixtures/cleanroom/minicpm_o/README.md.
"""

import importlib.util
import json
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from gptqmodel.models.ports.minicpm_o import modeling_minicpmo as minicpmo_45
from gptqmodel.models.ports.minicpm_o import tts_26
from gptqmodel.models.ports.minicpm_o.cache import cache_from_pairs, cache_pairs


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cleanroom" / "minicpm_o"
VQ_AVAILABLE = find_spec("vector_quantize_pytorch") is not None
needs_vq = pytest.mark.skipif(not VQ_AVAILABLE, reason="vector-quantize-pytorch is not installed")
# The fixture README allows atol=rtol=1e-5 when the implementation
# deliberately changes op order; the Transformers 5.23 port migration
# (d2dfb9385) did. Integer outputs still compare exactly by dtype.
EXACT = {"atol": 1e-5, "rtol": 1e-5}


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _meta():
    return json.loads((FIXTURES / "fixtures_meta.json").read_text())


def _tensors(name):
    return load_file(str(FIXTURES / f"{name}.safetensors"))


def _weights_of(tensors):
    return {key[2:]: value for key, value in tensors.items() if key.startswith("w.")}


def _recipe():
    spec = importlib.util.spec_from_file_location("minicpm_o_weights_recipe", FIXTURES / "weights_recipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tts_config():
    values = json.loads((FIXTURES / "tts_config.json").read_text())
    for key in ("model_type", "transformers_version", "architectures", "_name_or_path", "dtype"):
        values.pop(key, None)
    return tts_26.ConditionalChatTTSConfig(**values)


@pytest.fixture(scope="module")
def tts_model():
    if not VQ_AVAILABLE:
        pytest.skip("vector-quantize-pytorch is not installed")
    model = tts_26.ConditionalChatTTS(_tts_config()).eval()
    model.load_state_dict(_recipe().load("tts"), strict=True)
    return model


# ---------------------------------------------------------------------------------------------
# Repetition penalty and gen_logits (SPEC 6, 7)


PROCESSOR_CLASSES = [tts_26.CustomRepetitionPenaltyLogitsProcessorRepeat, minicpmo_45.CustomRepetitionPenaltyLogitsProcessorRepeat]


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES, ids=["2.6", "4.5"])
def test_repetition_penalty_cases(processor_cls):
    spec = json.loads((FIXTURES / "processors.json").read_text())
    tensors = _tensors("processors")
    for case in spec["repetition_cases"]:
        processor = processor_cls(case["penalty"], case["max_input_ids"], case["past_window"])
        scores = tensors[case["scores"]].clone()
        history = tensors[case["input_ids"]]
        out = processor(history, scores)
        torch.testing.assert_close(out, tensors[case["expected"]], **EXACT, msg=case["name"])
        torch.testing.assert_close(scores, tensors[case["scores"]], **EXACT)  # input untouched
        assert out.is_contiguous() and out.dtype == scores.dtype


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES, ids=["2.6", "4.5"])
def test_repetition_penalty_constructor_errors(processor_cls):
    spec = json.loads((FIXTURES / "processors.json").read_text())
    for case in spec["constructor_errors"]:
        penalty = eval(case["penalty"])  # literals recorded by the fixture generator
        with pytest.raises(ValueError) as excinfo:
            processor_cls(penalty, 16, 16)
        assert str(excinfo.value) == case["message"]
    processor_cls(np.float64(1.5), 16, 16)


def _describe(warpers, processors):
    described_warpers = []
    for warper in warpers:
        if isinstance(warper, tts_26.TopPLogitsWarper):
            described_warpers.append(f"TopPLogitsWarper(top_p={warper.top_p},min_keep={warper.min_tokens_to_keep})")
        else:
            described_warpers.append(f"TopKLogitsWarper(top_k={warper.top_k})")
    described_processors = [
        f"Repeat(penalty={p.penalty},max_input_ids={p.max_input_ids},past_window={p.past_window})" for p in processors
    ]
    return {"warpers": described_warpers, "processors": described_processors}


@pytest.mark.parametrize("version", ["2.6", "4.5"])
def test_gen_logits_cases(version):
    spec = json.loads((FIXTURES / "processors.json").read_text())
    tensors = _tensors("processors")
    factory = tts_26.gen_logits if version == "2.6" else minicpmo_45.gen_logits
    for case in spec["gen_logits_cases"]:
        kwargs = case["kwargs_26"] if version == "2.6" else case["kwargs_45"]
        built = factory(case["num_code"], **kwargs)
        assert isinstance(built, tuple) and len(built) == 2
        warpers, processors = built
        assert isinstance(warpers, list) and isinstance(processors, list)
        assert _describe(warpers, processors) == case["built"], case["name"]
        logits = tensors["gl.logits"].clone()
        for processor in processors:
            logits = processor(tensors["gl.hist"], logits)
        for warper in warpers:
            logits = warper(tensors["gl.hist"], logits)
        torch.testing.assert_close(logits, tensors[case["expected"]], **EXACT, msg=case["name"])


def test_gen_logits_rejects_int_penalty():
    with pytest.raises(ValueError, match="strictly positive float"):
        tts_26.gen_logits(16, repetition_penalty=2)
    with pytest.raises(ValueError, match="strictly positive float"):
        minicpmo_45.gen_logits(16, repetition_penalty=2)


# ---------------------------------------------------------------------------------------------
# ConvNeXt block, conv stack, GFSQ, DVAE (SPEC 2-5)


@pytest.mark.parametrize("name", ["convnext_ls", "convnext_k3d1", "convnext_noscale"])
def test_convnext_block(name):
    meta = _meta()["small_modules"][name]
    tensors = _tensors(name)
    block = tts_26.ConvNeXtBlock(**meta["ctor"]).eval()
    assert sorted(block.state_dict()) == meta["keys"]
    block.load_state_dict(_weights_of(tensors), strict=True)
    with torch.no_grad():
        out = block(tensors["input"])
        torch.testing.assert_close(out, tensors["output"], **EXACT)
        torch.testing.assert_close(block(tensors["input"], torch.ones(1)), tensors["output"], **EXACT)


def test_dvae_decoder_stack():
    meta = _meta()["small_modules"]["dvae_decoder"]
    tensors = _tensors("dvae_decoder")
    stack = tts_26.DVAEDecoder(**meta["ctor"]).eval()
    assert sorted(stack.state_dict()) == meta["keys"]
    stack.load_state_dict(_weights_of(tensors), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(stack(tensors["input"]), tensors["output"], **EXACT)
        torch.testing.assert_close(stack(tensors["input"], conditioning=tensors["input"]), tensors["output"], **EXACT)


@needs_vq
def test_gfsq_both_directions():
    meta = _meta()["small_modules"]["gfsq"]
    tensors = _tensors("gfsq")
    gfsq = tts_26.GFSQ(**meta["ctor"]).eval()
    assert sorted(gfsq.state_dict()) == meta["keys"]
    assert gfsq.n_ind == meta["n_ind"] and gfsq.G == 2 and gfsq.R == 2
    gfsq.load_state_dict(_weights_of(tensors), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(gfsq._embed(tensors["embed.codes"]), tensors["embed.output"], **EXACT)
        indices = gfsq(tensors["forward.input"].clone())
    assert indices.dtype == tensors["forward.indices"].dtype
    torch.testing.assert_close(indices, tensors["forward.indices"], **EXACT)


@pytest.fixture(scope="module")
def dvae():
    if not VQ_AVAILABLE:
        pytest.skip("vector-quantize-pytorch is not installed")
    module = tts_26.DVAE().eval()
    module.load_state_dict(_recipe().load("dvae"), strict=True)
    return module


def test_dvae_decode(dvae):
    tensors = _tensors("dvae_io")
    for codes, mel in (("decode.codes1", "decode.mel1"), ("decode.codes2", "decode.mel2")):
        torch.testing.assert_close(dvae(tensors[codes], mode="decode"), tensors[mel], **EXACT)
        torch.testing.assert_close(dvae(tensors[codes]), tensors[mel], **EXACT)


def test_dvae_encode_and_roundtrip(dvae):
    tensors = _tensors("dvae_io")
    for mel, indices in (("encode.mel", "encode.indices"), ("encode.mel_odd", "encode.indices_odd")):
        source = tensors[mel].clone()
        codes = dvae(source, mode="encode")
        torch.testing.assert_close(source, tensors[mel], **EXACT)  # caller's tensor untouched
        assert codes.dtype == tensors[indices].dtype
        torch.testing.assert_close(codes, tensors[indices], **EXACT)
    codes = dvae(tensors["encode.mel"], mode="encode")
    torch.testing.assert_close(dvae(codes.to(torch.long), mode="decode"), tensors["roundtrip.mel"], **EXACT)


@needs_vq
def test_dvae_state_dict_layout():
    manifest = json.loads((FIXTURES / "dvae.manifest.json").read_text())["shapes"]
    with torch.device("meta"):
        module = tts_26.DVAE()
    assert {k: list(v.shape) for k, v in module.state_dict().items()} == manifest


# ---------------------------------------------------------------------------------------------
# ConditionalChatTTS.generate and decode_to_mel_specs (SPEC 8, 9)

SCENARIOS = {
    "s1": "s1_first_call_no_eos",
    "s2": "s2_continuation",
    "s3": "s3_eos_mid_run",
    "s4": "s4_eos_at_step0_first_call",
    "s5": "s5_eos_at_step0_continuation",
    "s6": "s6_plain_sampling_int_eos_force_no_stop",
    "s7": "s7_temperature_len1",
}


def _cache(tensors, prefix):
    return cache_from_pairs([(tensors[f"{prefix}.cache.0.keys"].clone(), tensors[f"{prefix}.cache.0.values"].clone())])


@pytest.mark.parametrize("sid", sorted(SCENARIOS))
def test_generate_scenarios(tts_model, sid, monkeypatch):
    meta = _meta()["tts"]
    scenario = meta["scenarios"][SCENARIOS[sid]]
    kwargs = scenario["kwargs"]
    tensors = _tensors("tts_io")

    if kwargs["logits_warpers"]:
        warpers, processors = tts_26.gen_logits(16, top_P=0.7, top_K=5, repetition_penalty=1.5)
    else:
        warpers, processors = [], []
    eos = kwargs["eos_token"]
    eos = torch.tensor(eos, dtype=torch.long) if isinstance(eos, list) else eos
    cache = _cache(tensors, f"{sid}.in")

    recorded = []
    real_multinomial = torch.multinomial

    def recording_multinomial(probs, *args, **kwargs):
        recorded.append(probs.detach().clone())
        return real_multinomial(probs, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", recording_multinomial)
    torch.manual_seed(scenario["seed"])
    out = tts_model.generate(
        input_ids=tensors[f"{sid}.in.audio_input_ids"].clone(),
        past_key_values=cache,
        temperature=torch.tensor(kwargs["temperature"], dtype=torch.float32),
        eos_token=eos,
        streaming_tts_text_mask=tensors[f"{sid}.in.text_mask"],
        force_no_stop=kwargs.get("force_no_stop", False),
        min_new_token=kwargs["min_new_token"],
        max_new_token=kwargs["max_new_token"],
        logits_warpers=warpers,
        logits_processors=processors,
    )
    monkeypatch.undo()

    assert isinstance(out, tts_26.ConditionalChatTTSGenerationOutput)
    assert len(recorded) == scenario["steps"]
    torch.testing.assert_close(torch.stack(recorded), tensors[f"{sid}.out.probs"], atol=1e-6, rtol=0)
    assert torch.equal(out.audio_input_ids, tensors[f"{sid}.out.audio_input_ids"])
    assert torch.equal(out.new_ids, tensors[f"{sid}.out.new_ids"])
    assert out.audio_input_ids.shape[1] == scenario["audio_input_ids_len"]
    assert out.new_ids.shape[1] == scenario["new_ids_len"]
    assert out.past_key_values is cache
    assert cache.get_seq_length() == scenario["cache_len_after"]
    (keys, values), = cache_pairs(cache)
    torch.testing.assert_close(keys, tensors[f"{sid}.out.cache.0.keys"], **EXACT)
    torch.testing.assert_close(values, tensors[f"{sid}.out.cache.0.values"], **EXACT)
    assert isinstance(out.finished, torch.Tensor) and out.finished.dim() == 0
    assert bool(out.finished) is scenario["finished"]
    assert meta["condition_length"] == 11


def test_generate_rejects_inconsistent_cache(tts_model):
    tensors = _tensors("tts_io")
    with pytest.raises(AssertionError):
        tts_model.generate(
            input_ids=tensors["s2.in.audio_input_ids"],
            past_key_values=_cache(tensors, "s1.in"),
            temperature=tensors["inputs.temperature"],
            eos_token=15,
            streaming_tts_text_mask=tensors["s2.in.text_mask"],
            max_new_token=2,
        )


def test_decode_to_mel_specs(tts_model):
    tensors = _tensors("tts_io")
    torch.testing.assert_close(tts_model.decode_to_mel_specs(tensors["mel.a.codes"]), tensors["mel.a.out"], **EXACT)
    mel_b = tts_model.decode_to_mel_specs([tensors["mel.b.codes0"], tensors["mel.b.codes1"]])
    torch.testing.assert_close(mel_b, tensors["mel.b.out"], **EXACT)

    empty = tts_model.decode_to_mel_specs([])
    assert isinstance(empty, np.ndarray) and empty.dtype == np.float32 and empty.shape == (0,)
    with pytest.raises(Exception):
        tts_model.decode_to_mel_specs([torch.zeros(0, 4, dtype=torch.long)])


# ---------------------------------------------------------------------------------------------
# Regression: quantizer constants survive from_pretrained (SPEC 10)


@needs_vq
def test_minicpmo_26_save_load_decodes_like_fresh(tmp_path):
    from gptqmodel.models.ports.minicpm_o.modeling_minicpmo import MiniCPMO
    from tests.test_minicpm_o_local_port import tiny_config

    config = tiny_config(version="2.6", tts=True)
    config.tts_config.num_vq = 4
    torch.manual_seed(5)
    fresh = MiniCPMO._from_config(config).eval()
    tts_weights = _recipe().load("tts")
    fresh.tts.dvae.load_state_dict({k[5:]: v for k, v in tts_weights.items() if k.startswith("dvae.")}, strict=True)
    codes = _tensors("tts_io")["mel.a.codes"]
    expected = fresh.tts.decode_to_mel_specs(codes)
    torch.testing.assert_close(expected, _tensors("tts_io")["mel.a.out"], **EXACT)

    fresh.save_pretrained(tmp_path)
    loaded = MiniCPMO.from_pretrained(tmp_path, attn_implementation="eager").eval()
    fresh_buffers = dict(fresh.tts.dvae.vq_layer.quantizer.named_buffers())
    for name, buffer in loaded.tts.dvae.vq_layer.quantizer.named_buffers():
        assert torch.equal(buffer, fresh_buffers[name]), name
    torch.testing.assert_close(loaded.tts.decode_to_mel_specs(codes), expected, **EXACT)
