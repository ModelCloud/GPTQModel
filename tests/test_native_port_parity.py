# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Vendored Transformers 5.23 families must equal Transformers 5.23 exactly.

Each fixture under tests/fixtures/ports/native_parity was dumped from
Transformers' own model tester by scripts/generate_native_port_parity_fixtures.py.
For every model class, the native class (the oracle; tests may import
``transformers.models``) and the vendored class are built from the same config,
must have identical state_dict keys/shapes, and with identical weights must
produce bit-identical outputs: the default forward (SDPA) plus, with eager
attention, every call variant (hidden states, attentions, return_dict=False,
cached decoding, loss, greedy generate). bart, opt and llama4 have hand-written
tiny configs.
"""

import importlib
import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports" / "native_parity"
CASES = sorted(path.stem for path in FIXTURES.glob("*.json"))


def _classes(family, module, name):
    native = getattr(importlib.import_module(f"transformers.models.{family}.{module}"), name)
    vendored = getattr(importlib.import_module(f"gptqmodel.models.ports.native.{family}.{module}"), name)
    return native, vendored


def _inputs(family, spec):
    tensors = {}
    path = FIXTURES / f"{family}.safetensors"
    if path.exists():
        tensors.update(load_file(str(path)))
    generator = torch.Generator().manual_seed(0)
    for key in sorted(spec["float_inputs"]):
        meta = spec["float_inputs"][key]
        dtype = getattr(torch, meta["dtype"])
        tensors[key] = torch.randn(meta["shape"], generator=generator, dtype=torch.float32).to(dtype)
    tensors.update(spec["extra_inputs"])
    return tensors


def _flatten(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict) or hasattr(output, "to_tuple"):
        values = output.values() if isinstance(output, dict) else output.to_tuple()
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        return []
    return [tensor for value in values for tensor in _flatten(value)]


# Families whose *native* Transformers forward cannot run on the tester inputs here.
ORACLE_UNAVAILABLE = {
    "inkling": "native Inkling requires scipy",
}


def _class_inputs(name, inputs):
    """Multiple-choice heads take (batch, choices, seq) inputs; the testers' common inputs are 2-D."""

    if name.endswith("ForMultipleChoice"):
        return {k: v.unsqueeze(1).expand(-1, 2, -1).contiguous() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


@pytest.mark.parametrize("family", CASES)
def test_vendored_family_matches_transformers(family, tmp_path):
    if family in ORACLE_UNAVAILABLE:
        pytest.importorskip("scipy", reason=ORACLE_UNAVAILABLE[family])
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    native_config_class, vendored_config_class = _classes(family, spec["config_module"], spec["config_class"])
    assert vendored_config_class is not native_config_class
    family_inputs = _inputs(family, spec)

    for module, name in spec["model_classes"]:
        inputs = _class_inputs(name, family_inputs)
        native_class, vendored_class = _classes(family, module, name)
        native_config = native_config_class.from_dict(dict(spec["config"]))
        vendored_config = vendored_config_class.from_dict(dict(spec["config"]))
        torch.manual_seed(0)
        native = native_class(native_config).eval()
        torch.manual_seed(0)
        vendored = vendored_class(vendored_config).eval()

        native_state = native.state_dict()
        vendored_state = vendored.state_dict()
        assert list(vendored_state) == list(native_state), f"{name}: state_dict keys differ"
        assert all(vendored_state[k].shape == native_state[k].shape for k in native_state), f"{name}: shapes differ"
        vendored.load_state_dict(native_state, strict=True)
        for key, buffer in native.named_buffers():
            vendored.get_buffer(key).copy_(buffer)

        with torch.no_grad():
            expected = _flatten(native(**inputs))
            actual = _flatten(vendored(**inputs))
        assert expected, f"{name}: no tensor outputs"
        assert len(actual) == len(expected), f"{name}: output structure differs"
        for index, (want, got) in enumerate(zip(expected, actual)):
            assert torch.equal(want, got), f"{name}: output {index} differs"

        # A checkpoint written by Transformers must load into the vendored class through the
        # same conversion mappings (key renames, expert fusion) and give identical outputs.
        # The oracle is Transformers loading the same checkpoint (some native heads do not
        # round-trip bit-exactly on their own, e.g. GPT2DoubleHeadsModel).
        checkpoint = tmp_path / name
        native.save_pretrained(checkpoint)
        native_loaded = native_class.from_pretrained(checkpoint).eval()
        loaded = vendored_class.from_pretrained(checkpoint, config=vendored_config_class.from_pretrained(checkpoint)).eval()
        assert type(loaded) is vendored_class
        with torch.no_grad():
            expected_reload = _flatten(native_loaded(**inputs))
            reloaded = _flatten(loaded(**inputs))
        assert len(reloaded) == len(expected_reload), f"{name}: reloaded output structure differs"
        for index, (want, got) in enumerate(zip(expected_reload, reloaded)):
            assert torch.equal(want, got), f"{name}: output {index} differs after loading a Transformers checkpoint"


def test_every_vendored_family_has_parity_coverage():
    native = Path(__file__).resolve().parents[1] / "gptqmodel" / "models" / "ports" / "native"
    families = {p.name for p in native.iterdir() if p.is_dir() and p.name not in {"auto", "__pycache__"}}
    # No importable Transformers 5.23 model tester for these; hand-written parity tests below.
    without_tester = {"bart", "llama4", "opt"}
    assert families - without_tester <= set(CASES), sorted(families - without_tester - set(CASES))


# Llama4 has no Transformers 5.23 model tester, so its parity check is hand-written: tiny text-only and
# conditional-generation (vision + MoE text) configs, default dict outputs plus every optional output path.
LLAMA4_TEXT = {
    "vocab_size": 99, "hidden_size": 32, "intermediate_size": 37, "intermediate_size_mlp": 64,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "num_local_experts": 4, "num_experts_per_tok": 1, "interleave_moe_layer_step": 2, "attention_chunk_size": 4,
    "max_position_embeddings": 64, "pad_token_id": 0, "no_rope_layer_interval": 2, "floor_scale": 4,
}
LLAMA4_VISION = {
    "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
    "vision_output_dim": 32, "image_size": 16, "patch_size": 4, "projector_input_dim": 32,
    "projector_output_dim": 32, "pixel_shuffle_ratio": 0.5,
}
LLAMA4_IMAGE_TOKEN = 98
def _llama4_config(config_module, name):
    if name == "Llama4TextConfig":
        return config_module.Llama4TextConfig(**LLAMA4_TEXT, attn_implementation="eager")
    return config_module.Llama4Config(
        text_config=dict(LLAMA4_TEXT), vision_config=dict(LLAMA4_VISION),
        image_token_index=LLAMA4_IMAGE_TOKEN, attn_implementation="eager",
    )


def _llama4_inputs(model_name):
    pixel_values = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(1))
    if model_name == "Llama4VisionModel":
        return {"pixel_values": pixel_values}
    input_ids = torch.randint(1, 90, (2, 9), generator=torch.Generator().manual_seed(0))
    if model_name != "Llama4ForConditionalGeneration":
        return {"input_ids": input_ids}
    input_ids[:, 1:5] = LLAMA4_IMAGE_TOKEN  # 4 tokens per 16x16 image after pixel shuffle
    return {"input_ids": input_ids, "pixel_values": pixel_values}


@pytest.mark.parametrize("model_name", [
    "Llama4TextModel", "Llama4ForCausalLM", "Llama4VisionModel", "Llama4ForConditionalGeneration",
])
def test_vendored_llama4_matches_transformers(model_name, tmp_path):
    native_config_module = importlib.import_module("transformers.models.llama4.configuration_llama4")
    vendored_config_module = importlib.import_module("gptqmodel.models.ports.native.llama4.configuration_llama4")
    config_name = "Llama4TextConfig" if model_name in {"Llama4TextModel", "Llama4ForCausalLM"} else "Llama4Config"
    native_config = _llama4_config(native_config_module, config_name)
    vendored_config = _llama4_config(vendored_config_module, config_name)
    assert type(vendored_config) is not type(native_config)
    if model_name == "Llama4VisionModel":
        native_config, vendored_config = native_config.vision_config, vendored_config.vision_config

    native_class, vendored_class = _classes("llama4", "modeling_llama4", model_name)
    torch.manual_seed(0)
    native = native_class(native_config).eval()
    torch.manual_seed(0)
    vendored = vendored_class(vendored_config).eval()
    native_state = native.state_dict()
    assert list(vendored.state_dict()) == list(native_state)
    vendored.load_state_dict(native_state, strict=True)
    for key, buffer in native.named_buffers():
        vendored.get_buffer(key).copy_(buffer)

    inputs = _llama4_inputs(model_name)
    _assert_hand_variants("llama4", native, vendored, model_name, inputs)
    if model_name == "Llama4VisionModel":
        # Native crashes on output_attentions alone (NATIVE_BUGS); its attentions with both flags set
        # are the oracle for the vendored output_attentions-only call.
        with torch.no_grad():
            expected = native(**inputs, output_attentions=True, output_hidden_states=True)
            actual = vendored(**inputs, output_attentions=True)
        assert list(actual.keys()) == ["last_hidden_state", "attentions"]
        _assert_same(expected.last_hidden_state, actual.last_hidden_state, "last_hidden_state")
        _assert_same(expected.attentions, actual.attentions, "attentions")

    # A Transformers-saved checkpoint must load into the vendored class with identical outputs.
    checkpoint = tmp_path / model_name
    native.save_pretrained(checkpoint)
    native_loaded = native_class.from_pretrained(checkpoint).eval()
    loaded = vendored_class.from_pretrained(checkpoint, config=type(vendored_config).from_pretrained(checkpoint)).eval()
    assert type(loaded) is vendored_class
    with torch.no_grad():
        expected, actual = _flatten(native_loaded(**inputs)), _flatten(loaded(**inputs))
    assert expected and len(actual) == len(expected)
    for index, (want, got) in enumerate(zip(expected, actual)):
        assert torch.equal(want, got), f"{model_name}: output {index} differs after loading a Transformers checkpoint"


# ---------------------------------------------------------------------------------------------------
# Call variants. The default forward above is not enough: the vendored families carry GPT-QModel
# rewrites (5.23 output capturing instead of legacy return paths), so every optional output path,
# the tuple path, the cached decode step, the loss and greedy generation must also equal native.
# A variant is only "unsupported" when native Transformers itself raises on it; the vendored class
# must then raise the same exception type, and the pair is listed in NATIVE_UNSUPPORTED.
# ---------------------------------------------------------------------------------------------------

from transformers.cache_utils import Cache  # noqa: E402
from transformers.configuration_utils import PreTrainedConfig  # noqa: E402
from transformers.utils import ModelOutput  # noqa: E402


FORWARD_VARIANTS = {
    "default": {},
    "output_hidden_states": {"output_hidden_states": True},
    "output_attentions": {"output_attentions": True},
    "return_dict_false": {"return_dict": False},
    "return_dict_false_all_outputs": {"return_dict": False, "output_hidden_states": True, "output_attentions": True},
}
# Variants native Transformers 5.23 itself raises on (the vendored class must raise the same type,
# except for the NATIVE_BUGS where the vendored class must work and is checked separately).
# Absent capabilities (no `labels` argument, no cache returned, not a GenerationMixin) are not
# listed: those variants do not exist for the class.
NATIVE_UNSUPPORTED = {
    "llama4": {
        "Llama4VisionModel/output_attentions": (
            "native 5.23 bug: with output_attentions alone Llama4VisionModel reads output[2] of a 2-tuple "
            "(IndexError); the vendored class returns the attentions, checked against native's "
            "output_hidden_states+output_attentions call"
        ),
    },
    "timm_wrapper": {
        f"{name}/{variant}": "timm models do not expose attention weights (native raises ValueError)"
        for name in ("TimmWrapperModel", "TimmWrapperForImageClassification")
        for variant in ("output_attentions", "return_dict_false_all_outputs")
    },
}
NATIVE_BUGS = {"llama4": frozenset({"Llama4VisionModel/output_attentions"})}
CLASSIFICATION_SUFFIXES = ("ForSequenceClassification", "ForImageClassification", "ForAudioClassification", "ForMultipleChoice")
# Step-2 inputs keep only text inputs: like generation, media inputs are consumed by the prefill.
STEP_KEEP = {"input_ids", "attention_mask"}


def _assert_same(want, got, where, depth=0):
    """Structural, bit-exact equality: types, tuple lengths, output keys, caches and tensors."""

    assert depth < 12, where
    if isinstance(want, torch.Tensor):
        assert isinstance(got, torch.Tensor), f"{where}: {type(got).__name__} instead of a tensor"
        assert got.dtype == want.dtype and got.shape == want.shape, f"{where}: {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}"
        assert torch.equal(want, got), f"{where}: values differ"
        return
    if want is None or isinstance(want, (bool, int, float, str, torch.dtype, torch.device)):
        assert type(got) is type(want) and got == want, f"{where}: {got!r} != {want!r}"
        return
    assert type(got).__name__ == type(want).__name__, f"{where}: {type(got).__name__} != {type(want).__name__}"
    if isinstance(want, ModelOutput):
        assert list(got.keys()) == list(want.keys()), f"{where}: keys {list(got.keys())} != {list(want.keys())}"
        for key in want.keys():
            _assert_same(want[key], got[key], f"{where}.{key}", depth + 1)
    elif isinstance(want, (tuple, list)):
        assert len(got) == len(want), f"{where}: length {len(got)} != {len(want)}"
        for index, (w, g) in enumerate(zip(want, got)):
            _assert_same(w, g, f"{where}[{index}]", depth + 1)
    elif isinstance(want, dict):
        assert list(got) == list(want), f"{where}: keys differ"
        for key in want:
            _assert_same(want[key], got[key], f"{where}[{key!r}]", depth + 1)
    elif isinstance(want, Cache) or hasattr(want, "__dict__"):
        # Caches (framework or family-vendored hybrid caches) and their layers: compare state.
        want_vars = {k: v for k, v in vars(want).items() if not isinstance(v, (PreTrainedConfig, torch.nn.Module)) and not callable(v)}
        got_vars = {k: v for k, v in vars(got).items() if not isinstance(v, (PreTrainedConfig, torch.nn.Module)) and not callable(v)}
        assert sorted(got_vars) == sorted(want_vars), f"{where}: attributes differ"
        for key in sorted(want_vars):
            _assert_same(want_vars[key], got_vars[key], f"{where}.{key}", depth + 1)


def _run(model, method, kwargs):
    # Reseed per call: some families draw random numbers in forward/generate (diffusion masking).
    torch.manual_seed(0)
    try:
        with torch.no_grad():
            return getattr(model, method)(**kwargs), None
    except Exception as exc:  # noqa: BLE001 - native behavior is the oracle, including failures
        return None, exc


def _describe(exc):
    text = str(exc).splitlines()[0][:120] if str(exc) else ""
    return f"{type(exc).__name__}: {text}"


class _Checker:
    """Collects mismatches (all of them, not just the first) and natively unsupported variants."""

    def __init__(self, native_bugs=frozenset()):
        self.native_bugs = native_bugs
        self.failures = []
        self.unsupported = {}
        self.compared = []
        self.raised = []

    def compare(self, native, vendored, method, native_kwargs, label, vendored_kwargs=None):
        expected, native_error = _run(native, method, native_kwargs)
        actual, vendored_error = _run(vendored, method, native_kwargs if vendored_kwargs is None else vendored_kwargs)
        if native_error is not None:
            self.unsupported[label] = _describe(native_error)
            self.raised.append(label)
            if label in self.native_bugs:
                if vendored_error is not None:
                    self.failures.append(f"{label}: documented native bug, but vendored raised {_describe(vendored_error)}")
            elif vendored_error is None or type(vendored_error) is not type(native_error):
                self.failures.append(f"{label}: native raised {_describe(native_error)}; vendored gave {vendored_error!r}")
            return None, None
        if vendored_error is not None:
            self.failures.append(f"{label}: vendored raised {_describe(vendored_error)}; native did not")
            return None, None
        self.compared.append(label)
        try:
            _assert_same(expected, actual, label)
        except AssertionError as exc:
            self.failures.append(str(exc).splitlines()[0])
            return None, None
        return expected, actual


def _accepts(model_class, name):
    return name in inspect.signature(model_class.forward).parameters


def _label_kwargs(name, model_class, config, inputs):
    """Label arguments shaped for the head, or ``None`` when the class computes no loss."""

    batch = next((v for v in inputs.values() if isinstance(v, torch.Tensor)), None)
    if _accepts(model_class, "start_positions") and batch is not None:
        zeros = torch.zeros(batch.shape[0], dtype=torch.long)
        return {"start_positions": zeros, "end_positions": zeros + 1}
    if not _accepts(model_class, "labels"):
        return None
    if name.endswith(CLASSIFICATION_SUFFIXES):
        return {"labels": torch.zeros(batch.shape[0], dtype=torch.long)}
    if getattr(config, "is_encoder_decoder", False) and "decoder_input_ids" in inputs:
        return {"labels": inputs["decoder_input_ids"].clone()}
    if "input_ids" in inputs:
        return {"labels": inputs["input_ids"].clone()}
    return None


CACHE_KEYS = ("past_key_values", "cache_params")  # cache_params: Mamba-style families


def _next_steps(inputs, output, config, cache_key, keep_mask=True):
    """Inputs for one cached decode step after a use_cache=True prefill (``None`` if impossible)."""

    # The greedy next token when the head has logits, else (base models) repeat the last token.
    logits = output.get("logits")
    encoder_decoder = getattr(config, "is_encoder_decoder", False)
    if logits is not None and logits.dim() == 3:
        token = logits[:, -1:].argmax(-1)
    else:
        token = inputs.get("decoder_input_ids" if encoder_decoder else "input_ids")
        if token is None:
            return None
        token = token[:, -1:]
    if encoder_decoder:
        if "decoder_input_ids" not in inputs:
            return None
        step = {k: v for k, v in inputs.items() if not k.startswith("decoder_")}
        step["decoder_input_ids"] = token
    else:
        if "input_ids" not in inputs:
            return None
        step = {k: v for k, v in inputs.items() if k in STEP_KEEP and (keep_mask or k != "attention_mask")}
        step["input_ids"] = token
        if "attention_mask" in step:
            mask = step["attention_mask"]
            step["attention_mask"] = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=-1)
    step[cache_key] = output[cache_key]
    step["use_cache"] = True
    return step


def _cache_key(output):
    if isinstance(output, ModelOutput):
        return next((key for key in CACHE_KEYS if output.get(key) is not None), None)
    return None


def _build_pair(native_class, vendored_class, native_config, vendored_config, attn_implementation=None):
    torch.manual_seed(0)
    native = native_class(native_config).eval()
    torch.manual_seed(0)
    vendored = vendored_class(vendored_config).eval()
    vendored.load_state_dict(native.state_dict(), strict=True)
    for key, buffer in native.named_buffers():
        vendored.get_buffer(key).copy_(buffer)
    if attn_implementation is not None:
        for model in (native, vendored):
            try:
                model.set_attn_implementation(attn_implementation)
            except (ValueError, ImportError, AttributeError):
                pass
    return native, vendored


def assert_variants_match(native, vendored, name, inputs, native_bugs=frozenset()):
    """Run every call variant on an identically initialized (eager) native/vendored pair.

    Returns ({variant label: reason} for variants that do not apply or that Transformers itself
    does not support, the mismatches, the compared labels, the labels native raised on).
    """

    check = _Checker(native_bugs)
    config = native.config
    for variant, kwargs in FORWARD_VARIANTS.items():
        check.compare(native, vendored, "forward", {**inputs, **kwargs}, f"{name}/{variant}")

    # Cached decode: prefill with use_cache=True, then one cached step fed the next token. The step
    # keeps the (extended) attention mask unless native rejects that (M-RoPE families derive
    # position ids from a full-length mask; generation slices them in prepare_inputs_for_generation),
    # in which case the step runs without a mask. Each attempt starts from a fresh prefill so the
    # caches are never reused after a failed step.
    can_generate = native.can_generate()
    if can_generate or any(_accepts(type(native), key) for key in CACHE_KEYS):
        expected, _ = check.compare(native, vendored, "forward", {**inputs, "use_cache": True}, f"{name}/use_cache")
        cache_key = _cache_key(expected)
        if expected is not None and cache_key is None:
            check.unsupported[f"{name}/cached_step"] = "native returns no cache with use_cache=True"
        elif expected is not None:
            def prefill(model):
                torch.manual_seed(0)
                with torch.no_grad():
                    return model(**inputs, use_cache=True)

            keep_mask = True
            if "attention_mask" in inputs and not getattr(config, "is_encoder_decoder", False):
                probe = _next_steps(inputs, prefill(native), config, cache_key)
                keep_mask = probe is None or _run(native, "forward", probe)[1] is None
            native_step = _next_steps(inputs, prefill(native), config, cache_key, keep_mask)
            if native_step is None:
                check.unsupported[f"{name}/cached_step"] = "no text input to continue"
            else:
                vendored_step = _next_steps(inputs, prefill(vendored), vendored.config, cache_key, keep_mask)
                label = f"{name}/cached_step" if keep_mask else f"{name}/cached_step_without_mask"
                check.compare(native, vendored, "forward", native_step, label, vendored_step)

    labels = _label_kwargs(name, type(native), config, inputs)
    if labels is None:
        check.unsupported[f"{name}/loss"] = "forward takes no labels or no label-shaped input"
    else:
        expected, _ = check.compare(native, vendored, "forward", {**inputs, **labels}, f"{name}/loss")
        if isinstance(expected, ModelOutput) and expected.get("loss") is None:
            check.unsupported[f"{name}/loss"] = "native returns no loss"

    if can_generate:
        check.compare(native, vendored, "generate", {**inputs, "max_new_tokens": 3, "do_sample": False}, f"{name}/generate")
    return check.unsupported, check.failures, check.compared, check.raised


def _record_report(family, unsupported, failures, compared):
    report = os.environ.get("GPTQMODEL_PARITY_VARIANT_REPORT")
    if report:
        with open(report, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"family": family, "unsupported": unsupported, "failures": failures, "compared": compared}, sort_keys=True) + "\n")


@pytest.mark.parametrize("family", CASES)
def test_vendored_family_call_variants_match_transformers(family):
    if family in ORACLE_UNAVAILABLE:
        pytest.importorskip("scipy", reason=ORACLE_UNAVAILABLE[family])
    spec = json.loads((FIXTURES / f"{family}.json").read_text())
    native_config_class, vendored_config_class = _classes(family, spec["config_module"], spec["config_class"])
    family_inputs = _inputs(family, spec)
    unsupported, failures, compared, raised = {}, [], [], []
    for module, name in spec["model_classes"]:
        inputs = _class_inputs(name, family_inputs)
        native_class, vendored_class = _classes(family, module, name)
        native, vendored = _build_pair(
            native_class, vendored_class,
            native_config_class.from_dict(dict(spec["config"])), vendored_config_class.from_dict(dict(spec["config"])),
            attn_implementation="eager",
        )
        class_unsupported, class_failures, class_compared, class_raised = assert_variants_match(native, vendored, name, inputs)
        raised.extend(class_raised)
        unsupported.update(class_unsupported)
        failures.extend(class_failures)
        compared.extend(class_compared)
    _record_report(family, unsupported, failures, compared)
    assert not failures, "\n".join(failures)
    assert sorted(raised) == sorted(NATIVE_UNSUPPORTED.get(family, {})), {label: unsupported[label] for label in raised}


def _assert_hand_variants(family, native, vendored, model_name, inputs):
    unsupported, failures, compared, raised = assert_variants_match(
        native, vendored, model_name, inputs, NATIVE_BUGS.get(family, frozenset())
    )
    _record_report(family, unsupported, failures, compared)
    assert not failures, "\n".join(failures)
    assert sorted(raised) == sorted(label for label in NATIVE_UNSUPPORTED.get(family, {}) if label.startswith(f"{model_name}/")), {
        label: unsupported[label] for label in raised
    }
    assert f"{model_name}/default" in compared


def _assert_hand_family_parity(family, config_name, config_kwargs, model_name, inputs, tmp_path):
    """Hand-written tiny-config parity for families without a usable Transformers model tester."""

    native_config_class, vendored_config_class = _classes(family, f"configuration_{family}", config_name)
    assert vendored_config_class is not native_config_class
    native_class, vendored_class = _classes(family, f"modeling_{family}", model_name)
    native, vendored = _build_pair(
        native_class, vendored_class,
        native_config_class(**config_kwargs, attn_implementation="eager"),
        vendored_config_class(**config_kwargs, attn_implementation="eager"),
    )
    assert list(vendored.state_dict()) == list(native.state_dict())
    _assert_hand_variants(family, native, vendored, model_name, inputs)

    # A Transformers-saved checkpoint must load into the vendored class with identical outputs.
    checkpoint = tmp_path / model_name
    native.save_pretrained(checkpoint)
    native_loaded = native_class.from_pretrained(checkpoint).eval()
    loaded = vendored_class.from_pretrained(checkpoint, config=vendored_config_class.from_pretrained(checkpoint)).eval()
    assert type(loaded) is vendored_class
    with torch.no_grad():
        torch.manual_seed(0)
        expected = native_loaded(**inputs)
        torch.manual_seed(0)
        actual = loaded(**inputs)
    _assert_same(expected, actual, f"{model_name}/reloaded")


def _text_inputs(batch=2, length=9, eos=None, vocab=90, seed=0):
    """Token ids in [3, vocab) (0/1/2 are pad/bos/eos), optionally eos-terminated, row 1 left-padded."""

    input_ids = torch.randint(3, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))
    if eos is not None:
        input_ids[:, -1] = eos
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 0] = 0
    return input_ids, attention_mask


# BART and OPT: the Transformers 5.23 testers exist but cannot be imported here (they need the
# `timeout_decorator` test dependency), so these tiny configs are hand-written.
BART_CONFIG = {
    "vocab_size": 99, "d_model": 16, "encoder_layers": 2, "decoder_layers": 2, "encoder_attention_heads": 4,
    "decoder_attention_heads": 4, "encoder_ffn_dim": 32, "decoder_ffn_dim": 32, "max_position_embeddings": 64,
    "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2, "decoder_start_token_id": 2,
    "forced_eos_token_id": None,
}


def _bart_inputs(model_name):
    # BartForSequenceClassification pools the final <eos>; every row must contain the same number.
    input_ids, attention_mask = _text_inputs(eos=BART_CONFIG["eos_token_id"])
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if model_name in {"BartModel", "BartForConditionalGeneration"}:
        decoder_input_ids, _ = _text_inputs(length=5, seed=1)
        decoder_input_ids[:, 0] = BART_CONFIG["decoder_start_token_id"]
        inputs["decoder_input_ids"] = decoder_input_ids
    return inputs


@pytest.mark.parametrize("model_name", [
    "BartModel", "BartForConditionalGeneration", "BartForCausalLM", "BartForSequenceClassification",
    "BartForQuestionAnswering",
])
def test_vendored_bart_matches_transformers(model_name, tmp_path):
    config = dict(BART_CONFIG, is_decoder=True, is_encoder_decoder=False) if model_name == "BartForCausalLM" else BART_CONFIG
    _assert_hand_family_parity("bart", "BartConfig", config, model_name, _bart_inputs(model_name), tmp_path)


OPT_CONFIG = {
    "vocab_size": 99, "hidden_size": 16, "num_hidden_layers": 2, "ffn_dim": 32, "num_attention_heads": 4,
    "max_position_embeddings": 64, "word_embed_proj_dim": 16, "pad_token_id": 1, "bos_token_id": 2,
    "eos_token_id": 2,
}


@pytest.mark.parametrize("model_name", ["OPTModel", "OPTForCausalLM", "OPTForSequenceClassification", "OPTForQuestionAnswering"])
@pytest.mark.parametrize("projection", [False, True], ids=["plain", "project_in_out"])
def test_vendored_opt_matches_transformers(model_name, projection, tmp_path):
    # word_embed_proj_dim != hidden_size exercises project_in/project_out (OPT-350m style).
    config = dict(OPT_CONFIG, word_embed_proj_dim=8) if projection else OPT_CONFIG
    input_ids, attention_mask = _text_inputs()
    _assert_hand_family_parity(
        "opt", "OPTConfig", config, model_name, {"input_ids": input_ids, "attention_mask": attention_mask}, tmp_path
    )
