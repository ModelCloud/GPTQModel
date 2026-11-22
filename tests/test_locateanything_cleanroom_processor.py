# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Black-box LocateAnything processor checks against tests/fixtures/cleanroom/locateanything/processor."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import TokenizersBackend

from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.ports.locateanything import processing_locateanything as proc_mod
from gptqmodel.models.ports.locateanything.image_processing_locateanything import LocateAnythingImageProcessor
from gptqmodel.models.ports.locateanything.processing_locateanything import LocateAnythingProcessor
from gptqmodel.utils.image import ImageFetchError


FIXTURES = Path(__file__).parent / "fixtures/cleanroom/locateanything/processor"
META = json.loads((FIXTURES / "processor_expected.json").read_text())
TENSORS = load_file(str(FIXTURES / "processor_expected.safetensors"))


def _image(name):
    return Image.fromarray(TENSORS[f"img.{name}"].numpy())


def _processor():
    tokenizer = TokenizersBackend.from_pretrained(str(FIXTURES))
    return LocateAnythingProcessor(LocateAnythingImageProcessor(patch_size=2), tokenizer)


def _chat_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _image("c_rgb_w8_h4")},
                {"type": "video", "video": [_image("b_rgb_w20_h12"), _image("c_rgb_w8_h4")]},
                {"type": "text", "text": "Find 5 and 6"},
            ],
        },
        {"role": "assistant", "content": "7 8"},
    ]


@pytest.mark.parametrize("case", sorted(META["image_processor"]))
def test_image_processor_matches_fixture_exactly(case):
    spec = META["image_processor"][case]
    image = _image(spec["input"])
    if spec["applied_to_rgb_first"]:
        image = proc_mod.fetch_image({"image": image})
    out = LocateAnythingImageProcessor(**spec["kwargs"])(image, return_tensors="pt")
    assert out["image_grid_hws"].tolist() == spec["grid_hws"]
    assert torch.equal(out["image_grid_hws"], TENSORS[f"iproc.{case}.image_grid_hws"])
    assert torch.equal(out["pixel_values"], TENSORS[f"iproc.{case}.pixel_values"])


def test_image_processor_rejects_non_images_and_saves_its_keys(tmp_path):
    with pytest.raises(ValueError, match="Invalid image type"):
        LocateAnythingImageProcessor()(np.zeros((4, 4, 3), dtype=np.uint8))
    LocateAnythingImageProcessor(patch_size=2).save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "preprocessor_config.json").read_text())
    assert saved == {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "in_token_limit": 4096,
        "merge_kernel_size": [2, 2],
        "patch_size": 2,
        "image_processor_type": "LocateAnythingImageProcessor",
    }


def test_image_processor_rejects_oversized_grid():
    with pytest.raises(ValueError, match="Exceed pos emb"):
        LocateAnythingImageProcessor(patch_size=1, in_token_limit=10**7)(Image.new("RGB", (512, 4)))


def test_chat_template_and_chat_case_match_fixture():
    processor = _processor()
    case = META["chat_case"]
    messages = _chat_messages()
    text = processor.py_apply_chat_template(messages, add_generation_prompt=True)
    assert text == case["py_apply_chat_template(add_generation_prompt=True)"]

    images, videos = processor.process_vision_info(messages)
    assert len(images) == 1 and len(videos) == 1 and len(videos[0]) == 2
    expanded = processor.replace_media_placeholder(text, images, videos, None, None, images_kwargs={}, videos_kwargs={})
    assert expanded[0] == case["expanded_text"]
    assert expanded[3:] == (1, 1)

    out = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    assert out["input_ids"].tolist() == case["input_ids"]
    assert out["attention_mask"].tolist() == case["attention_mask"]
    assert out["image_grid_hws"].tolist() == case["image_grid_hws"]
    assert torch.equal(out["pixel_values"], TENSORS["chat_case.pixel_values"])


def test_inline_placeholder_is_not_duplicated():
    messages = [
        {"role": "system", "content": "sys 9"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "unused"},
                {"type": "text", "text": "look <image-1> here"},
            ],
        },
    ]
    rendered = _processor().py_apply_chat_template(messages, add_generation_prompt=False)
    assert rendered == META["chat_inline_placeholder"]["py_apply_chat_template(add_generation_prompt=False)"]


def test_video_tensor_expansion_with_timestamps_and_fps():
    case = META["video_tensor_case"]
    video = TENSORS["video_tensor.input"]
    text, pixel_values, grids, num_images, num_videos = _processor().replace_media_placeholder(
        "<video-1> and <image-1>", [_image("c_rgb_w8_h4")], [video], [[0.0, 0.5]], [2.0],
        images_kwargs={}, videos_kwargs={},
    )
    assert text == case["expanded_text"]
    assert grids.tolist() == case["image_grid_hws"]
    assert (num_images, num_videos) == (case["num_images"], case["num_videos"])
    assert torch.equal(pixel_values, TENSORS["video_tensor.pixel_values"])


def test_unit_range_video_tensor_expansion():
    case = META["video_tensor01_case"]
    video = TENSORS["video_tensor.input"] / 255.0
    text, pixel_values, grids, _, _ = _processor().replace_media_placeholder(
        "<video-1>", [], [video], None, None, images_kwargs={}, videos_kwargs={},
    )
    assert text == case["expanded_text"]
    assert grids.tolist() == case["image_grid_hws"]
    assert torch.equal(pixel_values, TENSORS["video_tensor01.pixel_values"])


def test_two_sample_batch_consumes_media_in_order():
    case = META["batch_two_samples"]
    out = _processor()(
        text=["a <image-1> b", "c <image-1> d"],
        images=[_image("c_rgb_w8_h4"), _image("b_rgb_w20_h12")],
        return_tensors="pt",
        padding=True,
    )
    assert out["input_ids"].tolist() == case["input_ids"]
    assert out["attention_mask"].tolist() == case["attention_mask"]
    assert out["image_grid_hws"].tolist() == case["image_grid_hws"]
    assert torch.equal(out["pixel_values"], TENSORS["batch_two_samples.pixel_values"])


@pytest.mark.parametrize("index", range(len(META["get_video_frame_indices"])))
def test_get_video_frame_indices(index):
    case = META["get_video_frame_indices"][index]
    idx, sample_fps = proc_mod.get_video_frame_indices(case["ele"], case["total_frames"], case["video_fps"])
    assert list(idx) == case["idx"]
    assert sample_fps == pytest.approx(case["sample_fps"], rel=1e-12)


def test_spec_two_frame_list_video_case():
    """Image plus a two-frame list video on 4x4 frames gives three 2x2 grids (SPEC §12)."""
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 256, (4, 4, 3), dtype=np.uint8) for _ in range(3)]
    frames = [Image.fromarray(array) for array in arrays]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": frames[0]},
        {"type": "video", "video": frames[1:]},
        {"type": "text", "text": "where"},
    ]}]
    processor = _processor()
    text = processor.py_apply_chat_template(messages, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    out = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    assert tuple(out["pixel_values"].shape) == (12, 3, 2, 2)
    assert out["image_grid_hws"].tolist() == [[2, 2]] * 3
    assert out["pixel_values"][0, 0, 0, 0].item() == pytest.approx(arrays[0][0, 0, 0] / 127.5 - 1, abs=1e-6)
    expanded = processor.replace_media_placeholder(text, images, videos, None, None, images_kwargs={}, videos_kwargs={})[0]
    assert expanded.count("<IMG_CONTEXT>") == 3
    assert "Frame-1:" in expanded and "Frame-2:" in expanded


def test_missing_image_and_invalid_text_raise():
    processor = _processor()
    with pytest.raises(IndexError):
        processor(text=["<image-2>"], images=[_image("c_rgb_w8_h4")])
    with pytest.raises(ValueError, match="Invalid input text"):
        processor(text=[])
    with pytest.raises(ValueError, match="Invalid input text"):
        processor(text=3)


def test_save_and_reload_round_trip(tmp_path):
    processor = _processor()
    with pytest.raises(ValueError):
        (tmp_path / "file").write_text("x")
        processor.save_pretrained(tmp_path / "file")
    target = tmp_path / "saved"
    processor.save_pretrained(target)
    saved = json.loads((target / "processor_config.json").read_text())
    for key in ("image_end_token", "image_placeholder", "image_start_token", "image_token",
                "merge_kernel_size", "processor_class", "video_placeholder", "video_token", "image_processor"):
        assert key in saved, key
    assert (target / "preprocessor_config.json").is_file()

    reloaded = LocateAnythingProcessor.from_pretrained(target, tokenizer=processor.tokenizer)
    assert type(reloaded) is LocateAnythingProcessor
    assert type(reloaded.image_processor) is LocateAnythingImageProcessor
    assert reloaded.image_processor.to_dict() == processor.image_processor.to_dict()
    text = processor.py_apply_chat_template(_chat_messages(), add_generation_prompt=True)
    images, videos = processor.process_vision_info(_chat_messages())
    first = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    second = reloaded(text=[text], images=images, videos=videos, return_tensors="pt")
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert reloaded.model_input_names == list(dict.fromkeys(
        processor.tokenizer.model_input_names + ["pixel_values", "image_grid_hws"]
    ))


def test_from_pretrained_ignores_hostile_auto_map(tmp_path, monkeypatch):
    processor = _processor()
    processor.save_pretrained(tmp_path)
    hostile = {"AutoProcessor": "evil.Processor", "AutoTokenizer": ["evil.Tok", None],
               "AutoImageProcessor": "evil.ImageProcessor"}
    for name in ("processor_config.json", "tokenizer_config.json", "preprocessor_config.json"):
        payload = json.loads((tmp_path / name).read_text())
        payload["auto_map"] = hostile
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "evil.py").write_text("raise SystemExit('checkpoint code ran')\n")
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *_a, **_k: pytest.fail("processor loading imported checkpoint Python"),
    )
    config_path = Path(__file__).parent / "fixtures/cleanroom/locateanything/qwen2_gqa/config.json"
    (tmp_path / "config.json").write_text(config_path.read_text())

    reloaded = LocateAnythingQModel.loader.load_processor(str(tmp_path), trust_remote_code=True)
    assert type(reloaded) is LocateAnythingProcessor
    assert type(reloaded.image_processor) is LocateAnythingImageProcessor
    assert reloaded.image_token_id == 59
    assert reloaded(text=["x"])["input_ids"] == processor(text=["x"])["input_ids"]


def test_remote_image_uses_bounded_module_fetch(monkeypatch):
    calls = []

    def fake_load(url, *, max_bytes):
        calls.append((url, max_bytes))
        return Image.new("RGBA", (4, 4), (0, 0, 0, 0))

    monkeypatch.setattr(proc_mod, "load_image_from_url", fake_load)
    image = proc_mod.fetch_image({"image_url": "https://example.invalid/a.png"})
    assert calls == [("https://example.invalid/a.png", proc_mod.REMOTE_MEDIA_MAX_BYTES)]
    assert proc_mod.REMOTE_MEDIA_MAX_BYTES == 64 * 1024 * 1024
    assert image.mode == "RGB" and image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("url", ["http://127.0.0.1/a.png", "http://10.0.0.1/a.png", "http://[::1]/a.png"])
def test_remote_image_rejects_loopback_and_private_hosts(url):
    with pytest.raises(ImageFetchError):
        proc_mod.fetch_image({"image": url})


def test_remote_http_error_propagates(monkeypatch):
    def not_found(url, *, max_bytes):
        raise ImageFetchError(url, "HTTP 404")

    monkeypatch.setattr(proc_mod, "load_image_from_url", not_found)
    monkeypatch.setattr(proc_mod, "load_bytes_from_url", not_found)
    with pytest.raises(ImageFetchError, match="404"):
        proc_mod.fetch_image({"image": "https://example.invalid/missing.png"})
    with pytest.raises(ImageFetchError, match="404"):
        proc_mod.fetch_video({"video": "https://example.invalid/missing.mp4"})


def test_remote_video_downloads_to_suffixed_temp_file(monkeypatch):
    calls = {}

    def fake_bytes(url, *, max_bytes):
        calls["bytes"] = (url, max_bytes)
        return b"not-really-a-video"

    def fake_reader(ele):
        path = ele["video"]
        calls["path"] = path
        assert Path(path).read_bytes() == b"not-really-a-video"
        return torch.full((3, 3, 4, 4), 7, dtype=torch.uint8), 1.5, [0.0, 0.5, 1.0]

    monkeypatch.setattr(proc_mod, "load_bytes_from_url", fake_bytes)
    monkeypatch.setitem(proc_mod.VIDEO_READER_BACKENDS, "torchvision", fake_reader)
    video, fps, timestamps = proc_mod.fetch_video(
        {"video": "https://example.invalid/clip.mp4?sig=1"}, return_video_sample_fps=True,
    )
    assert calls["bytes"] == ("https://example.invalid/clip.mp4?sig=1", proc_mod.REMOTE_MEDIA_MAX_BYTES)
    assert calls["path"].endswith(".mp4")
    assert not os.path.exists(calls["path"])
    assert video.dtype is torch.float32 and tuple(video.shape) == (3, 3, 4, 4)
    assert (fps, timestamps) == (1.5, [0.0, 0.5, 1.0])


def test_video_reader_failure_falls_back_to_torchvision_and_caps_pixels(monkeypatch):
    def broken(ele):
        raise RuntimeError("decoder missing")

    def reader(ele):
        return torch.full((2, 3, 8, 8), 100, dtype=torch.uint8), 2.0, [0.0, 0.5]

    monkeypatch.setitem(proc_mod.VIDEO_READER_BACKENDS, "decord", broken)
    monkeypatch.setitem(proc_mod.VIDEO_READER_BACKENDS, "torchvision", reader)
    video = proc_mod.fetch_video({"video": "file:///tmp/x.mp4", "video_total_pixels": 32}, video_reader_backend="decord")
    assert tuple(video.shape) == (2, 3, 4, 4)
    assert video.dtype is torch.float32
