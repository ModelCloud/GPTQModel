# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import tempfile
import unittest
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import META_FIELD_CALIBRATION_PATHS
from gptqmodel.utils.calibration import (
    _extract_calibration_paths,
    _record_calibration_source,
    prepare_calibration_dataset,
)


class TestCalibrationPaths(unittest.TestCase):
    def test_extract_ignores_plain_text_with_separators(self):
        samples = [
            "plain text sample that is not a path",
            "and/or",
            "10/12/2020",
            "C:\\Windows\\data.json",
            "relative.csv",
            "/abs/path/to/data.txt",
        ]
        self.assertEqual(_extract_calibration_paths(samples), [])

    def test_extract_keeps_uris_and_basenames_of_existing_files(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            basename = os.path.basename(path)
            samples = [
                path,
                "https://huggingface.co/datasets/foo",
                "s3://bucket/data.jsonl",
                "not-a-file",
                "hello world",
            ]
            self.assertEqual(
                _extract_calibration_paths(samples),
                [basename, "https://huggingface.co/datasets/foo", "s3://bucket/data.jsonl"],
            )
            self.assertEqual(_extract_calibration_paths(path), [basename])
        finally:
            os.unlink(path)

    def test_extract_redacts_uri_credentials_query_and_fragment(self):
        cases = (
            (
                "https://user:secret@example.com:8443/data/train.jsonl?token=signed-secret#private",
                "https://example.com:8443/data/train.jsonl",
            ),
            (
                "s3://access:secret@bucket/data.jsonl?signature=signed-secret#private",
                "s3://bucket/data.jsonl",
            ),
            (
                "https://user:secret@[2001:db8::1]:8443/data.jsonl?token=signed-secret",
                "https://[2001:db8::1]:8443/data.jsonl",
            ),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_extract_calibration_paths(value), [expected])

        dataset = SimpleNamespace(
            cache_files=[],
            info=SimpleNamespace(
                dataset_name="https://user:secret@example.com/private.json?token=signed-secret",
                builder_name=None,
            ),
        )
        self.assertEqual(
            _extract_calibration_paths(dataset),
            ["https://example.com/private.json"],
        )

        duplicate_dataset = SimpleNamespace(
            cache_files=[],
            info=SimpleNamespace(
                dataset_name="https://first:secret@example.com/private.json?token=first",
                builder_name="https://second:secret@example.com/private.json?token=second",
            ),
        )
        self.assertEqual(
            _extract_calibration_paths(duplicate_dataset),
            ["https://example.com/private.json"],
        )

        malformed_uri_dataset = SimpleNamespace(
            cache_files=[],
            info=SimpleNamespace(
                dataset_name="https://user:secret@[invalid-host/private.json",
                builder_name=None,
            ),
        )
        self.assertEqual(_extract_calibration_paths(malformed_uri_dataset), [])

        whitespace_uri_dataset = SimpleNamespace(
            cache_files=[],
            info=SimpleNamespace(
                dataset_name="https://user:sec ret@example.com/private.json",
                builder_name=None,
            ),
        )
        self.assertEqual(_extract_calibration_paths(whitespace_uri_dataset), [])

    def test_extract_redacts_or_rejects_cache_file_uris(self):
        dataset = SimpleNamespace(
            cache_files=[
                {"filename": "https://user:secret@example.com/data.arrow?token=signed-secret#private"},
                {"filename": "https://user:secret@[invalid-host/data.arrow?token=signed-secret"},
            ],
            info=None,
        )
        self.assertEqual(_extract_calibration_paths(dataset), ["https://example.com/data.arrow"])

    def test_extract_rejects_incomplete_or_invalid_uris(self):
        cases = (
            "https:///user:secret@example.com/private.json",
            "https://user:secret@example.com:not-a-port/private.json",
            "https://user:secret@[invalid-host/private.json",
            "https://user:sec ret@example.com/private.json",
        )
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(_extract_calibration_paths(value), [])

    def test_extract_existing_directory_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_extract_calibration_paths(tmp), [os.path.basename(tmp)])

    def test_record_calibration_appends_unique_paths(self):
        qcfg = QuantizeConfig(bits=4, group_size=128)
        qmodel = SimpleNamespace(quantize_config=qcfg)

        _record_calibration_source(qmodel, ["https://example.com/first.json", "https://example.com/second.json"])
        self.assertEqual(
            qcfg.meta[META_FIELD_CALIBRATION_PATHS],
            ["https://example.com/first.json", "https://example.com/second.json"],
        )

        _record_calibration_source(qmodel, ["https://example.com/second.json", "https://example.com/third.json"])
        self.assertEqual(
            qcfg.meta[META_FIELD_CALIBRATION_PATHS],
            [
                "https://example.com/first.json",
                "https://example.com/second.json",
                "https://example.com/third.json",
            ],
        )

    def test_extract_hf_dataset_cache_basenames_and_info_name(self):
        fake_cache = [{"filename": "/cache/data-00000-of-00001.arrow"}, "/cache/data-00001-of-00001.arrow"]
        fake_info = SimpleNamespace(dataset_name="foo/bar", builder_name=None)
        dataset = SimpleNamespace(cache_files=fake_cache, info=fake_info)
        self.assertEqual(
            _extract_calibration_paths(dataset),
            [
                "data-00000-of-00001.arrow",
                "data-00001-of-00001.arrow",
                "foo/bar",
            ],
        )

    def test_extract_hf_dataset_falls_back_to_info_name(self):
        fake_info = SimpleNamespace(dataset_name="HuggingFaceFW/fineweb", builder_name=None)
        dataset = SimpleNamespace(cache_files=[], info=fake_info)
        self.assertEqual(_extract_calibration_paths(dataset), ["HuggingFaceFW/fineweb"])

    def test_record_calibration_ignores_non_path_strings(self):
        qcfg = QuantizeConfig(bits=4, group_size=128)
        qmodel = SimpleNamespace(quantize_config=qcfg)

        _record_calibration_source(qmodel, ["hello world", "and/or", "10/12/2020"])
        self.assertNotIn(META_FIELD_CALIBRATION_PATHS, qcfg.meta)


def test_prepare_calibration_preserves_source_weights_without_duplicating_rows():
    qmodel = SimpleNamespace(
        tokenizer=None,
        support_batch_quantize=True,
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32)),
    )
    rows = [
        {"input_ids": list(range(12)), "source_name": "yaqa"},
        {"input_ids": list(range(13)), "source_name": "nm"},
    ]

    batches = prepare_calibration_dataset(
        qmodel,
        rows,
        batch_size=2,
        source_weight_column="source_name",
        source_weights=(("yaqa", 2.0), ("nm", 1.0)),
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 2
    torch.testing.assert_close(
        batches[0]["fisher_sequence_weight"],
        torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


def test_prepare_calibration_source_weights_fail_closed_on_unmapped_source():
    qmodel = SimpleNamespace(
        tokenizer=None,
        support_batch_quantize=True,
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32)),
    )
    with pytest.raises(ValueError, match="unmapped source"):
        prepare_calibration_dataset(
            qmodel,
            [{"input_ids": list(range(12)), "source_name": "other"}],
            source_weight_column="source_name",
            source_weights=(("yaqa", 2.0), ("nm", 1.0)),
        )


if __name__ == "__main__":
    unittest.main()
