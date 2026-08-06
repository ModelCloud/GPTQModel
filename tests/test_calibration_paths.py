# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import tempfile
import unittest
from types import SimpleNamespace

from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import META_FIELD_CALIBRATION_PATHS
from gptqmodel.utils.calibration import _extract_calibration_paths, _record_calibration_source


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


if __name__ == "__main__":
    unittest.main()
