# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from model_test import ModelTest

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.utils.torch import torch_empty_cache


# Only quantize the first four decoder blocks so the memory regression stays fast enough for CI.
FIRST4_ONLY_NEGATIVE_MATCH = r"^model\.layers\.(?!(?:0|1|2|3)\.)\d+\."
# The Llama 3.2 checkpoint must stay monolithic to reproduce the original mmap-retention risk.
MONOLITHIC_SAFETENSORS_FILE = "model.safetensors"
# Finalized layers should stay within a small RSS band once prior source weights have been released.
MAX_FINALIZED_RSS_GROWTH_GIB = 0.2


@dataclass(frozen=True)
class LayerMemoryRecord:
    """Capture host-memory state at a finalized layer boundary."""

    layer_idx: int
    rss_gib: float
    mmaps: int
    cuda_alloc_gib: float
    cuda_reserved_gib: float


class TestLlama3_2LazyTurtleMemory(ModelTest):
    """Verify lazy turtle keeps host memory bounded on a monolithic Llama safetensors file."""

    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    TORCH_DTYPE = "bfloat16"
    USE_FLASH_ATTN = False
    QUANT_BATCH_SIZE = 1
    DATASET_SIZE = 32
    DATASET_CONCAT_SIZE = 2048
    OFFLOAD_TO_DISK = True
    DYNAMIC = {
        f"-:{FIRST4_ONLY_NEGATIVE_MATCH}": {},
    }

    @staticmethod
    def _read_rss_gib() -> float:
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
        raise RuntimeError("VmRSS entry missing from /proc/self/status")

    @staticmethod
    def _count_file_mmaps(path: Path) -> int:
        resolved = str(path.resolve())
        with Path("/proc/self/maps").open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if resolved in line)

    @staticmethod
    def _expected_layer_indices() -> list[int]:
        return [0, 1, 2, 3]

    def _assert_monolithic_checkpoint_layout(self, checkpoint_dir: Path) -> Path:
        safetensor_files = sorted(path.name for path in checkpoint_dir.glob("*.safetensors"))
        self.assertEqual(
            safetensor_files,
            [MONOLITHIC_SAFETENSORS_FILE],
            f"Expected a single top-level safetensors file under {checkpoint_dir}.",
        )

        checkpoint_path = checkpoint_dir / MONOLITHIC_SAFETENSORS_FILE
        self.assertTrue(checkpoint_path.is_file(), f"Missing monolithic checkpoint file at {checkpoint_path}.")
        self.assertFalse(
            (checkpoint_dir / "model.safetensors.index.json").exists(),
            "Monolithic checkpoint regression should not use a shard index.",
        )
        return checkpoint_path

    def _build_layer_probe(self, checkpoint_path: Path):
        records: list[LayerMemoryRecord] = []

        class _Probe:
            """Record finalized-layer memory and stop once the first four layers have completed."""

            def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
                if not submodule_finalized:
                    return None

                gc.collect()
                torch.cuda.synchronize()
                records.append(
                    LayerMemoryRecord(
                        layer_idx=layer_idx,
                        rss_gib=self_owner._read_rss_gib(),
                        mmaps=self_owner._count_file_mmaps(checkpoint_path),
                        cuda_alloc_gib=torch.cuda.memory_allocated(0) / (1024 ** 3),
                        cuda_reserved_gib=torch.cuda.memory_reserved(0) / (1024 ** 3),
                    )
                )
                if layer_idx >= self_owner._expected_layer_indices()[-1]:
                    raise StopMainLoop

        self_owner = self
        return _Probe(), records

    def _assert_memory_records(self, records: list[LayerMemoryRecord]) -> None:
        self.assertEqual(
            [record.layer_idx for record in records],
            self._expected_layer_indices(),
            f"Expected finalized-layer records for layers {self._expected_layer_indices()}, got {records}.",
        )
        self.assertTrue(records, "Expected at least one finalized-layer memory record.")
        self.assertTrue(
            all(record.mmaps == 0 for record in records),
            f"Monolithic safetensors mmap should be released after each finalized layer, got {records}.",
        )

        baseline_rss = records[0].rss_gib
        later_peak_rss = max(record.rss_gib for record in records[1:])
        self.assertLessEqual(
            later_peak_rss,
            baseline_rss + MAX_FINALIZED_RSS_GROWTH_GIB,
            f"Finalized host RSS kept growing instead of flattening: {records}.",
        )
        self.assertLessEqual(
            records[-1].rss_gib,
            baseline_rss + MAX_FINALIZED_RSS_GROWTH_GIB,
            f"Last finalized layer retained too much host memory: {records}.",
        )

    def test_lazy_turtle_releases_monolithic_checkpoint_memory_between_layers(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the lazy-turtle memory regression test.")
        if not Path("/proc/self/maps").exists():
            self.skipTest("/proc/self/maps is required to inspect live safetensors mmaps.")

        checkpoint_dir = Path(self.NATIVE_MODEL_ID)
        checkpoint_path = self._assert_monolithic_checkpoint_layout(checkpoint_dir)

        model = None
        dataset = None
        try:
            quantize_config = self._build_quantize_config()
            quantize_config.device = torch.device("cuda")
            quantize_config.wait_for_submodule_finalizers = True

            model = GPTQModel.load(
                self.NATIVE_MODEL_ID,
                quantize_config=quantize_config,
                dtype=self.TORCH_DTYPE,
                attn_implementation="eager",
            )

            probe, records = self._build_layer_probe(checkpoint_path)
            model.layer_callback = probe
            dataset = self.load_dataset(model.tokenizer, rows=self.DATASET_SIZE)

            try:
                model.quantize(
                    dataset,
                    calibration_concat_size=self.DATASET_CONCAT_SIZE,
                    calibration_sort=self.DATASET_SORT,
                    batch_size=self.QUANT_BATCH_SIZE,
                )
            except StopMainLoop:
                # The layer callback raises this sentinel once layers 0-3 have
                # produced the finalized memory samples this regression needs.
                pass

            self._assert_memory_records(records)
        finally:
            del dataset
            del model
            torch_empty_cache()
