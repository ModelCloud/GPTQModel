# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import torch
import transformers

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import QVQConfig


def _prepared_calibration(**kwargs):
    return kwargs["calibration_dataset"]


def test_qvq_conv1d_lifecycle_preserves_orientation_and_serialized_output():
    torch.manual_seed(20260812)
    root = torch.nn.Module()
    root.proj = transformers.Conv1D(16, 16)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=QVQConfig(bits=2, device="cpu", offload_to_disk=False),
        calibration=[
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }
        ],
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    processor.preprocess(named)

    source = torch.randn((1, 4, 16), dtype=torch.float32)
    output = root.proj(source)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 4), dtype=torch.bool)
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), output)
    processor.process(named, device=torch.device("cpu"))
    dense_replay = root.proj(source)

    qmodule = processor.submodule_finalize(named, SimpleNamespace(model=root))
    live_output = qmodule(source)
    torch.testing.assert_close(live_output, dense_replay, rtol=1e-5, atol=1e-5)

    checkpoint_tensors = {name: tensor.clone() for name, tensor in qmodule.state_dict().items()}
    reloaded = QVQLinear(
        bits=2,
        in_features=16,
        out_features=16,
        bias=True,
        dtype=torch.float32,
        tensors=checkpoint_tensors,
    )
    assert set(checkpoint_tensors) == {"trellis", "SU", "SV", "bias"}
    torch.testing.assert_close(reloaded(source), live_output, rtol=0, atol=0)
