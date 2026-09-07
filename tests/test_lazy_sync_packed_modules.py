# SPDX-License-Identifier: Apache-2.0
import threading
from types import SimpleNamespace

import torch

from gptqmodel.utils.structure import LazyTurtle


def test_lazy_sync_excludes_packed_modules_from_dense_source():
    class Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty(32, 1, dtype=torch.uint8, device="meta"))

    model = torch.nn.Module()
    model.packed = Packed()
    model.norm = torch.nn.Linear(4, 4)
    seen = []

    def materialize(**kwargs):
        seen.append(kwargs["module_path"])
        assert kwargs["shell_sub"] is not model.packed
        return 0

    turtle = SimpleNamespace(_lock=threading.RLock(), _materialize_direct_meta_tensors=materialize)
    LazyTurtle.sync_all_meta(turtle, shell_model=model, skip_module_types=(Packed,), tie_after=False)
    assert seen == ["", "norm"]
    assert model.packed.weight.device.type == "meta"
