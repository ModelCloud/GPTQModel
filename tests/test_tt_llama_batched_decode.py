# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Batched decode has to stop paying for rows that no longer generate.

Flash-decode divides the cores between the query rows it is handed, so a step
costs what its width costs: the module docstring records 3.4 ms per layer at 32
rows against 0.13 ms with one at 16k context. A batch that drains to a single
sequence must therefore shrink its width, which needs two things - a captured
trace per width (capturing again costs the compile time) and moving the live
slots to the front of the page table (KV blocks stay put, only the rows move).

These tests drive the bookkeeping on CPU with a fake device; the numeric result
is covered by the hardware test.
"""


import pytest
import torch

from gptqmodel.utils.tt_llama import DECODE_BATCH, KVBlockPool, TTLlama


class _FakeTtnn:
    """Records the trace lifecycle and the host-to-device copies."""

    bfloat16 = "bf16"
    uint32 = "uint32"
    int32 = "int32"
    TILE_LAYOUT = "tile"
    ROW_MAJOR_LAYOUT = "row"

    def __init__(self):
        self.captures = []
        self.releases = []
        self.executions = []
        self.copies = 0

    def from_torch(self, tensor, dtype=None, layout=None):
        return ("host", tensor)

    def to_device(self, tensor, device):
        return ("device", tensor)

    def copy_host_to_device_tensor(self, host, device):
        self.copies += 1

    def begin_trace_capture(self, device, cq_id=0):
        token = ("trace", len(self.captures))
        self.captures.append(token)
        return token

    def end_trace_capture(self, device, trace, cq_id=0):
        return None

    def execute_trace(self, device, trace, cq_id=0, blocking=False):
        self.executions.append(trace)

    def release_trace(self, device, trace):
        self.releases.append(trace)

    def deallocate(self, tensor):
        return None


def _model(*, users=DECODE_BATCH, blocks=64, block_size=64, per_user=8):
    """A TTLlama with the device-facing parts replaced."""

    model = object.__new__(TTLlama)
    model.users = users
    model.max_context = block_size * per_user
    model.ttnn = _FakeTtnn()
    model.device = object()
    model.pool = KVBlockPool(num_blocks=blocks, block_size=block_size, users=users,
                             max_blocks_per_user=per_user)
    model.max_traces = 2
    model._traces = {}
    model.active = DECODE_BATCH
    model.syncs = 0
    model.widths = []
    model.tables = []

    def _sync():
        model.syncs += 1
        model.tables.append(model.pool.table.clone())

    model._sync_page_table = _sync
    model.decode_inputs = lambda tokens, positions, on_device=False: [("x", tokens), ("p", positions)]

    def _decode(*inputs, active=DECODE_BATCH):
        model.widths.append(active)
        return ("ids", active)

    model.decode_step = _decode
    model._embed_rows = lambda ids: torch.zeros(1, 1)
    return model


def test_a_width_is_captured_once_and_reused():
    model = _model()
    tokens, positions = [0] * model.users, [0] * model.users

    model.capture(tokens, positions, active=4)
    model.capture(tokens, positions, active=4)
    assert len(model.ttnn.captures) == 1          # second capture only refreshed inputs

    model.replay(active=4)
    assert model.ttnn.executions == [model._traces[4][0]]

    model.release()
    assert model._traces == {}
    assert model.ttnn.releases == [model.ttnn.captures[0]]


def test_only_the_newest_widths_keep_a_trace():
    """Each trace pins a device region, so the cache is bounded."""

    model = _model()
    model.max_traces = 2
    tokens, positions = [0] * model.users, [0] * model.users
    for width in (8, 4, 1):
        model.capture(tokens, positions, active=width)

    assert set(model._traces) == {4, 1}
    assert len(model.ttnn.releases) == 1              # the 8-wide trace was evicted
    model.release()
    assert model._traces == {}
    assert len(model.ttnn.releases) == 3


def test_replaying_a_width_that_was_never_captured_requires_inputs():
    model = _model()
    with pytest.raises(RuntimeError, match="no trace captured for active=1"):
        model.replay(active=1)

    model.replay([0] * model.users, [0] * model.users, active=1)
    assert 1 in model._traces


def test_generate_shrinks_the_width_as_requests_finish():
    model = _model(users=8, blocks=32, per_user=8)
    model.prefill = lambda user, prompt: torch.nn.functional.one_hot(
        torch.tensor(100 + user), 256).float()

    # Slot widths the fake model is asked for, step by step.
    scripted = [
        [5, 99, 7],      # user 1 hits EOS -> width 4 -> 2
        [99, 8],         # user 0 hits EOS -> width 2 -> 1
        [99],            # user 2 hits EOS
    ]
    calls = {"index": 0}

    def _read_ids(_ids):
        batch = scripted[min(calls["index"], len(scripted) - 1)]
        calls["index"] += 1
        padded = list(batch) + [0] * (model.users - len(batch))
        return torch.tensor(padded)

    model.read_ids = _read_ids

    generated = model.generate([["a"], ["b"], ["c"]], max_new_tokens=5, eos_token_ids=(99,))

    assert model.widths[0] == 4                   # three live slots need the 4-wide trace
    assert 2 in model.widths and 1 in model.widths
    assert generated[1][-1] == 99                 # EOS recorded for the user that finished first
    assert generated[2] == [102, 7, 8, 99]
    # Compaction ran after each shrink: 3 syncs (initial + two shrink points) or more.
    assert model.syncs >= 3
    # Snapshots after the final release are empty, so look at the last table
    # that still had live rows: shrinking 3 -> 2 -> 1 leaves exactly one.
    live = [int((table.sum(dim=1) != 0).sum()) for table in model.tables]
    assert max(live) == 3
    assert 1 in live          # after the final shrink only one row is live


def test_generate_keeps_the_width_when_shrinking_is_disabled():
    model = _model(users=8, blocks=32, per_user=8)
    model.prefill = lambda user, prompt: torch.nn.functional.one_hot(
        torch.tensor(100 + user), 256).float()
    calls = {"index": 0}

    def _read_ids(_ids):
        batch = [[5, 99, 7], [6, 8], [9]][min(calls["index"], 2)]
        calls["index"] += 1
        return torch.tensor(list(batch) + [0] * (model.users - len(batch)))

    model.read_ids = _read_ids
    model.generate([["a"], ["b"], ["c"]], max_new_tokens=5, eos_token_ids=(99,), shrink_active=False)

    assert set(model.widths) == {4}
    assert model.syncs == 2  # initial sync plus the final release sync
