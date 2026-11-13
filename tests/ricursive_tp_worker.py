# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Two-rank integration worker; launched by the Ripple regression suite."""

import json
import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
from ricursive_test_utils import compatibility_module
from ricursive_tp_oracle import oracle as partitioned_oracle
from test_ricursive_modeling_numerics import oracle
from test_ricursive_review_regressions import review_model
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache, StaticCache


def main():
    mode, runtime, dtype_name = sys.argv[1:]
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(runtime, local_rank) if runtime == "cuda" else torch.device("cpu")
    if runtime == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if runtime == "cuda" else "gloo")
    saved = [tempfile.mkdtemp(prefix="ripple-tp-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(saved)
    try:
        dtype = getattr(torch, dtype_name)
        reference = review_model(mode, num_attention_heads=8)
        ids = torch.tensor([[3, 5, 6, 8, 9, 2, 4]])
        # Independent CPU FP32 arithmetic establishes the unsharded baseline.
        with torch.no_grad():
            expected_oracle = oracle(reference, ids, torch.ones_like(ids))
            torch.testing.assert_close(reference(ids, use_cache=False).logits, expected_oracle, atol=2e-5, rtol=2e-5)
        reference.to(device=device, dtype=dtype)
        ids = ids.to(device)
        if dist.get_rank() == 0:
            reference.save_pretrained(saved[0])
        dist.barrier()
        mesh = init_device_mesh(runtime, (dist.get_world_size(),), mesh_dim_names=("tp",))
        model = (
            type(reference)
            .from_pretrained(
                saved[0],
                tp_plan="auto",
                device_mesh=mesh,
                torch_dtype=dtype,
            )
            .eval()
        )
        if mode != "native":
            compatibility_module().apply_ricursive_generate_patch(model)
        with torch.no_grad():
            unsharded = reference(ids, use_cache=False).logits
            trace = {}
            expected = partitioned_oracle(reference, ids, trace=trace)
            handles = []
            projection_errors = []
            for name, module in model.named_modules():
                if name in trace:

                    def check_projection(_m, _args, out, name=name):
                        projection_errors.append((out - trace[name]).abs().max().item())
                        torch.testing.assert_close(out, trace[name], atol=2e-3, rtol=2e-3, msg=name)

                    handles.append(module.register_forward_hook(check_projection))
            default = model(ids, use_cache=False, output_attentions=True).logits
            for handle in handles:
                handle.remove()
            assert torch.isfinite(default).all() and default.dtype == dtype
            assert torch.equal(default.argmax(-1), unsharded.argmax(-1))
            if dtype != torch.bfloat16:
                torch.testing.assert_close(default, unsharded, atol=2e-3, rtol=2e-3)
            assert torch.equal(
                model.generate(ids, max_new_tokens=3, do_sample=False),
                reference.generate(ids, max_new_tokens=3, do_sample=False),
            )

        # Isolate partition/reduction arithmetic from fused SDPA's BF16 rounding.
        # Each shard rounds its GEMM output before reduction; the independent
        # oracle models that rounding explicitly. Default-backend projection
        # error and token parity were checked above; whole-model drift is reported.
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            actual = model(ids, use_cache=False, output_attentions=True)
            torch.testing.assert_close(actual.logits, expected, atol=2e-3, rtol=2e-3)
            assert torch.equal(actual.logits.argmax(-1), expected.argmax(-1))
            assert actual.logits.dtype == dtype
            for cache in [DynamicCache(config=model.config), StaticCache(config=model.config, max_cache_len=16)]:
                chunks = []
                for start, end in [(0, 3), (3, 6), (6, 7)]:
                    chunks.append(model(ids[:, start:end], past_key_values=cache, use_cache=True).logits)
                cached = torch.cat(chunks, dim=1)
                torch.testing.assert_close(cached, expected, atol=2e-3, rtol=2e-3)
                assert torch.equal(cached.argmax(-1), expected.argmax(-1))
            assert torch.equal(
                model.generate(ids, max_new_tokens=3, do_sample=False),
                reference.generate(ids, max_new_tokens=3, do_sample=False),
            )
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "dtype": dtype_name,
                            "mode": mode,
                            "runtime": runtime,
                            "default_unsharded_drift": (default - unsharded).abs().max().item(),
                            "default_projection_error": max(projection_errors),
                            "oracle_error": (actual.logits - expected).abs().max().item(),
                        }
                    ),
                    flush=True,
                )
    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(saved[0])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
