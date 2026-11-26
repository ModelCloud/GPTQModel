# LoRA adapters and the forward-dtype contract

`contracts/kernel/forward-dtype.md` freezes a module's activation dtype at the
first forward (K1) and forbids parameter/buffer dtype changes afterwards (K2).
`Lora` is not a kernel module, but it is applied inside one: `apply()` runs on
every forward of the layer that owns it, so it follows the same rule.

## Alignment happens once

| bind site | dtype source | when the projections are aligned |
| --- | --- | --- |
| dense `nn.Linear` (`utils/model.py`) | `weight.dtype` | `post_init` |
| quantized linear (`BaseQuantLinear.post_init`) | unknown until the first forward | first eager forward |

`Lora.post_init(..., dtype=...)` casts `lora_A`/`lora_B` once and records
`_forward_dtype`. A float64 export is still narrowed to float32. Without a dtype
hint the first eager forward aligns the projections and records the lock; while
`torch.compile` is tracing, the lock is checked but not recorded, matching
`ForwardDtypeLockMixin`.

After the lock, a forward whose activation dtype differs raises instead of
re-casting:

```console
RuntimeError: Adapter: LoRA projections are locked to `torch.bfloat16` after the
first forward, but the activation is `torch.float16`. Cast the model and its
adapter before the first forward instead of switching dtype later
(contracts/kernel/forward-dtype.md, K1/K2).
```

Device moves stay allowed: `apply()` moves the projections with the module,
which changes no dtype (`lora_A`/`lora_B` are plain attributes, so a module
`.to(device)` does not carry them).

## Scale folding

`_fold_scale_into_b()` multiplies `lora_scale` into `lora_B` only when the fold
survives the dtype the module runs in. The dtype is the bind-time one when
known; a dtype hint that is not the storage dtype is validated through that
dtype. Without a hint the conservative FP16 bound is kept, because the first
forward may pick FP16 and a value finite there is finite in the wider FP32/BF16
activations.

## Why `optimize()` stays off

`TorchLinear.post_init` calls `optimize()` for every module, and that method
forwards to the adapter. Enabling compilation here would compile one adapter per
layer at load time; that needs its own measurement. The cast removal is the
prerequisite, not the decision.
