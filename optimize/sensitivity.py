# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical, teacher-forced linear-kernel propagation measurements.

Reuse calibration_coverage's target census, then verify input sharing in actual
forwards. Only selected projection outputs change; blocks, heads, embeddings and
nonlinear operations keep their original implementations. Built-in noise probes
are diagnostics, never kernel acceptance or model-quality evidence.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .calibration_coverage import find_target_groups
from .sensitivity_metrics import (
    ErrorStats,
    logit_metrics,
    measured_gain,
    uncorrelated_prediction,
)


@dataclass(frozen=True)
class ProbeContext:
    name: str
    batch: int
    invocation: int
    input_group: str
    amplitude: float
    seed: int


def _noise(value: torch.Tensor, amplitude: float, key: str, seed: int) -> torch.Tensor:
    if value.numel() == 0:
        return value.clone()
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    noise = torch.randn(value.shape, generator=generator, dtype=torch.float64)
    norm = value.detach().double().norm().item()
    noise *= amplitude * norm / noise.norm().item()
    return (value.double() + noise.to(value.device)).to(value.dtype)


def output_noise(context: ProbeContext, module, args, kwargs, reference: torch.Tensor) -> torch.Tensor:
    """Independent, reproducible output directions, fixed across amplitude/subset arms."""
    return _noise(reference, context.amplitude, f"{context.batch}:{context.name}:{context.invocation}", context.seed)


def shared_input_noise(context: ProbeContext, module, args, kwargs, reference: torch.Tensor) -> torch.Tensor:
    """One diagnostic input-error direction shared by consumers of the same runtime view.

    Return candidate outputs without mutating the actual shared input. Custom
    quantized/fused operators should supply their own pure candidate callback.
    """
    if not isinstance(module, nn.Linear):
        raise TypeError("shared_input_noise supports nn.Linear; provide a callback for other operators")
    x = args[0] if args else kwargs["input"]
    changed = _noise(x, context.amplitude, context.input_group, context.seed)
    return torch.nn.functional.linear(changed, module.weight, module.bias)


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_clone(item) for item in value)
    return value


def _logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        return output["logits"]
    return output.logits


def _rows(value, mask):
    if mask is not None and tuple(value.shape[:-1]) == tuple(mask.shape):
        return value[mask.to(value.device)]
    return value.reshape(-1, value.shape[-1])


class SensitivitySweep:
    """First sweep all linears in each layer, then singles and subsets in top layers.

    ``layers`` names actual decoder blocks (e.g. model.layers.0), not a guessed
    architecture. Explicit ``subsets`` map labels to full module names and can
    describe definition groups even when their runtime inputs differ. Observed
    sharing is a diagnostic fact, not authorization to fuse a group.

    ``candidate(context, module, args, kwargs, reference)`` must be pure and return
    the same output shape/dtype/device. The reference is evaluated on the same live
    input as the candidate. Do not call module() from its own candidate hook.
    ``reset_state(model)`` is required for models with private mutable caches;
    standard input caches are rejected and use_cache=False is passed when supported.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[str],
        *,
        subsets: Mapping[str, Sequence[str]] | None = None,
        module_names: Sequence[str] | None = None,
        candidate: Callable = output_noise,
        seed: int = 0,
        reset_state: Callable | None = None,
    ):
        self.model, self.layers = model, tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Provide distinct decoder layer paths")
        self.blocks = {name: model.get_submodule(name) for name in self.layers}
        if any(a.startswith(b + ".") for a in self.layers for b in self.layers if a != b):
            raise ValueError("Layer scopes must not overlap")
        if len({id(m) for m in self.blocks.values()}) != len(self.blocks):
            raise ValueError("Aliased decoder layers require an invocation-aware adapter")
        self.targets, self.owner, self.hints = {}, {}, {}
        for layer, block in self.blocks.items():
            if isinstance(block, nn.Linear):
                raise ValueError(  # noqa: TRY004 - preserve the public validation contract
                    "Layer scopes must be decoder blocks containing projections"
                )
            for group in find_target_groups(block, include_all_linear=True):
                names = []
                for relative, _role in group.members:
                    name = f"{layer}.{relative}"
                    self.targets[name] = model.get_submodule(name)
                    self.owner[name] = layer
                    names.append(name)
                self.hints[f"{layer}:{group.group_id}"] = tuple(names)
        if module_names is not None:
            if not module_names or len(set(module_names)) != len(module_names):
                raise ValueError("Provide distinct explicit module names")
            self.targets, self.owner = {}, {}
            for name in module_names:
                owners = [layer for layer in self.layers if name.startswith(layer + ".")]
                if len(owners) != 1:
                    raise ValueError("Explicit modules must be inside a selected decoder layer")
                self.targets[name] = model.get_submodule(name)
                self.owner[name] = owners[0]
            if any(a.startswith(b + ".") for a in self.targets for b in self.targets if a != b):
                raise ValueError("Explicit module scopes must not overlap")
        if not self.targets or any(layer not in self.owner.values() for layer in self.layers):
            raise ValueError("Every layer must contain supported linear targets")
        target_ids = {id(m) for m in self.targets.values()}
        aliases = Counter(id(m) for _, m in model.named_modules(remove_duplicate=False) if id(m) in target_ids)
        if any(count != 1 for count in aliases.values()):
            raise ValueError("Aliased target modules require an invocation-aware adapter")
        # Prevent accidentally selecting a head as a layer even if the supplied scope is broad.
        for getter in ("get_input_embeddings", "get_output_embeddings"):
            endpoint = getattr(model, getter, lambda: None)()
            if endpoint is not None and id(endpoint) in target_ids:
                raise ValueError("Layer scopes must exclude embedding and output head")
        self.subsets = {label: tuple(names) for label, names in (subsets or {}).items()}
        for names in self.subsets.values():
            if len(names) < 2 or len(set(names)) != len(names) or any(n not in self.targets for n in names):
                raise ValueError("Subsets must contain at least two distinct inventoried targets")
            if len({self.owner[n] for n in names}) != 1:
                raise ValueError("Module subsets must belong to one layer")
        self.candidate, self.seed, self.reset_state = candidate, seed, reset_state
        signature = inspect.signature(model.forward)
        self.use_cache_arg = "use_cache" in signature.parameters or any(
            p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
        )

    def _run(self, batches, active=(), amplitude=0.0, references=None):
        active = set(active)
        local = {name: ErrorStats() for name in active}
        full_local = {name: ErrorStats() for name in active}
        calls = Counter()
        batch_calls = Counter()
        masked_calls = Counter()
        sharing = Counter()
        logits, per_batch = [], []
        final_stats = ErrorStats()
        frames, block_calls = {}, Counter()
        batch_index, mask = 0, None
        handles = []
        modes = {module: module.training for module in self.model.modules()}
        cuda_devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})

        def begin(layer):
            def hook(module, args):
                block_calls[layer] += 1
                frames[layer] = {"inputs": {}, "holds": [], "members": defaultdict(list)}

            return hook

        def finish(layer):
            def hook(module, args, output):
                frame = frames.pop(layer)
                for members in frame["members"].values():
                    # Repeat calls to one module do not establish sibling sharing.
                    names = tuple(sorted(set(members)))
                    if len(names) > 1:
                        sharing[names] += 1

            return hook

        def intervene(name):
            def hook(module, args, kwargs, output):
                nonlocal mask
                layer = self.owner[name]
                frame = frames[layer]
                x = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
                if not torch.is_tensor(x) or not torch.is_tensor(output) or not output.is_floating_point():
                    raise TypeError(f"{name}: expected tensor input and floating tensor output")
                # Retain views until the enclosing block returns, preventing allocator
                # pointer reuse from masquerading as shared input. Include version so
                # in-place writes between consumers split the group.
                key = (
                    x.device,
                    x.dtype,
                    x.untyped_storage().data_ptr(),
                    x.storage_offset(),
                    tuple(x.shape),
                    tuple(x.stride()),
                    x._version,
                )
                if key not in frame["inputs"]:
                    frame["inputs"][key] = len(frame["inputs"])
                    frame["holds"].append(x.detach())
                input_id = frame["inputs"][key]
                frame["members"][input_id].append(name)
                batch_calls[name] += 1
                calls[name] += 1
                if name not in active:
                    return output
                reference = output.detach().clone()
                context = ProbeContext(
                    name,
                    batch_index,
                    batch_calls[name],
                    f"{batch_index}:{layer}:{block_calls[layer]}:{input_id}",
                    amplitude,
                    self.seed,
                )
                candidate = self.candidate(context, module, args, kwargs, reference.clone())
                if not torch.is_tensor(candidate) or (candidate.shape, candidate.dtype, candidate.device) != (
                    reference.shape,
                    reference.dtype,
                    reference.device,
                ):
                    raise ValueError(f"{name}: candidate must preserve output shape, dtype and device")
                full_local[name].update(reference, candidate)
                if mask is not None and tuple(reference.shape[:-1]) == tuple(mask.shape):
                    masked_calls[name] += 1
                local[name].update(_rows(reference, mask), _rows(candidate, mask))
                return candidate

            return hook

        try:
            self.model.eval()
            for name, block in self.blocks.items():
                handles.append(block.register_forward_pre_hook(begin(name)))
                handles.append(block.register_forward_hook(finish(name)))
            for name, module in self.targets.items():
                handles.append(module.register_forward_hook(intervene(name), with_kwargs=True))
            with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(self.seed)
                for index in cuda_devices:
                    torch.cuda.default_generators[index].manual_seed(self.seed)
                for batch_index, batch in enumerate(batches):
                    frames.clear()
                    block_calls.clear()
                    batch_calls.clear()
                    if self.reset_state is not None:
                        self.reset_state(self.model)
                    values = _clone(batch)
                    mask = values.pop("sensitivity_mask", values.get("attention_mask"))
                    if mask is not None:
                        mask = mask.bool()
                    if self.use_cache_arg:
                        values["use_cache"] = False
                    output = _logits(self.model(**values))
                    if mask is not None and tuple(output.shape[:-1]) != tuple(mask.shape):
                        raise ValueError("Logit positions do not match the evaluation mask")
                    selected = _rows(output, mask).detach().cpu().clone()
                    if references is None:
                        if not bool(torch.isfinite(selected).all()):
                            raise ValueError("Reference logits must be finite")
                        # Also reject empty/vocabulary-one output before any intervention.
                        logit_metrics(selected, selected)
                        logits.append(selected)
                    else:
                        per_batch.append(logit_metrics(references[batch_index], selected))
                        final_stats.update(references[batch_index], selected)
        finally:
            for handle in handles:
                handle.remove()
            for module, mode in modes.items():
                module.training = mode
            frames.clear()
        return {
            "logits": logits,
            "per_batch": per_batch,
            "calls": dict(calls),
            "sharing": sharing,
            "masked_calls": dict(masked_calls),
            "final": final_stats.report(),
            "local": {n: s.report() for n, s in local.items()},
            "full_local": {n: s.report() for n, s in full_local.items()},
        }

    def sweep(
        self,
        batches: Sequence[Mapping],
        *,
        amplitudes=(1e-3, 2e-3),
        top_k=3,
        pairwise=True,
        max_group_size=8,
        progress: Callable | None = None,
    ) -> dict:
        if not batches or top_k < 0 or max_group_size < 2:
            raise ValueError("Provide nonempty batches, top_k >= 0 and max_group_size >= 2")
        amplitudes = tuple(float(a) for a in amplitudes)
        if (
            not amplitudes
            or len(set(amplitudes)) != len(amplitudes)
            or any(not math.isfinite(a) or a < 0 for a in amplitudes)
        ):
            raise ValueError("Amplitudes must be distinct finite nonnegative values")
        for batch in batches:
            if any(k in batch for k in ("past_key_values", "past_key_value", "cache_params", "mems")):
                raise ValueError("Use fresh teacher-forced batches without input caches")
            if "sensitivity_mask" in batch and "attention_mask" in batch and (
                batch["sensitivity_mask"].shape != batch["attention_mask"].shape
                or bool((batch["sensitivity_mask"].bool() & ~batch["attention_mask"].bool()).any())
            ):
                raise ValueError("sensitivity_mask must select only valid attention_mask positions")
        baseline = self._run(batches)
        repeated = self._run(batches, references=baseline["logits"])
        if not repeated["final"]["finite"]:
            raise ValueError("Repeated baseline is nonfinite")
        noise = repeated["final"]["relative_l2"] or 0.0
        rows, cache = [], {}

        def trial(stage, label, members, amplitude):
            members = tuple(sorted(members))
            key = (members, amplitude)
            if key not in cache:
                result = self._run(batches, members, amplitude, baseline["logits"])
                final = result["per_batch"]
                finite = (
                    result["final"]["finite"]
                    and all(r["finite"] for r in final)
                    and all(r["finite"] for r in result["full_local"].values())
                )
                errors = [r["relative_l2"] for r in final]
                final_error = max(errors) if all(e is not None for e in errors) and finite else None
                eps = [result["local"][n]["relative_l2"] for n in members]
                # Match the aggregation of final E and local epsilons across batches.
                gain, status = measured_gain(result["final"]["relative_l2"] if finite else None, eps, noise)
                observed = all(result["local"][n]["count"] > 0 for n in members)
                cache[key] = {
                    "members": list(members),
                    "amplitude": amplitude,
                    "per_batch": final,
                    "final_relative_l2_max": final_error,
                    "g_effective": gain,
                    "g_status": status,
                    "final": result["final"],
                    "finite": finite,
                    "observed": observed,
                    "local": result["local"],
                    "local_all_positions": result["full_local"],
                    "calls": result["calls"],
                    "local_masked_calls": result["masked_calls"],
                    "shared_input_observations": sum(
                        count for names, count in result["sharing"].items() if set(members) <= set(names)
                    )
                    if len(members) > 1
                    else None,
                    "within_kernel_output_tolerance": finite
                    and observed
                    and all(r["max_abs"] is not None and r["max_abs"] <= 2e-3 for r in result["full_local"].values()),
                }
            row = dict(cache[key], stage=stage, target=label)
            rows.append(row)
            if progress is not None:
                progress(row)
            return row

        layer_rows = defaultdict(list)
        for layer in self.layers:
            for amplitude in amplitudes:
                layer_rows[layer].append(
                    trial("layer", layer, [n for n in self.targets if self.owner[n] == layer], amplitude)
                )

        # Prioritize nonfinite/unobserved cases for investigation, never mark them safe.
        def score(layer):
            values = layer_rows[layer]
            if any(not r["finite"] or not r["observed"] or r["final_relative_l2_max"] is None for r in values):
                return math.inf
            return max(r["final_relative_l2_max"] for r in values)

        ranking = sorted(self.layers, key=lambda name: (-score(name), self.layers.index(name)))
        selected = ranking[:top_k]
        groups, skipped = {}, []
        for names, count in baseline["sharing"].items():
            if len(names) > max_group_size:
                skipped.append({"members": list(names), "reason": "max_group_size", "observations": count})
                continue
            groups[names] = {"source": "observed_shared_input", "observations": count}
            if pairwise:
                for pair in itertools.combinations(names, 2):
                    groups.setdefault(pair, {"source": "shared_input_pair", "observations": count})
        for label, members in self.subsets.items():
            names = tuple(sorted(members))
            groups[names] = {
                "source": "declared_subset",
                "label": label,
                "observations": sum(
                    count for observed, count in baseline["sharing"].items() if set(names) <= set(observed)
                ),
            }
        for layer in selected:
            for name in self.targets:
                if self.owner[name] == layer:
                    for amplitude in amplitudes:
                        trial("module", name, [name], amplitude)
            for members, evidence in groups.items():
                if self.owner[members[0]] != layer:
                    continue
                for amplitude in amplitudes:
                    row = trial("subset", evidence.get("label", " + ".join(members)), members, amplitude)
                    row["sharing"] = evidence
                    isolated = [cache[((name,), amplitude)]["final"]["relative_l2"] for name in members]
                    predicted = uncorrelated_prediction(isolated)
                    row["uncorrelated_prediction"] = predicted
                    row["joint_over_prediction"] = measured_gain(row["final"]["relative_l2"], [predicted], 0)[0]
        for amplitude in amplitudes:
            row = trial("combined", "all_target_linears", self.targets, amplitude)
            row["uncorrelated_layer_prediction"] = uncorrelated_prediction(
                [
                    next(
                        r["final"]["relative_l2"] if r["finite"] and r["observed"] else None
                        for r in layer_rows[layer]
                        if r["amplitude"] == amplitude
                    )
                    for layer in self.layers
                ]
            )
        return {
            "schema_version": 1,
            "probe": getattr(self.candidate, "__name__", type(self.candidate).__name__),
            "seed": self.seed,
            "amplitudes": list(amplitudes),
            "layer_ranking": ranking,
            "aggregation": {
                "final": "raw logits over all selected positions across batches",
                "local": "matching mask when shape aligns; otherwise all invocation rows",
                "gain": "final relative L2 / sqrt(sum(module relative L2 squared))",
                "ranking": "worst batch final relative L2 across declared amplitudes",
            },
            "selected_layers": selected,
            "targets": {n: {"layer": self.owner[n], "class": type(m).__name__} for n, m in self.targets.items()},
            "name_group_hints": self.hints,
            "baseline_repeat": repeated["per_batch"],
            "baseline_calls": baseline["calls"],
            "skipped_groups": skipped,
            "rows": rows,
            "limitations": [
                "Diagnostic sweeps do not certify task accuracy or relax the max-absolute 2e-3 gate.",
                "Unvisited modules and unmeasured inputs have unknown sensitivity.",
                "Sharing is observed only on these forwards and does not establish legal fusion.",
                "Fused projections remain physical modules; logical output slices need an adapter.",
            ],
        }
