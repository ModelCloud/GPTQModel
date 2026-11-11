# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quantization projection fusion, selected exclusively from ModulePlan.

The original registered leaves remain the checkpoint owners. A parent invocation
consumes split outputs directly; only concatenated *packed* operands may be cached.
"""

from __future__ import annotations

import ast
import copy
import gc
import inspect
import math
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import FunctionType, MethodType

import torch

from ..nn_modules.qlinear._runtime_cache import tensor_cache_key
from ..nn_modules.qlinear.marlin import MarlinLinear
from ..nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from ..nn_modules.qlinear.torch import TorchLinear
from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
from ..utils.marlin import replace_parameter
from ..utils.model import get_layers_with_prefixes
from ._const import DEVICE, normalize_device
from .fusable import require_fusable_config
from .fusion_kernels import kernel_spec


_FIELDS = (
    "bits", "requested_group_size", "group_size", "sym", "desc_act", "pack_dtype",
    "backend", "format", "_qzeros_format", "compute_dtype", "in_features",
    "fp32", "weight_type", "is_k_full", "_marlin_tile_padding",
)
_TENSORS = ("qweight", "qzeros", "scales", "bias", "g_idx", "g_idx_sort_indices")
_SHARED_OPERANDS = (("qweight", 1), ("qzeros", 1), ("scales", 1), ("bias", 0))


def _set_operand(module, name, tensor):
    """Retain the checkpoint registration kind when rebinding packed storage."""
    if name in module._parameters:
        tensor = torch.nn.Parameter(tensor, requires_grad=module._parameters[name].requires_grad)
    setattr(module, name, tensor)


def _retired_guard():
    return False


@lru_cache(maxsize=32)
def _parent_template(original, names):
    """Replace only declared projection calls in the installed native forward.

    Everything after projection -- RoPE, cache updates, attention selection,
    activation, multiplication and output projection -- remains upstream code.
    An unfamiliar/decorated forward fails closed instead of guessing its flow.
    """
    if original.__closure__ or hasattr(original, "__wrapped__"):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return None
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef) or node.decorator_list or len(node.args.args) < 2:
        return None
    if any(isinstance(n, ast.Name) and n.id.startswith("__gptq_") for n in ast.walk(node)):
        return None
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node for n in ast.walk(node)):
        return None
    owner, input_name = node.args.args[0].arg, node.args.args[1].arg
    if any(isinstance(n, ast.Name) and n.id == input_name and isinstance(n.ctx, (ast.Store, ast.Del))
           for n in ast.walk(node)):
        return None
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and isinstance(n.func.value, ast.Name) and n.func.value.id == input_name
           and n.func.attr.endswith("_") for n in ast.walk(node)):
        return None

    def projection(call):
        return (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == owner
                and call.func.attr in names)

    # Declared roles identify the live parent, regardless of its Python class.
    # Preserve the native preamble, including cache/indexer updates. Reject
    # additional input consumers between projections, where fusion would move
    # a later projection ahead of a potentially mutating callback.
    calls = [child for child in ast.walk(node) if projection(child)]
    if not calls:
        return None
    first = min((call.lineno, call.col_offset) for call in calls)
    last = max((call.lineno, call.col_offset) for call in calls)
    first_statement = next(index for index, statement in enumerate(node.body)
                           if statement.lineno <= first[0] <= statement.end_lineno)
    insertion = (node.body[first_statement].lineno, node.body[first_statement].col_offset)
    original_continuation = copy.deepcopy(node.body[first_statement:])
    parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
    for use in ast.walk(node):
        if (not isinstance(use, ast.Name) or use.id != input_name or not isinstance(use.ctx, ast.Load)
                or (use.lineno, use.col_offset) > last):
            continue
        parent = parents[use]
        if projection(parent) and parent.args == [use] and not parent.keywords:
            continue
        if isinstance(parent, ast.Call) and (parent.lineno, parent.col_offset) < insertion:
            continue
        if isinstance(parent, ast.Attribute) and parent.value is use:
            if parent.attr in ("shape", "dtype", "device", "ndim"):
                continue
            outer = parents.get(parent)
            if (parent.attr in ("size", "dim", "numel") and isinstance(outer, ast.Call)
                    and outer.func is parent and not outer.keywords):
                try:
                    if all(isinstance(ast.literal_eval(arg), int) for arg in outer.args):
                        continue
                except (ValueError, TypeError):
                    # A dynamic argument is not proven constant metadata access.
                    return None
        return None
    counts = [0] * len(names)
    valid = True

    class Projections(ast.NodeTransformer):
        conditional = 0

        def generic_visit(self, child):
            guarded = isinstance(child, (ast.If, ast.IfExp, ast.BoolOp, ast.For, ast.AsyncFor, ast.While,
                                         ast.Try, getattr(ast, "TryStar", ast.Try), ast.Match,
                                         ast.With, ast.AsyncWith, ast.comprehension,
                                         ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                                         ast.GeneratorExp))
            self.conditional += guarded
            result = super().generic_visit(child)
            self.conditional -= guarded
            return result

        def visit_Call(self, call):
            nonlocal valid
            target = call.func
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == owner and target.attr in names):
                index = names.index(target.attr)
                if (self.conditional or len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Name)
                        or call.args[0].id != input_name):
                    valid = False
                    return call
                counts[index] += 1
                return ast.copy_location(ast.Subscript(
                    value=ast.Name(id="__gptq_outputs", ctx=ast.Load()),
                    slice=ast.Constant(index), ctx=ast.Load()), call)
            return self.generic_visit(call)

    node = Projections().visit(node)
    if not valid or counts != [1] * len(names):
        return None
    if any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
           and n.value.id == owner and n.attr in names for n in ast.walk(node)):
        return None  # An additional weight/metadata use needs a separate proof.
    args = [ast.Name(id=a.arg, ctx=ast.Load()) for a in (*node.args.posonlyargs, *node.args.args)[1:]]
    if node.args.vararg:
        args.append(ast.Starred(value=ast.Name(id=node.args.vararg.arg, ctx=ast.Load()), ctx=ast.Load()))
    keywords = [ast.keyword(arg=a.arg, value=ast.Name(id=a.arg, ctx=ast.Load())) for a in node.args.kwonlyargs]
    if node.args.kwarg:
        keywords.append(ast.keyword(arg=None, value=ast.Name(id=node.args.kwarg.arg, ctx=ast.Load())))
    entry = ast.parse("if __gptq_group_state['parent_forward'] is not __gptq_native_forward:\n    return None").body[0]
    entry.body[0].value = ast.Call(func=ast.Name(id="__gptq_fallback", ctx=ast.Load()), args=args, keywords=keywords)
    prelude = ast.parse(f"__gptq_outputs = __gptq_execute({input_name})\nif __gptq_outputs is None:\n    pass").body
    # A fallback continues after the already-executed preamble. Calling the
    # entire original forward here would run its side effects twice.
    prelude[1].body = original_continuation
    prelude[1].orelse = node.body[first_statement:]
    node.body = [entry, *node.body[:first_statement], *prelude]
    node.name = "__gptq_direct_forward"
    # Resolve annotations lazily, as upstream modules using future annotations do.
    import __future__

    namespace = original.__globals__.copy()
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename,
                 "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[node.name].__code__


def _mutation_guard(parent, modules, names, packed=None, *, fields=_FIELDS, tensors=_TENSORS,
                    axes=_SHARED_OPERANDS, mappings=()):
    """Compile direct dictionary bindings and tensor version/storage checks.

    The guard creates no metadata tuples and performs no nn.Module attribute
    searches. Inference tensors keep the explicit-clear contract for writes
    without a version counter, as in other packed operand caches.
    """
    namespace = {}
    checks = []
    versioned = set()

    def bind(value):
        key = f"v{len(namespace)}"
        namespace[key] = value
        return key

    parent_state = bind(vars(parent))
    children = bind(parent._modules)
    checks.append(f"{parent_state}['_modules'] is {children}")
    checks.append(f"{parent_state}.get('_hf_hook') is None")
    for name, module in zip(names, modules):
        state = bind(vars(module))
        checks.append(f"{children}.get({name!r}) is {bind(module)}")
        for field in (*fields, "out_features", "adapter", "online_full_had", "online_partial_had", "training", "forward"):
            value = vars(module).get(field)
            if isinstance(value, (dict, list)):
                checks.append(f"{state}.get({field!r}) == {bind(copy.deepcopy(value))}")
            else:
                checks.append(f"{state}.get({field!r}) is {bind(value)}")
        checks.extend((f"not {state}['_forward_hooks']", f"not {state}['_forward_pre_hooks']",
                       f"not {state}['_backward_hooks']", f"not {state}['_backward_pre_hooks']",
                       f"{state}.get('_hf_hook') is None"))
        for mapping_name in ("_parameters", "_buffers"):
            checks.append(f"{state}[{mapping_name!r}] is {bind(vars(module)[mapping_name])}")
        for mapping_name in mappings:
            mapping = vars(module).get(mapping_name)
            checks.append(f"{state}.get({mapping_name!r}) is {bind(mapping)}")
            if mapping is not None:
                for key, value in mapping.items():
                    checks.append(f"{bind(mapping)}.get({key!r}) is {bind(value)}")
        for tensor_name in tensors:
            mapping = module._parameters if tensor_name in module._parameters else module._buffers if tensor_name in module._buffers else vars(module)
            tensor = mapping.get(tensor_name)
            ref = bind(tensor)
            checks.append(f"{bind(mapping)}.get({tensor_name!r}) is {ref}")
            if not isinstance(tensor, torch.Tensor):
                continue
            if not tensor.is_inference():
                # _share_operands constructs sibling views with the packed
                # tensor's version counter. One check covers writes through
                # every slice; all binding/storage/metadata checks stay per leaf.
                version = (getattr(packed, tensor_name) if packed is not None and tensor.numel()
                           and tensor_name in dict(axes) else tensor)
                if id(version) not in versioned:
                    checks.append(f"{bind(version)}._version == {version._version}")
                    versioned.add(id(version))
            checks.extend((f"{ref}.data_ptr() == {tensor.data_ptr()}",
                           f"{ref}.dtype is {bind(tensor.dtype)}",
                           f"{ref}.device == {bind(tensor.device)}",
                           f"{ref}.shape == {tuple(tensor.shape)!r}"))
            if tensor.is_floating_point():
                checks.append(f"not {ref}.requires_grad")
    # Native scratch is written by kernels, so its version is deliberately not
    # checked. Rebinding/resizing/moving it must still retire the descriptor.
    if packed is not None and isinstance(getattr(packed, "workspace", None), torch.Tensor):
        workspace = packed.workspace
        ref = bind(workspace)
        mapping = packed._buffers if "workspace" in packed._buffers else vars(packed)
        checks.extend((f"{bind(mapping)}.get('workspace') is {ref}",
                       f"{ref}.data_ptr() == {workspace.data_ptr()}",
                       f"{ref}.shape == {tuple(workspace.shape)!r}",
                       f"{ref}.dtype is {bind(workspace.dtype)}",
                       f"{ref}.device == {bind(workspace.device)}",
                       f"{ref}.is_contiguous()"))
    exec("def valid():\n    return " + " and ".join(checks), namespace)
    # Avoid a function -> globals -> function cycle retaining old operand views
    # after invalidation; packed GPU storage must be released immediately.
    return namespace.pop("valid")


@dataclass(frozen=True)
class FusionDecision:
    family: str
    parent: str
    paths: tuple[str, ...]
    reason: str = ""
    profile_origins: tuple[str, ...] = ()
    policy: str = "default: enabled"

    @property
    def enabled(self):
        return not self.reason


def _effective_config(config, path):
    values = copy.deepcopy(config.to_dict())
    values.pop("dynamic", None)
    override = config.dynamic_get(path)
    if override is False:
        return None

    def merge(base, patch):
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(values, override or {})
    return values


def _incompatible(modules):
    first = modules[0]
    spec = kernel_spec(first)
    if spec is None or any(type(m) is not type(first) for m in modules):
        return "executor requires identical kernels with an explicit native fusion layout"
    if spec.layout == "gptq_fp8" and any(getattr(m, "_require_activation_stream", False) for m in modules):
        return "encoded W4A activation streams require a separate parent executor"
    marlin = type(first) in (MarlinLinear, AwqMarlinLinear)
    if marlin and (first.bits != 4 or first.requested_group_size != 128 or not first.sym or first.desc_act):
        return "Marlin fusion requires symmetric INT4, group 128, no activation order"
    if first.bits not in (2, 4, 8):
        return "executor requires 2, 4 or 8-bit word-aligned storage"
    if first.pack_dtype != (torch.int16 if spec.layout == "awq_fast" else torch.int32):
        return f"executor requires {'int16' if spec.layout == 'awq_fast' else 'int32'} packed words"
    if type(first) is AwqTorchLinear and len(modules) == 3 and len({m.out_features for m in modules}) > 1:
        # The packed weights concatenate exactly, but changing this GEMM's N
        # changes CUDA reduction rounding. Real Llama 3.2 1B exceeded the logits
        # drift gate; keep its separate QKV calls until a validated executor exists.
        return "AWQ GQA QKV fusion requires a numerically validated executor"
    fields = (*_FIELDS, *spec.fields)
    signature = tuple(getattr(first, field, None) for field in fields)
    for module in modules:
        if signature != tuple(getattr(module, field, None) for field in fields):
            return "quantized kernel settings differ"
        if getattr(module.forward, "__func__", None) is not type(module).forward:
            return "projection forward is outside the validated kernel executor"
        if module.adapter is not None or module.online_full_had or module.online_partial_had:
            return "adapters and online rotation are not supported by this executor"
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or module._backward_pre_hooks or hasattr(module, "_hf_hook")):
            return "projection has runtime hooks"
        if module.out_features * module.bits % module.pack_dtype_bits:
            return "projection output is not packed-word aligned"
        for name in (spec.weight_name, spec.zero_name, spec.scale_name):
            if name is None:
                continue
            tensor = getattr(module, name, None)
            reference = getattr(first, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                return "quantized operands are not materialized"
            if tensor.device != reference.device or tensor.dtype != reference.dtype or tensor.device != getattr(module, spec.weight_name).device:
                return "operand devices or dtypes differ"
            if tensor.requires_grad:
                return "trainable quantized operands"
        if not spec.cpu and spec.layout not in ("allspark", "int8") and module.scales.dtype not in module.SUPPORTS_DTYPES:
            return "operand dtype is unsupported by the executor"
        if (module.bias is None) != (first.bias is None):
            return "bias layouts differ"
        if module.bias is not None and (
            module.bias.device != getattr(first, spec.scale_name).device or module.bias.dtype != first.bias.dtype
            or module.bias.requires_grad
        ):
            return "bias device, dtype or gradient policy differs"
        columns = module.out_features * module.bits // module.pack_dtype_bits
        rows = math.ceil(module.in_features / module.group_size)
        if spec.cpu or spec.layout in ("awq_gemv", "awq_fast", "exllamav2", "allspark", "int8"):
            reason = spec.shape_error(module)
            if reason:
                return reason
            if spec.gptq and spec.layout not in ("allspark", "int8") and (
                    module.g_idx.shape != first.g_idx.shape or not torch.equal(module.g_idx, first.g_idx)):
                return "activation-order group indices differ"
            continue
        if marlin:
            if (normalize_device(module.qweight.device) != DEVICE.CUDA
                    or getattr(module, "_marlin_tile_padding", True) is not None or module.out_features % 64
                    or module.g_idx.numel() or module.g_idx_sort_indices.numel()
                    or module.workspace.dtype != torch.int32 or module.workspace.device != module.qweight.device
                    or not module.workspace.is_contiguous()
                    or module.workspace.numel() < max(128, torch.cuda.get_device_properties(module.qweight.device).multi_processor_count)):
                return "Marlin fusion requires finalized aligned operands without activation reordering"
            if (tuple(module.qweight.shape) != (module.in_features // 16, module.out_features * 2)
                    or tuple(module.scales.shape) != (rows, module.out_features)
                    or (type(first) is AwqMarlinLinear and tuple(module.qzeros.shape) != (rows, columns))):
                return "Marlin packed operand shapes differ from the validated layout"
            if type(first) is MarlinLinear and module.qzeros.numel():
                return "symmetric GPTQ Marlin requires implicit zero points"
            continue
        weight_shape = ((math.ceil(module.in_features * module.bits / module.pack_dtype_bits), module.out_features)
                        if spec.gptq else (module.in_features, columns))
        if (tuple(module.qweight.shape) != weight_shape or tuple(module.qzeros.shape) != (rows, columns)
                or tuple(module.scales.shape) != (rows, module.out_features)):
            return "packed operand shapes require padding or a different layout"
        if spec.gptq and (
            not isinstance(module.g_idx, torch.Tensor)
            or module.g_idx.device.type == "meta"
            or module.g_idx.shape != first.g_idx.shape
            or module.g_idx.numel() != module.in_features
            or not torch.equal(module.g_idx, first.g_idx)
        ):
            return "activation-order group indices differ"
    if len({id(m) for m in modules}) != len(modules):
        return "projections alias the same module"
    return ""


class _FusionGroup:
    def __init__(self, decision, parent, modules, activation_dtype=None):
        self.decision = decision
        self.parent = parent
        self.modules = modules
        self.kernel = kernel_spec(modules[0])
        self.activation_dtype = activation_dtype
        self.fields = (*_FIELDS, *self.kernel.fields, *self.kernel.private_fields)
        self.tensors = tuple(dict.fromkeys((*_TENSORS, *self.kernel.extra_tensors)))
        self.axes = self.kernel.axes
        self.originals = [m.forward for m in modules]
        self.original_clears = [getattr(m, "clear_weight_cache", None) for m in modules]
        self.parent_forward = parent.forward
        self.native_forward = self.parent_forward
        self.names = tuple(path.rsplit(".", 1)[-1] for path in decision.paths)
        self.widths = tuple(module.out_features for module in modules)
        self.guard = None
        self.packed_forward = None
        self.parent_state = vars(parent)
        self.active = False
        self.installed_clears = []
        self.packed = None
        self.packed_key = None
        self.calls = 0

    def _key(self):
        return (id(getattr(self.parent, "_hf_hook", None)), tuple(
            (tuple(copy.deepcopy(value) if isinstance(value, (dict, list)) else value
                   for value in (getattr(m, field, None) for field in self.fields)), m.training, m.out_features,
             id(m.adapter), m.online_full_had, m.online_partial_had,
             tuple(m._forward_hooks), tuple(m._forward_pre_hooks), id(getattr(m, "_hf_hook", None)),
             tuple(tensor_cache_key(getattr(m, name, None)) for name in self.tensors))
            for m in self.modules
        ))

    def _share_operands(self, fused):
        for name, dim in self.axes:
            tensor = getattr(fused, name)
            if tensor is None or not tensor.numel():
                continue
            sizes = [getattr(module, name).shape[dim] for module in self.modules]
            for module, view in zip(self.modules, tensor.split(sizes, dim=dim)):
                _set_operand(module, name, view)
        suppress_restore = self.kernel.layout in ("cpu_aten", "cpu_legacy", "cpu_avx512")
        try:
            if suppress_restore:
                for module in self.modules:
                    module._suspend_fused_train_restore = True
            for clear in self.original_clears:
                if clear is not None:
                    clear()
            for module in self.modules:
                self.kernel.rebind(module)
        finally:
            if suppress_restore:
                for module in self.modules:
                    module.__dict__.pop("_suspend_fused_train_restore", None)
        self.invalidate()
        self.packed = fused
        self.packed_key = self._key()

    @torch.inference_mode(False)
    @torch.no_grad()
    def _operand(self, key):
        # Inference tensors use the kernels' explicit clear_weight_cache contract
        # for writes without version counters (installed on every source below).
        if self.packed is not None and key == self.packed_key:
            return self.packed
        first = self.modules[0]
        if self.kernel.layout not in ("marlin", "gptq", "awq") or type(first) not in (TorchLinear, AwqTorchLinear, MarlinLinear, AwqMarlinLinear):
            fused = self.kernel.combine(self.modules)
            self._share_operands(fused)
            return fused
        if type(first) in (MarlinLinear, AwqMarlinLinear):
            # Marlin permutes within aligned N=64 tiles. Concatenating whole
            # tiles preserves the native weight/scale/zero-point permutations.
            # Clone the finalized shell, never run post_init/repack a second time.
            fused = copy.copy(first)
            fused._parameters = first._parameters.copy()
            fused._buffers = first._buffers.copy()
            fused._modules = first._modules.copy()
            fused._forward_hooks = first._forward_hooks.copy()
            fused._forward_pre_hooks = first._forward_pre_hooks.copy()
            fused.__dict__.pop("forward", None)
            fused.__dict__.pop("clear_weight_cache", None)
            fused.out_features = sum(m.out_features for m in self.modules)
            for name in ("qweight", "scales"):
                replace_parameter(fused, name, torch.cat([getattr(m, name) for m in self.modules], dim=1))
            if first.qzeros.numel():
                replace_parameter(fused, "qzeros", torch.cat([m.qzeros for m in self.modules], dim=1))
            if first.bias is not None:
                fused.bias = torch.cat([m.bias for m in self.modules])
            # Marlin's reduction barriers require zero-initialized scratch.
            fused.workspace = torch.zeros_like(first.workspace)
            self._share_operands(fused)
            return fused
        kwargs = {
            "bits": first.bits, "group_size": first.requested_group_size, "sym": first.sym,
            "desc_act": first.desc_act, "in_features": first.in_features,
            "out_features": sum(m.out_features for m in self.modules), "bias": first.bias is not None,
            "pack_dtype": first.pack_dtype, "register_buffers": False, "dtype": first.scales.dtype,
            "backend": first.backend,
        }
        if isinstance(first, TorchLinear):
            kwargs["format"] = first.format
        fused = type(first)(**kwargs)
        for name in ("qweight", "qzeros", "scales"):
            fused.register_buffer(name, torch.cat([getattr(m, name) for m in self.modules], dim=1))
        fused.register_buffer("bias", None if first.bias is None else torch.cat([m.bias for m in self.modules]))
        if isinstance(first, TorchLinear):
            fused.register_buffer("g_idx", first.g_idx)
            fused._qzeros_format = first._qzeros_format
            fused._init_wf_unsqueeze_buffers()
            # Constructors without registered operands initially create their
            # unpacking shifts on CPU. Large-M fallback decoding uses them too.
            fused.to(device=first.qweight.device)
        fused.training = False
        self._share_operands(fused)
        return fused

    def invalidate(self):
        if self.guard is not None:
            # Free-threaded Python may defer destroying a function even after
            # its final reference is dropped. Release generated operand bindings
            # explicitly; retained callers see a safely retired guard.
            self.guard.__code__ = _retired_guard.__code__
            self.guard.__globals__.clear()
        self.guard = self.packed_forward = None
        # Completed Dynamo traces can leave unreachable LazyCache cycles with
        # borrowed packed inputs. Collect at this cold storage-lifecycle boundary
        # so retired allocations do not wait for the next automatic GC cycle.
        if any(hasattr(getattr(module, "dequantize_weight", None), "_torchdynamo_orig_callable")
               for module in self.modules):
            gc.collect()
        if self.packed is not None:
            clear = getattr(self.packed, "clear_weight_cache", None)
            if clear is not None:
                clear()
            # nn.Module hooks can form cycles. Drop storage bindings eagerly
            # rather than waiting for cyclic GC to release a retired executor.
            self.packed._parameters.clear()
            self.packed._buffers.clear()
            self.packed._modules.clear()
            self.packed.__dict__.pop("workspace", None)
            self.kernel.release(self.packed)
        self.packed = self.packed_key = None

    def _refresh(self, activation_dtype=None):
        """Cold path: validate the live owners, rebuild operands and descriptors."""
        if (any(self.parent._modules.get(name) is not module for name, module in zip(self.names, self.modules))
                or any(module.training for module in self.modules) or hasattr(self.parent, "_hf_hook")
                or _incompatible(self.modules)):
            self.invalidate()
            return False
        activation_dtype = activation_dtype or self.activation_dtype
        self.kernel.prepare(self.modules, activation_dtype)
        if _incompatible(self.modules):
            self.invalidate()
            return False
        if self.kernel.combined_shape_error(self.modules, activation_dtype):
            self.invalidate()
            return False
        # A failed guard may concern executor-owned scratch rather than source
        # tensors. Rebuild even when the checkpoint key itself did not change.
        self.invalidate()
        self._operand(self._key())
        self.packed_forward = self.packed.forward
        self.activation_dtype = activation_dtype
        self.input_dtype = activation_dtype or self.kernel.input_dtype(self.modules[0])
        self.input_device = getattr(self.modules[0], self.kernel.scale_name).device
        self.widths = tuple(module.out_features for module in self.modules)
        self.guard = _mutation_guard(self.parent, self.modules, self.names, self.packed,
                                     fields=self.fields, tensors=self.tensors, axes=self.axes,
                                     mappings=self.kernel.private_mappings)
        return True

    def execute(self, x):
        if (not self.active or self.parent_state["training"] or x.requires_grad or torch.compiler.is_compiling()
                or self.parent_forward is not self.native_forward
                or torch.nn.modules.module._global_forward_pre_hooks
                or torch.nn.modules.module._global_forward_hooks):
            return None
        if self.guard is None or not self.guard():
            if not self._refresh(x.dtype):
                return None
        if x.dtype != self.input_dtype or x.device != self.input_device:
            return None
        output = self.packed_forward(x)
        self.calls += 1
        return output.split(self.widths, dim=-1)

    def _fallback(self, *args, **kwargs):
        return self.parent_forward(*args, **kwargs)

    def install(self):
        original = self.native_forward.__func__
        code = _parent_template(original, self.names)
        namespace = original.__globals__.copy()
        namespace.update(__gptq_execute=self.execute, __gptq_fallback=self._fallback,
                         __gptq_group_state=vars(self), __gptq_native_forward=self.native_forward)
        direct = FunctionType(code, namespace, original.__name__, original.__defaults__)
        direct.__kwdefaults__ = original.__kwdefaults__
        direct.__annotations__ = original.__annotations__.copy()
        direct.__doc__ = original.__doc__
        direct.__qualname__ = original.__qualname__
        self.installed_forward = MethodType(direct, self.parent)
        self.parent.forward = self.installed_forward
        self.active = True
        self.installed_clears = []
        for module in self.modules:
            def clear_weight_cache():
                self.invalidate()
                for sibling_clear in self.original_clears:
                    if sibling_clear is not None:
                        sibling_clear()
            module.clear_weight_cache = clear_weight_cache
            self.installed_clears.append(clear_weight_cache)
        # Build before serving tokens. Training/offloaded/partial models keep
        # the original path until their operands become eligible.
        self._refresh()

    def remove(self):
        self.active = False
        if self.parent.forward is self.installed_forward:
            self.parent.forward = self.native_forward
        for module, clear, installed in zip(self.modules, self.original_clears, self.installed_clears):
            if vars(module).get("clear_weight_cache") is not installed:
                continue
            if clear is None:
                module.__dict__.pop("clear_weight_cache", None)
            else:
                module.clear_weight_cache = clear
        # Separate kernels regain independent contiguous checkpoint operands.
        # Cloning also releases an entire fused allocation held by small views.
        with torch.inference_mode(False), torch.no_grad():
            for module in self.modules:
                for name, _dim in self.axes:
                    tensor = getattr(module, name)
                    if tensor is not None and tensor.numel():
                        _set_operand(module, name, tensor.clone(memory_format=torch.contiguous_format))
                # An export may have restored the checkpoint layout while the
                # group was still installed. Rebuilding a CPU runtime descriptor
                # from checkpoint-shaped operands would reinterpret GPTQ codes as
                # int4pack storage; the next forward re-finalizes instead.
                if self.kernel.cpu and not self.kernel.runtime_ready(module):
                    continue
                self.kernel.rebind(module)
            for clear in self.original_clears:
                if clear is not None:
                    clear()
        self.invalidate()


class FusionPlanner:
    """Inspectable decisions and optional execution for a quantized wrapper."""

    def __init__(self, model, overrides=None):
        self.decisions = []
        self.groups = []
        if not model.quantized:
            raise RuntimeError("Fusion requires a post-quantized model")
        self.config = require_fusable_config(type(model), getattr(model.model.config, "model_type", None))
        if not isinstance(model.model, torch.nn.Module):
            return
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("Fusion overrides must be a mapping")
        if any(type(value) is not bool for value in (overrides or {}).values()):
            raise TypeError("Fusion overrides must map family or parent paths to bool")
        plan = model.module_plan()
        profiles = self.config.profiles_for(getattr(model.model.config, "model_type", None))
        claimed_parents = {}
        layers, layer_names = get_layers_with_prefixes(model.model, model.extract_layers_node())
        if not layers:
            raise ValueError("No declared decoder layers resolved for fusion")
        for ordinal, (layer_name, layer) in enumerate(zip(layer_names, layers)):
            index = layer_name.rsplit(".", 1)[-1]
            layer_index = int(index) if index.isdecimal() else ordinal
            for parent in dict.fromkeys(spec.parent for spec in plan.specs if spec.roles):
                if not parent or plan.is_moe(parent) or "{expert_index}" in parent:
                    continue
                for profile in profiles:
                    family, roles = profile.name, profile.roles
                    candidates = [plan.role_paths_in_parent(parent, role) for role in roles]
                    if not any(candidates):
                        continue
                    full_parent = f"{layer_name}.{parent}"
                    paths = tuple(paths[0] for paths in candidates if len(paths) == 1)
                    reason = ""
                    enabled, policy = self.config.policy_for(
                        family, layer_index, parent=parent, full_parent=full_parent, overrides=overrides)
                    if not enabled:
                        reason = "disabled by fusion override"
                    elif any(len(paths) != 1 for paths in candidates) or len(set(paths)) != len(roles):
                        reason = "requires exactly one distinct declared projection per role"
                    elif any(not plan.find(path).quantize for path in paths):
                        reason = "projection is excluded or capture-only"
                    try:
                        owner = layer.get_submodule(parent)
                        modules = [layer.get_submodule(path) for path in paths]
                    except AttributeError:
                        reason = reason or "declared projections are absent on this layer"
                        modules = []
                    if not reason:
                        configs = [_effective_config(model.quantize_config, f"{layer_name}.{p}") for p in paths]
                        if configs[0] is None or any(c != configs[0] for c in configs[1:]):
                            reason = "effective quantization configurations differ or exclude a projection"
                        elif hasattr(owner, "_hf_hook"):
                            reason = "parent has offload hooks"
                        else:
                            reason = _incompatible(modules)
                            if not reason:
                                reason = kernel_spec(modules[0]).combined_shape_error(
                                    modules, getattr(model.model, "dtype", None))
                            forward = owner.forward
                            if not reason and (
                                getattr(forward, "__func__", None) is not type(owner).forward
                                or _parent_template(forward.__func__, tuple(path.rsplit(".", 1)[-1] for path in paths)) is None
                            ):
                                reason = "native forward cannot be safely rewritten from declared projection calls"
                    decision = FusionDecision(family, full_parent, paths, reason,
                                              self.config.profile_origins(family), policy)
                    self.decisions.append(decision)
                    if decision.enabled:
                        if id(owner) in claimed_parents:
                            raise ValueError(f"Multiple fusion profiles match {full_parent}: "
                                             f"{claimed_parents[id(owner)]!r} and {family!r}; "
                                             "declare only one profile per parent")
                        claimed_parents[id(owner)] = family
                        self.groups.append(_FusionGroup(decision, owner, modules,
                                                        activation_dtype=getattr(model.model, "dtype", None)))
    def apply(self):
        installed = []
        try:
            for group in self.groups:
                installed.append(group)
                group.install()
        except Exception:
            for group in reversed(installed):
                group.remove()
            raise
        return self

    def remove(self):
        for group in reversed(self.groups):
            group.remove()

    def to_dict(self):
        return {"fusable": self.config.to_dict(),
                "groups": [{"family": d.family, "parent": d.parent, "paths": list(d.paths),
                                "enabled": d.enabled, "reason": d.reason,
                                "profile_origins": list(d.profile_origins), "policy": d.policy} for d in self.decisions],
                "fused_calls": sum(g.calls for g in self.groups)}

    def render(self):
        lines = [f"FusionPlanner: {len(self.groups)} eligible / {len(self.decisions)} declared groups"]
        for decision in self.decisions:
            status = "eligible" if decision.enabled else f"skip: {decision.reason}"
            lines.append(f"  {decision.parent} [{decision.family}] [{status}]"
                         f" [from: {', '.join(decision.profile_origins)}; {decision.policy}]")
        return "\n".join(lines)

    def print(self, *, file=None):
        print(self.render(), file=file)
