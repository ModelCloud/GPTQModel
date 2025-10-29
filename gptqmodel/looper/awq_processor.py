# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from ..nn_modules.qlinear.awq_gemv import AwqGEMVQuantLinear
from ..nn_modules.qlinear.awq_gemv_fast import AwqGEMVFastQuantLinear
from ..nn_modules.qlinear.awq_marlin import AwqMarlinQuantLinear
from ..quantization.awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV, WQLinear_GEMVFast, WQLinear_Marlin
from ..quantization.awq.quantize.scale import apply_clip, apply_scale
from ..quantization.awq.utils.module import append_str_prefix, get_op_name, set_op_by_name
from ..quantization.awq.utils.utils import get_best_device
from ..quantization.config import FORMAT, METHOD, QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.ctx import ctx
from ..utils.model import get_module_by_name_prefix, move_to
from ..utils.torch import CPU

log = setup_logger()


@dataclass
class _AWQLayerState:
    modules: Dict[str, NamedModule] = field(default_factory=dict)
    subset_total: Optional[int] = None
    processed_subsets: Set[int] = field(default_factory=set)
    layer_module: Optional[torch.nn.Module] = None
    previous_weight_scale: Optional[float] = None
    quantized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

class AWQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int, gptq_model, model,
                 require_fwd: bool = True, calculate_w_wq_diff: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration=calibration,
                         calibration_concat_size=calibration_concat_size, calibration_sort=calibration_sort,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         require_fwd=require_fwd, fwd_after_process=False)

        self.calculate_w_wq_diff = calculate_w_wq_diff
        self.avg_losses = []
        self.nsamples = 0
        self._nsamples_total = 0
        self._quant_batch_size = batch_size

        self._layer_states: Dict[int, _AWQLayerState] = {}
        self._layer_states_lock = threading.Lock()
        self._scale_context = threading.local()
        self.gptq_model = gptq_model
        self.model = model
        # Whether to apply clipping to the model during quantization. Some models may perform better with this set to False.
        self.apply_clip = True
        # "The loss computation and per-channel mean is optimized into chunked computations."
        # " Adjust this parameter to increase or decrease memory usage for these computations."
        # " Default is 1GB (1024 * 1024 * 1024)."
        self.max_chunk_memory = 1024 * 1024 * 1024

        # This argument avoids real quantization by only applying the scales without quantizing down to FP16.
        self.export_compatible = False

        self.version = qcfg.format

        # Whether to scale using both w/x or just x.
        self.duo_scaling = True

        self._module_forward_kwargs: Dict[str, torch.Tensor] = {}
        self._initialize_sample_counts()
        self._module_forward_kwargs.setdefault("attention_mask", None)

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("AWQProcessor's calibration_dataset cannot be modified")

    def _get_layer_state(self, layer_index: int) -> _AWQLayerState:
        with self._layer_states_lock:
            state = self._layer_states.get(layer_index)
            if state is None:
                state = _AWQLayerState()
                self._layer_states[layer_index] = state
        return state

    def _initialize_sample_counts(self) -> None:
        total = 0
        dataset = getattr(self, "calibration_dataset", None)
        if dataset is None:
            self._nsamples_total = 0
            self.nsamples = 0
            return

        for row in dataset:
            if not isinstance(row, dict):
                continue
            input_ids = row.get("input_ids")
            if input_ids is None:
                continue
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() <= 1:
                    total += 1
                else:
                    total += input_ids.shape[0]
            else:
                try:
                    total += len(input_ids)
                except TypeError:
                    total += 1

        self._nsamples_total = total
        self.nsamples = total

    def _record_input_feature(self, module_name: str, feature: torch.Tensor) -> None:
        if feature.device.type != "cpu":
            feature = feature.detach().cpu()
        else:
            feature = feature.detach()

        with self.lock:
            entry = self.tasks.get(module_name)
            if entry is None:
                entry = {"inputs": []}
                self.tasks[module_name] = entry
            inputs_list = entry.setdefault("inputs", [])
            inputs_list.append(feature)

    def _capture_previous_subset_scale(self, previous_subset: Optional[Dict[str, NamedModule]]) -> Optional[float]:
        if not previous_subset:
            return None

        values: List[float] = []
        for named_module in previous_subset.values():
            weight = getattr(named_module.module, "weight", None)
            if weight is None:
                continue
            with torch.no_grad():
                values.append(float(weight.detach().abs().mean().item()))

        if not values:
            return None

        return float(sum(values) / len(values))

    def _layer_input_features(self, state: _AWQLayerState) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        for name in state.modules:
            entry = self.tasks.get(name) or {}
            tensors: List[torch.Tensor] = entry.get("inputs", [])  # type: ignore[arg-type]
            if not tensors:
                features[name] = torch.empty(0)
                continue
            try:
                features[name] = torch.cat(tensors, dim=0)
            except RuntimeError:
                features[name] = tensors[0]
            if features[name] is not None and features[name].numel() > 0:
                log.info(
                    "AWQProcessor: input feature `%s` shape=%s",
                    name,
                    tuple(features[name].shape),
                )
        return features

    def _refresh_forward_kwargs_from_cache(self) -> None:
        cache = getattr(self, "inputs_cache", None)
        if cache is None:
            return

        if getattr(cache, "attention_masks", None):
            mask = cache.attention_masks[-1]
            self._module_forward_kwargs["attention_mask"] = mask

        rotary = getattr(getattr(self.model, "model", self.model), "rotary_emb", None)
        pos_ids_cache = cache.position_ids[-1] if getattr(cache, "position_ids", None) else None
        hidden_cache = None
        if getattr(cache, "layer_inputs", None):
            last_inputs = cache.layer_inputs[-1]
            if last_inputs:
                hidden_cache = last_inputs[0]

        if rotary is not None and hidden_cache is not None:
            x_for_rotary = hidden_cache
            if x_for_rotary.dim() == 2:
                x_for_rotary = x_for_rotary.unsqueeze(0)
            seq_len = x_for_rotary.shape[1] if x_for_rotary.dim() >= 2 else x_for_rotary.shape[0]
            batch = x_for_rotary.shape[0] if x_for_rotary.dim() >= 2 else 1

            target_device = getattr(getattr(rotary, "inv_freq", None), "device", None)
            if target_device is not None and x_for_rotary.device != target_device:
                x_for_rotary = x_for_rotary.to(target_device)

            if pos_ids_cache is not None and pos_ids_cache.shape[-1] == seq_len:
                pos_for_rotary = pos_ids_cache.to(x_for_rotary.device)
            else:
                pos_for_rotary = torch.arange(seq_len, device=x_for_rotary.device, dtype=torch.long)
                pos_for_rotary = pos_for_rotary.unsqueeze(0).expand(batch, -1)

            self._module_forward_kwargs["position_ids"] = pos_for_rotary
            try:
                pe = rotary(x_for_rotary, pos_for_rotary)
                self._module_forward_kwargs["position_embeddings"] = pe
            except Exception:
                pass
        elif pos_ids_cache is not None:
            self._module_forward_kwargs["position_ids"] = pos_ids_cache

        if getattr(cache, "layer_input_kwargs", None):
            latest_kwargs = cache.layer_input_kwargs[-1] or {}
            for key, value in latest_kwargs.items():
                self._module_forward_kwargs[key] = value

    def _quantize_layer(self, layer_index: int, state: _AWQLayerState) -> None:
        with state.lock:
            if state.quantized:
                return

            layer_module = state.layer_module
            if layer_module is None and state.modules:
                sample_module = next(iter(state.modules.values()))
                layer_path = sample_module.full_name.rsplit(".", 1)[0]
                layer_module, _ = get_module_by_name_prefix(self.gptq_model.model, layer_path)
                state.layer_module = layer_module

            layer_module_ref = state.layer_module

            if layer_module_ref is None:
                raise RuntimeError(f"AWQProcessor: unable to resolve layer module for layer index {layer_index}")

            named_childs = dict(state.modules)

        input_feat = self._layer_input_features(state)
        missing = [name for name, tensor in input_feat.items() if tensor.numel() == 0]
        if missing:
            raise RuntimeError(
                f"AWQProcessor: missing calibration features for modules: {', '.join(missing)}."
            )

        module_kwargs_global = dict(self._module_forward_kwargs)

        setattr(self._scale_context, "layer_index", layer_index)
        setattr(self._scale_context, "prev_scale", state.previous_weight_scale)

        module_config = self.gptq_model.awq_get_modules_for_scaling(
            layer_module_ref,
            input_feat,
            module_kwargs_global,
        )

        if not module_config:
            log.warning(
                "AWQProcessor: no module configuration generated for layer index %s; skipping quantization.",
                layer_index,
            )
            with state.lock:
                state.quantized = True
                state.processed_subsets.clear()
                state.subset_total = None
                state.previous_weight_scale = None
            if hasattr(self._scale_context, "layer_index"):
                delattr(self._scale_context, "layer_index")
            if hasattr(self._scale_context, "prev_scale"):
                delattr(self._scale_context, "prev_scale")
            return

        sanitized_module_config: List[Dict] = []
        for entry in module_config:
            entry = dict(entry)
            inspect_module = entry.get("module2inspect") or layer_module_ref
            entry_kwargs = entry.get("kwargs") or module_kwargs_global
            entry["kwargs"] = self._sanitize_kwargs(entry_kwargs, inspect_module)
            sanitized_module_config.append(entry)

        scales_list = [
            self._search_best_scale(layer_module_ref, **layer)
            for layer in sanitized_module_config
        ]

        apply_scale(layer_module_ref, scales_list, input_feat_dict=input_feat)
        scales_list = append_str_prefix(
            scales_list,
            get_op_name(self.model, layer_module_ref) + ".",
        )

        clip_list = None
        if self.apply_clip:
            clip_list = self._search_best_clip(
                layer_module_ref,
                {name: named.module for name, named in named_childs.items()},
                input_feat,
            )
            apply_clip(layer_module_ref, clip_list)
            clip_list = append_str_prefix(
                clip_list,
                get_op_name(self.model, layer_module_ref) + ".",
            )

        if not self.export_compatible:
            start = time.time()
            self._apply_quant(layer_module_ref, named_childs, start, scales_list)

        with state.lock:
            state.quantized = True
            state.processed_subsets.clear()
            state.subset_total = None
            state.previous_weight_scale = None

        with self.lock:
            for name in named_childs:
                task_entry = self.tasks.get(name)
                if task_entry and "inputs" in task_entry:
                    task_entry["inputs"].clear()

        if hasattr(self._scale_context, "layer_index"):
            delattr(self._scale_context, "layer_index")
        if hasattr(self._scale_context, "prev_scale"):
            delattr(self._scale_context, "prev_scale")

    @torch.inference_mode()
    def _search_best_scale(
            self,
            module,
            prev_op,
            layers: List[nn.Linear],
            inp: torch.Tensor,
            module2inspect=None,
            kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.qcfg.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        del weight

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2  # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        del x_sum

        # [STEP 3]: Compute output of module
        module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
        global_kwargs = getattr(self, "_module_forward_kwargs", {})
        global_allowed_kwargs = self._sanitize_kwargs(global_kwargs, module2inspect)
        for key, value in global_allowed_kwargs.items():
            module_kwargs.setdefault(key, value)

        with ctx(torch.inference_mode()):
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)

        # [STEP 4]: Compute loss
        best_scales, loss = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
            loss
        )

    @torch.inference_mode()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())

            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.inference_mode()
    def _compute_best_clip(
            self,
            w: torch.Tensor,
            input_feat: torch.Tensor,
            n_grid=20,
            max_shrink=0.5,
            n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.qcfg.group_size if self.qcfg.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)
        del input_feat
        del org_out

        return best_max_val.squeeze(1)

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.qcfg.group_size > 0:
            assert org_w_shape[-1] % self.qcfg.group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({self.qcfg.group_size})!"
            w = w.reshape(-1, self.qcfg.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if self.qcfg.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.qcfg.bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.qcfg.bits - 1) - 1
            min_int = -(2 ** (self.qcfg.bits - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        prev_scale_hint = getattr(self._scale_context, "prev_scale", None)
        if prev_scale_hint is not None:
            w_mean = w_mean * float(prev_scale_hint)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)
            int_w_output = int_w_output.clip(torch.finfo(int_w_output.dtype).min, torch.finfo(int_w_output.dtype).max)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            log.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu(), best_error

    @torch.inference_mode()
    def _compute_loss(
            self,
            fp16_output: torch.Tensor,
            int_w_output: torch.Tensor,
            device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.inference_mode()
    def _module_forward(
            self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        target_device = None
        try:
            target_device = next(module.parameters()).device
        except StopIteration:
            target_device = None
        except Exception:
            target_device = None

        for key, value in list(module_kwargs.items()):
            if isinstance(value, torch.Tensor):
                if target_device is not None and value.device != target_device:
                    module_kwargs[key] = value.to(target_device)
            elif isinstance(value, (list, tuple)):
                converted = []
                changed = False
                for item in value:
                    if isinstance(item, torch.Tensor) and target_device is not None and item.device != target_device:
                        converted.append(item.to(target_device))
                        changed = True
                    else:
                        converted.append(item)
                if changed:
                    module_kwargs[key] = type(value)(converted)

        seq_len = None
        batch_dim = None
        if x.dim() >= 2:
            batch_dim = x.shape[0]
            seq_len = x.shape[1]
        else:
            batch_dim = 1
            seq_len = x.shape[0]

        supports_position_ids = False
        supports_position_embeddings = False
        try:
            signature = inspect.signature(module.forward).parameters
            supports_position_ids = "position_ids" in signature
            supports_position_embeddings = "position_embeddings" in signature
        except (ValueError, TypeError):
            pass

        rotary = getattr(getattr(self.model, "model", self.model), "rotary_emb", None)
        if seq_len is not None and rotary is not None and supports_position_embeddings:
            pos_ids = module_kwargs.get("position_ids") if supports_position_ids else None
            if not supports_position_ids:
                pos_ids = None
            if pos_ids is None or pos_ids.shape[-1] != seq_len:
                pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
                if x.dim() >= 2:
                    pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_values
                pos_for_rotary = pos_values
            else:
                pos_for_rotary = pos_ids.to(target_device or pos_ids.device)
                if supports_position_ids:
                    module_kwargs["position_ids"] = pos_for_rotary

            x_for_rotary = x if target_device is None else x.to(target_device)
            module_kwargs["position_embeddings"] = rotary(x_for_rotary, pos_for_rotary)
        elif supports_position_ids and seq_len is not None and "position_ids" not in module_kwargs:
            pos_values = torch.arange(seq_len, device=target_device or x.device, dtype=torch.long)
            if x.dim() >= 2:
                pos_values = pos_values.unsqueeze(0).expand(batch_dim, -1)
            module_kwargs["position_ids"] = pos_values

        if self._quant_batch_size is None or self._quant_batch_size <= 1:
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
            return module_output

        def _slice_value(val, length):
            if isinstance(val, torch.Tensor) and val.shape[0] == module_kwargs.get("position_ids", val).shape[0]:
                return val[:length]
            if isinstance(val, torch.Tensor) and val.shape[0] != length:
                return val
            if isinstance(val, torch.Tensor):
                return val[:length]
            if isinstance(val, (list, tuple)):
                sliced = [_slice_value(item, length) for item in val]
                return type(val)(sliced)
            return val

        outputs = []
        for x_partial in torch.split(x, self._quant_batch_size, dim=0):
            partial_kwargs = {
                key: _slice_value(value, x_partial.shape[0])
                for key, value in module_kwargs.items()
            }
            partial_output = module(x_partial, **partial_kwargs)
            if isinstance(partial_output, tuple):
                partial_output = partial_output[0]
            outputs.append(partial_output)

        module_output = torch.cat(outputs, dim=0)

        return module_output

    def _apply_quant(self, module, named_linears: Dict[str, NamedModule], start_time, scales_list):
        for name, named_module in named_linears.items():
            self.pb.title(f"Quantizing {named_module.name} in layer ").draw()
            linear_layer = named_module.module
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            tp_info = named_module.state.get("tp_pad_info")
            pad_cols = 0
            original_cols = linear_layer.weight.data.shape[1]
            if isinstance(tp_info, dict):
                pad_cols = int(tp_info.get("pad_cols", 0) or 0)
                original_cols = int(tp_info.get("original_columns", original_cols))

            weight_for_quant = linear_layer.weight.data
            if pad_cols:
                pad = weight_for_quant.new_zeros(weight_for_quant.shape[0], pad_cols)
                weight_for_quant = torch.cat((weight_for_quant, pad), dim=1)

            wq, scales, zeros = self.pseudo_quantize_tensor(
                weight_for_quant
            )

            if pad_cols:
                wq = wq[:, :original_cols]
                if self.qcfg.group_size > 0:
                    valid_groups = max(1, math.ceil(original_cols / self.qcfg.group_size))
                    scales = scales[:, :valid_groups]
                    if zeros is not None:
                        zeros = zeros[:, :valid_groups]
                else:
                    scales = scales[:, :original_cols]
                    if zeros is not None:
                        zeros = zeros[:, :original_cols]
            if self.calculate_w_wq_diff:
                if named_module.weight.data.dtype == torch.float16:
                    # diff in float16
                    w_wq_diff = linear_layer.weight.data - wq
                else:
                    # diff in float32
                    w_wq_diff = linear_layer.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)

                named_module.state.update({
                    "w_wq_diff": w_wq_diff,
                })

            named_module.state.update({
                "wq": wq,  # fp16, quantized weight but not int4 (packed qweight)
            })

            if isinstance(tp_info, dict):
                named_module.state.pop("tp_pad_info", None)

            linear_layer.weight.data = wq

            if self.version == "gemm":
                scales = scales.t().contiguous()
                if zeros is not None:
                    zeros = zeros.t().contiguous()
                q_linear_module = WQLinear_GEMM

            elif self.version == "gemv":
                q_linear_module = WQLinear_GEMV

            elif self.version == "marlin":
                q_linear_module = WQLinear_Marlin

            elif self.version == "gemv_fast":
                q_linear_module = WQLinear_GEMVFast

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.qcfg.bits,
                group_size=self.qcfg.group_size,
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)

            # records
            duration = time.time() - start_time

            avg_loss_value = None
            for _, layer_names, _, loss in scales_list:
                if any(named_module.name in layer_name for layer_name in layer_names):
                    avg_loss_value = loss
                    break

            if avg_loss_value is None:
                # Scaling not applied for this layer in AWQ; no meaningful loss metric.
                loss_summary = "not applicable"
            else:
                loss_summary = f"{avg_loss_value:.10f}"

            # TODO "loss" and "nsamples" may not be consistent with the semantics of gptq quantization.
            stat = {
                PROCESS_LOG_NAME: self.name(),
                PROCESS_LOG_LAYER: named_module.layer_index,
                PROCESS_LOG_MODULE: named_module.name,
                MODULE_FEATURE_COLUMN: self.module_feature_summary(named_module),
                DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(named_module),
                QUANT_LOG_LOSS: loss_summary,
                QUANT_LOG_NSAMPLES: f"{self._nsamples_total}",
                # QUANT_LOG_DAMP: f"{damp_percent:.5f}",
                PROCESS_LOG_TIME: f"{duration:.3f}",
                # PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
                PROCESS_USED_MEMORY: self.device_memory_report(),
            }
            with self.lock:
                self.module_names.append(f"layer-{named_module.layer_index}-{named_module.name}")
                self.log.append(stat)

            # Log the new row
            self.log_new_row(stat)

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def preprocess(self, module: NamedModule, fail_safe: Optional[bool] = None):
        layer_state = self._get_layer_state(module.layer_index)
        with layer_state.lock:
            layer_state.modules[module.name] = module
            layer_module_ref = module.state.get("layer_module")
            if layer_module_ref is not None:
                layer_state.layer_module = layer_module_ref
        with self.lock:
            entry = self.tasks.get(module.name)
            if entry is None:
                self.tasks[module.name] = {"inputs": []}
            else:
                entry.setdefault("inputs", [])

    def is_skipped(self, module: NamedModule) -> bool:
        return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            if not inp:
                return
            feature = inp[0]
            if isinstance(feature, (tuple, list)) and feature:
                feature = feature[0]
            self._record_input_feature(name, feature)
        return hook

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        self._refresh_forward_kwargs_from_cache()
        layer_index = module.layer_index
        state = self._get_layer_state(layer_index)

        with state.lock:
            if subset is not None:
                state.modules.update(subset)
                if state.layer_module is None:
                    for candidate in subset.values():
                        layer_module_ref = candidate.state.get("layer_module")
                        if layer_module_ref is not None:
                            state.layer_module = layer_module_ref
                            break

            if subset_total is not None:
                state.subset_total = subset_total
            if subset_index is not None:
                state.processed_subsets.add(subset_index)

            if previous_subset:
                state.previous_weight_scale = self._capture_previous_subset_scale(previous_subset)

            should_quantize = (
                state.subset_total is not None
                and len(state.processed_subsets) >= state.subset_total
                and not state.quantized
            )

        if should_quantize:
            self._quantize_layer(layer_index, state)

    # submodule_finalized is called in reverse after all next sequential processes are called
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        # generate complete, safe to move to cpu
        module.weight.data = move_to(module.weight.data, device=CPU) # large weights is slow to init on cpu
        module.state.pop("w", None) # no need for original weights now

    def finalize(self, model: BaseQModel, **kwargs):
        if model.quantize_config.format == FORMAT.GEMM:
            model.qlinear_kernel = AwqGEMMQuantLinear
        elif model.quantize_config.format == FORMAT.GEMV:
            model.qlinear_kernel = AwqGEMVQuantLinear
        elif model.quantize_config.format == FORMAT.GEMV_FAST:
            model.qlinear_kernel = AwqGEMVFastQuantLinear
        elif model.quantize_config.format == FORMAT.MARLIN:
            model.qlinear_kernel = AwqMarlinQuantLinear
        else:
            raise Exception(f"unkown format: {model.quantize_config.format}")

        # set quantized state
        model.quantized = True

        model.quantize_config.quant_method = METHOD.AWQ

        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    @classmethod
    def name(cls) -> str:
        return "awq"
