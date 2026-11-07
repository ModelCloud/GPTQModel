# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
import typing

import torch
from transformers import PreTrainedModel

from ...models.module_plan import ModulePlan
from ...utils.logger import setup_logger
from ...utils.model import get_module_by_name_prefix
from ...utils.torch import torch_empty_cache
from .hadamard_utils import apply_exact_had_to_linear, get_hadK, random_hadamard_matrix


log = setup_logger()


def fuse_ln_linear(
        layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def reset_ln(ln):
    W_norm = ln.weight.data
    ln.weight.data = torch.ones_like(W_norm)


def _relative_roles(plan: ModulePlan, parent: str, role: str) -> list[str]:
    return list(plan.child_roles(parent, role))


def _resolve_role(layer, plan: ModulePlan, parent: str, role: str):
    names = _relative_roles(plan, parent, role)
    if len(names) != 1:
        raise ValueError(f"Rotation requires exactly one `:{role}` projection under {parent or '<layer>'}, found {names}")
    path = f"{parent}.{names[0]}" if parent else names[0]
    try:
        return layer.get_submodule(path)
    except AttributeError as exc:
        raise ValueError(f"Rotation declared `:{role}` module is missing: {path}") from exc


def _default_rotation_plan() -> ModulePlan:
    return ModulePlan.from_tree([
        "model", "layers", "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1:o"),
            "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ])


def _rotation_parents(plan: ModulePlan):
    attn_parents = {
        spec.parent for spec in plan.specs
        if "q" in spec.roles and spec.parent and "." not in spec.parent
    }
    mlp_parents = {
        spec.parent for spec in plan.specs
        if "gate" in spec.roles and spec.parent and "." not in spec.parent
    }
    if len(attn_parents) != 1 or len(mlp_parents) != 1:
        raise ValueError("Rotation requires one declared attention parent and one MLP parent in module_tree")
    return next(iter(attn_parents)), next(iter(mlp_parents))


def fuse_layer_norms(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str,
                     plan: ModulePlan = None):
    layers , _ = get_module_by_name_prefix(model, layers_node)
    plan = plan or _default_rotation_plan()
    norm_roles = tuple(spec.name for spec in plan.specs if not spec.parent and "norm" in spec.flags)
    if len(norm_roles) != 2:
        raise ValueError(f"Rotation requires two declared layer norms, found {list(norm_roles)}")
    input_norm, post_norm = norm_roles
    attn_parent, mlp_parent = _rotation_parents(plan)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.get_submodule(post_norm),
            [_resolve_role(layer, plan, mlp_parent, "up"), _resolve_role(layer, plan, mlp_parent, "gate")],
        )
        fuse_ln_linear(
            layer.get_submodule(input_norm),
            [
                _resolve_role(layer, plan, attn_parent, "q"),
                _resolve_role(layer, plan, attn_parent, "k"),
                _resolve_role(layer, plan, attn_parent, "v"),
            ],
        )
        reset_ln(layer.get_submodule(post_norm))
        reset_ln(layer.get_submodule(input_norm))

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_embeddings(model, embedding_name: str = None) -> list[torch.nn.Module]:
    if embedding_name is not None:
        return [model.get_submodule(embedding_name)]
    getter = getattr(model, "get_input_embeddings", None)
    return [getter()] if callable(getter) else [model.model.embed_tokens]


def rotate_embeddings(model, Q, device, embedding_name: str = None) -> None:
    # Rotate the embeddings.
    for W in get_embeddings(model, embedding_name=embedding_name):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, device, plan: ModulePlan = None, attn_parent: str = None) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    if plan is None or attn_parent is None:
        raise ValueError("Rotation requires an explicit attention parent from module_tree")
    inputs = [_resolve_role(layer, plan, attn_parent, role) for role in ("q", "k", "v")]
    for W in inputs:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, device, plan: ModulePlan = None, attn_parent: str = None) -> None:
    # Rotate output matrix of the self-attention layer.
    if plan is None or attn_parent is None:
        raise ValueError("Rotation requires an explicit attention parent from module_tree")
    W = _resolve_role(layer, plan, attn_parent, "o")
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, device, plan: ModulePlan = None, mlp_parent: str = None):
    # Rotate the MLP input weights.
    if plan is None or mlp_parent is None:
        raise ValueError("Rotation requires an explicit MLP parent from module_tree")
    mlp_inputs = [_resolve_role(layer, plan, mlp_parent, role) for role in ("up", "gate")]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, device, plan: ModulePlan = None, mlp_parent: str = None):
    # Rotate the MLP output weights and bias.
    if plan is None or mlp_parent is None:
        raise ValueError("Rotation requires an explicit MLP parent from module_tree")
    W = _resolve_role(layer, plan, mlp_parent, "down")
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    apply_exact_had_to_linear(W, had_dim=-1, output=False)  # apply exact (inverse) hadamard on the weights of mlp output

    # down_proj receives the elementwise SiLU*up activation, so the Hadamard
    # fused into the weight must be re-applied to the activation at both
    # calibration and inference time.
    W.online_full_had = True
    W.online_partial_had = False
    W.had_dim = -1
    had_K, K = get_hadK(W.in_features)
    W.had_K = had_K
    W.K = K

    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q, device, lm_head_name) -> None:
    # Rotate the head.
    W, _ = get_module_by_name_prefix(model, lm_head_name)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, plan: ModulePlan = None, attn_parent: str = None):
    if plan is None or attn_parent is None:
        raise ValueError("Rotation requires an explicit attention parent from module_tree")
    v_proj = _resolve_role(layer, plan, attn_parent, "v")
    o_proj = _resolve_role(layer, plan, attn_parent, "o")
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


@torch.inference_mode()
def rotate_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                 layers_node: str, Q=None, plan: ModulePlan = None, embedding_name: str = None):
    Q = (
        get_orthogonal_matrix(
            model.config.hidden_size, rotate_mode, device
        )
        if Q is None
        else Q
    )
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, Q, device, embedding_name=embedding_name)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    plan = plan or _default_rotation_plan()
    attn_parent, mlp_parent = _rotation_parents(plan)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device, plan=plan, attn_parent=attn_parent)
        rotate_attention_output(layer, Q, device, plan=plan, attn_parent=attn_parent)
        rotate_mlp_input(layer, Q, device, plan=plan, mlp_parent=mlp_parent)
        rotate_mlp_output(layer, Q, device, plan=plan, mlp_parent=mlp_parent)
        rotate_ov_proj(layer, num_heads, head_dim, plan=plan, attn_parent=attn_parent)
    return model, Q
