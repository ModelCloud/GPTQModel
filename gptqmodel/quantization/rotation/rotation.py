# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
import typing

import torch
from transformers import PreTrainedModel

from ...utils.logger import setup_logger
from ...utils.model import get_module_by_name_prefix
from ...utils.torch import torch_empty_cache
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix

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


def fuse_layer_norms(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    layers = get_module_by_name_prefix(model, layers_node)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    pre_head_layernorm = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head = get_module_by_name_prefix(model, lm_head_name)

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


def get_embeddings(model) -> list[torch.nn.Module]:
    return [model.model.embed_tokens]


def rotate_embeddings(model, Q, device) -> None:
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, device) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, device) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q, device, lm_head_name) -> None:
    # Rotate the head.
    W = get_module_by_name_prefix(model, lm_head_name)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


@torch.inference_mode()
def rotate_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                 layers_node: str, Q=None):
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

    rotate_embeddings(model, Q, device)
    rotate_head(model, Q, device, lm_head_name)
    torch_empty_cache()
    layers = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
        rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q
