import copy
import math

import gptqmodel_exllama_eora
import torch
# from eora_test import fused_concurrent, fused_sequential, cublas_reference, gptq_gemm_eora, gptq_gemm
from gptqmodel_exllama_eora import gptq_gemm, gptq_gemm_eora
from gptqmodel_exllama_kernels import make_q4, q4_matmul
from safetensors import safe_open

# model_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/"
# lora_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors"

target =  'model.layers.6.self_attn.q_proj'
eora_tensors = {}
with safe_open("/mnt/home/shihyangl/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        if target in k:
            eora_tensors[k] = f.get_tensor(k)
# print(eora_tensors)

qw_tensors = {}
with safe_open("/mnt/home/shihyangl/llama3.2-1b-4bit-group128/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        if target in k:
            qw_tensors[k] = f.get_tensor(k)




m = 1
k = eora_tensors[f'{target}.lora_A.weight'].shape[1]
n = eora_tensors[f'{target}.lora_B.weight'].shape[0]
r = 128



bit = 4
use_exllama = True

x = torch.rand((m, k), device='cuda', dtype=torch.float16)
eora_a = eora_tensors[f'{target}.lora_A.weight'].to('cuda:0').T
eora_b = torch.clone(eora_tensors[f'{target}.lora_B.weight'].T, memory_format=torch.contiguous_format)
# torch.zeros((r, n),device='cuda'
# eora_b.data = torch.transpose(eora_tensors[f'{target}.lora_B.weight'], 0, 1)
# eora_b = eora_tensors[f'{target}.lora_B.weight'].to('cuda:0').T



# print(eora_b)
# print(eora_b)

# eora_b = torch.rand((r, n), device='cuda', dtype=torch.float16) / (100 * 4)
# eora_b = torch.normal(-2.7120113372802734e-05, 0.0248565673828125, size=(r, n), device='cuda', dtype=torch.float16)

# eora_b = torch.normal(-2.7120113372802734e-05, 0.0248565673828125, size=(n, r), device='cuda', dtype=torch.float16).T

# eora_b.data = torch.normal(-2.7120113372802734e-05, 0.0248565673828125, size=(n, r), device='cuda', dtype=torch.float16).T


# eora_b[[0,2,4,6,8],:] = 0
# eora_b_mean = eora_b.mean()
# eora_b_std = eora_b.std()
# sample_range = 2040
# sample_idx = torch.randint(0,2048,(1,sample_range)).flatten()
# eora_b[:,sample_idx] = 0

# list_range = list(range(1,128))
# eora_b[list_range,:] = 0
# print(eora_b)


# print(eora_b.shape)
# print(eora_b)
# print(eora_b)
# print(f"eora_b max {eora_b.max()}")
# print(f"eora_b max {eora_b.min()}")
# print(f"eora_b mean {eora_b.mean()}")
# print(f"eora_b std {eora_b.std()}")

# real_eora_a = eora_tensors[f'{target}.lora_A.weight'].to('cuda:0').T
# real_eora_b = eora_tensors[f'{target}.lora_B.weight'].to('cuda:0').T
# print(f"real eora_a max {real_eora_a.max()}")
# print(f"real eora_a min {real_eora_a.min()}")
# print(f"real eora_a mean {real_eora_a.mean()}")
# print(f"real eora_a std {real_eora_a.std()}")
# print(f"real eora_b max {real_eora_b.max()}")
# print(f"real eora_b min {real_eora_b.min()}")
# print(f"real eora_b mean {real_eora_b.mean()}")
# print(f"real eora_b std {real_eora_b.std()}")

# eora_a = torch.randn((k, r), device='cuda', dtype=torch.float16) / (100 * 4)
# eora_b = torch.randn((r, n), device='cuda', dtype=torch.float16) / (5)

# eora_a = torch.rand((k, r), device='cuda', dtype=torch.float16) / (100 * 4)
# eora_b = torch.rand((r, n), device='cuda', dtype=torch.float16) / (100 * 4)
# print(f"dummy eora_a max {eora_a.max()}")
# print(f"dummy eora_a max {eora_a.min()}")
# print(f"dummy eora_a mean {eora_a.mean()}")
# print(f"dummy eora_a std {eora_a.std()}")

# print(f"dummy eora_b max {eora_b.max()}")
# print(f"dummy eora_b max {eora_b.min()}")
# print(f"dummy eora_b mean {eora_b.mean()}")
# print(f"dummy eora_b std {eora_b.std()}")

gptq_groups = 128
# weight = qw_tensors[f'{target}.qweight'].to('cuda:0')
# zeros = qw_tensors[f'{target}.qzeros'].to('cuda:0')
# scales = qw_tensors[f'{target}.scales'].to('cuda:0')
# idx = qw_tensors[f'{target}.g_idx'].to('cuda:0')

weight = torch.zeros_like(qw_tensors[f'{target}.qweight'], device='cuda', dtype=torch.int32)
zeros =  torch.zeros_like(qw_tensors[f'{target}.qzeros'], device='cuda', dtype=torch.int32)
scales =  torch.zeros_like(qw_tensors[f'{target}.scales'], device='cuda', dtype=torch.float16)
idx = qw_tensors[f'{target}.g_idx'].to('cuda:0')


pack_dtype_bits = 32
bits = 4

pack_factor = pack_dtype_bits // bits
wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32).unsqueeze(0).to("cuda:0")
maxq = 2 ** bits - 1
num_itr = idx.shape[0] // 2048
def dequantize_weight(bits, wf, qweight, qzeros, maxq, scales, g_idx, dequant_dtype = torch.int8, num_itr: int=1):
    if bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, pack_factor),
            wf.unsqueeze(0),
        ).to(dequant_dtype)
        zeros = torch.bitwise_and(zeros, maxq).reshape(scales.shape)

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
                wf.unsqueeze(-1),
            ).to(dequant_dtype),
            maxq
        )
    elif bits == 3:
        zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(scales.shape)

        weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if num_itr == 1:
        weights = scales[g_idx.long()] * (weight - zeros[g_idx.long()])
    else:
        num_dim = g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = scales[:, i * num_dim: (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim: (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim: (i + 1) * num_dim]
            g_idx_i = g_idx[i * num_dim: (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights = torch.cat(weights, dim=1)

    return weights

def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    gptqmodel_exllama_eora.gptq_shuffle(q_weight, q_perm, bit)

## exllama GPTQModel
def exllama_output( x , pack_dtype_bits, bits, group_size, qweight, qzeros, scales,g_idx, out_features, in_features ):
    NON_TENSOR = torch.empty((1, 1), device="meta")
    def ext_make_q4(qweight, qzeros, scales, g_idx, device):
        """Construct Q4Matrix, return handle"""
        return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else NON_TENSOR, device)


    def ext_q4_matmul(x, q4, q4_width):
        """Matrix multiplication, returns x @ q4"""
        outshape = x.shape[:-1] + (q4_width,)
        x = x.view(-1, x.shape[-1])
        output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

        q4_matmul(x, q4, output)

        return output.view(outshape)

    original_out_features = out_features
    original_in_features = in_features

    # auto pad
    group_size = group_size if group_size != -1 else in_features
    out_features = out_features + (-out_features % 32)
    in_features = in_features + (-in_features % group_size)
    in_features_padding_size = in_features - original_in_features
    in_features_padding_shape = (0, in_features_padding_size)

    if out_features != original_out_features or in_features != original_in_features:
        qweight.resize_(in_features // pack_dtype_bits * bits, out_features)
        qzeros.resize_(
            math.ceil(in_features / group_size),
            out_features // pack_dtype_bits * bits
        )
        scales.resize_((math.ceil(in_features / group_size), out_features), )
        g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32, device=g_idx.device)

    width = qweight.shape[1]

    # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
    q4 = ext_make_q4(
        qweight,
        qzeros,
        scales,
        None,
        qweight.device.index,
    )

    out = ext_q4_matmul(x, q4, width)

    return out



ax = x @ eora_a
def test_eora_kernel():

    # zeros_copy = zeros.clone() + 0b00010001000100010001000100010001
    # exllama_out = exllama_output( x , pack_dtype_bits, bits, group_size=gptq_groups, qweight = weight, qzeros = zeros_copy, scales = scales,g_idx =idx , out_features = n, in_features = k)
    # exllama_out = exllama_out  + (ax @ eora_b)
    
    # deq_weight = dequantize_weight(bits=4, wf = wf, qweight=weight, qzeros=zeros_copy, maxq=maxq, scales=scales, g_idx=idx, dequant_dtype=torch.int8, num_itr=num_itr)
    # torch_kernel_out = torch.matmul(x, deq_weight).reshape(m,512) + (ax @ eora_b)

    idx.data = torch.argsort(idx).to(torch.int32)
    gptq_shuffle(weight, idx, bits)
    
    ## I confirmed this part to be identical to that of test_kernel_output.py


    out_shape = x.shape[:-1] + (weight.shape[-1],)
    reshaped_x = x.reshape(-1, x.shape[-1])

    gptq_pytorch_out = gptq_gemm(reshaped_x, weight, zeros, scales, idx, use_exllama, bit) + (ax @ eora_b)

    gptq_eora_fused_out = gptq_gemm_eora(reshaped_x, weight, zeros, scales, idx, use_exllama, bit, ax, eora_b)
    torch.set_printoptions(precision=6)
    # print("gptq exllama kernel out: ")
    # print(exllama_out[0][:10])
    # print("gptq torch kernel out: ")
    # print(torch_kernel_out[0][:10])
    # I want this to match the above two output
    print("vllm exllama_pytorch_out: ")
    print(gptq_pytorch_out[0][:10])

    print("vllm exllama_eora_fused_out: ")
    print(gptq_eora_fused_out[0][:10])
    torch.testing.assert_close(gptq_pytorch_out, gptq_eora_fused_out, rtol=0.05, atol=0.5)  # 5 % relative tolerance, 0.5 absolute tolerance

test_eora_kernel()