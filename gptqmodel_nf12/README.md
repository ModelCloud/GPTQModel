# GPT-QModel NF12 inference API

`gptqmodel_nf12` is the lightweight NF12 API shipped in the GPT-QModel wheel.
It imports without initializing the GPT-QModel quantization framework and builds
the CUDA operators from `gptqmodel_ext/nf12` on first use. It does not need a
separate installation of the standalone `nf12` package.

The BF16 weight format and native linear, embedding, and MoE kernels come from
[dphnAI/nf12](https://github.com/dphnAI/nf12). The public Python API follows
that package, with `linear_fp32` added for FP32 inference from BF16 checkpoint
weights. `linear_fp32` decodes values in registers and never keeps a dense copy
of the weight. Passing a one-dimensional `rows` tensor computes selected output
rows for shortlists.

```python
import gptqmodel_nf12 as nf12

packed = nf12.encode(bf16_cuda_weight)  # [N, K], N % 16 == 0, K % 64 == 0
embeddings = nf12.embedding(token_ids, packed)
bf16_logits = nf12.linear(bf16_hidden, packed)
fp32_scores = nf12.linear_fp32(fp32_hidden, packed)
shortlist = nf12.linear_fp32(fp32_hidden, packed, rows=selected_rows)
markov_scores = nf12.linear_bf16_fp32(bf16_embedding, packed)
```

`NF12Linear`, `NF12Embedding`, `encode_moe`, and `fused_moe` are also exported.
`NF12Linear` accepts BF16 or FP32 activations; its output follows the activation
dtype. Encoding and decoding are bit exact for BF16 weights. Small-row FP32
projection changes the reduction order relative to cuBLAS, so selected token
IDs and routing decisions must be qualified by each serving application before
promotion. For larger batches `linear_fp32` decodes a temporary dense weight
for the current cuBLAS call and releases it before returning. The test suite
compares output and decisions against dense Torch oracles on SM80.
`linear_bf16_fp32` keeps BF16 inputs exact while returning FP32 accumulations;
its short-K multirow path uses packed Tensor Core tiles for Markov heads.

The format currently covers matrices with aligned dimensions. Norm vectors,
biases, and other one-dimensional checkpoint tensors are outside its scope.
