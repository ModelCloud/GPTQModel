# GPTQ-eora

## Introduction

Draft implementation of 4-bit CUDA kernel for "EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation" (https://arxiv.org/abs/2410.21271) paper. 
The implementation is bootstrapped from vllm implementation of gptq: https://github.com/vllm-project/vllm/tree/f0ef37233ea0ba5251edaea7362984110411e7eb/csrc/quantization/gptq 
by forking `gemm_half_q_half_gptq_4bit_kernel` into `gemm_half_q_half_gptq_4bit_kernel_eora`, which accepts additional input: `Ax` and `B` matrices along with LORA rank.

To see the delta between the proposed and the original implementation one can diff `q_gemm.cu` and `q_gemm_original.cu` ignoring whitespaces and blank lines. 

## Getting started
- install miniconda https://docs.anaconda.com/miniconda/install/ 
- `conda create -n test-eora  python=3.12 pip`
- `conda activate test-eora`
- `conda install -c conda-forge libstdcxx-ng` # to avoid ` version `GLIBCXX_3.4.32' not found` error
- `pip install  -r requirements.txt` 
- `pip install .`
- `pytest test_eora.py` # correctness test
- `python3 benchmark.py` # benchmarking

### Benchmarking results:
Speedup ranging between 2.05x and 1.09x is observed for batch sizes ranging from 1 to 8 on a single RTX 3090 GPU. 
The baseline is `gptq kernel + pytorch for LORA` is compared with `gptq eora kernel`.
```bash
gptq-eora ➜ python3 ./benchmark.py                                                                                           t    1
pytorch baseline: 0.10021328926086426 msec
pytorch LORA baseline: 0.11120986938476562 msec
pytorch baseline: 0.07351875305175781 msec
pytorch LORA baseline: 0.0958395004272461 msec
gptq: 0.018501758575439453 msec
gptq + pytorch for LORA: 0.04210519790649414 msec
gptq eora kernel: 0.020452022552490234 msec
gptq+pytorch/fused_kernel ratio for batch size 1: 2.0587302697535614
pytorch_lora/fused_kernel ratio for batch size 1: 4.686064675572964

pytorch baseline: 0.09366106986999512 msec
pytorch LORA baseline: 0.12542033195495605 msec
gptq: 0.019073963165283203 msec
gptq + pytorch for LORA: 0.043236494064331055 msec
gptq eora kernel: 0.02179884910583496 msec
gptq+pytorch/fused_kernel ratio for batch size 2: 1.9834301276372346
pytorch_lora/fused_kernel ratio for batch size 2: 5.7535299843597905

pytorch baseline: 0.09362173080444336 msec
pytorch LORA baseline: 0.12170100212097168 msec
gptq: 0.019705533981323242 msec
gptq + pytorch for LORA: 0.0429532527923584 msec
gptq eora kernel: 0.023361921310424805 msec
gptq+pytorch/fused_kernel ratio for batch size 3: 1.8386010389133252
pytorch_lora/fused_kernel ratio for batch size 3: 5.209374712972129

pytorch baseline: 0.09506535530090332 msec
pytorch LORA baseline: 0.1078331470489502 msec
gptq: 0.020968198776245117 msec
gptq + pytorch for LORA: 0.04309487342834473 msec
gptq eora kernel: 0.025162220001220703 msec
gptq+pytorch/fused_kernel ratio for batch size 4: 1.7126816881123388
pytorch_lora/fused_kernel ratio for batch size 4: 4.285518012469442

pytorch baseline: 0.09542036056518555 msec
pytorch LORA baseline: 0.1076815128326416 msec
gptq: 0.022510766983032227 msec
gptq + pytorch for LORA: 0.052427053451538086 msec
gptq eora kernel: 0.028439998626708984 msec
gptq+pytorch/fused_kernel ratio for batch size 5: 1.843426722331204
pytorch_lora/fused_kernel ratio for batch size 5: 3.7862699730060525

pytorch baseline: 0.09557318687438965 msec
pytorch LORA baseline: 0.10774064064025879 msec
gptq: 0.025467395782470703 msec
gptq + pytorch for LORA: 0.04637646675109863 msec
gptq eora kernel: 0.033232927322387695 msec
gptq+pytorch/fused_kernel ratio for batch size 6: 1.395497492628543
pytorch_lora/fused_kernel ratio for batch size 6: 3.241984661630401

pytorch baseline: 0.09484624862670898 msec
pytorch LORA baseline: 0.10790395736694336 msec
gptq: 0.02785944938659668 msec
gptq + pytorch for LORA: 0.04564833641052246 msec
gptq eora kernel: 0.03971362113952637 msec
gptq+pytorch/fused_kernel ratio for batch size 7: 1.149437777284161
pytorch_lora/fused_kernel ratio for batch size 7: 2.717051587611289

pytorch baseline: 0.0950167179107666 msec
pytorch LORA baseline: 0.10870051383972168 msec
gptq: 0.029795169830322266 msec
gptq + pytorch for LORA: 0.044673919677734375 msec
gptq eora kernel: 0.04362607002258301 msec
gptq+pytorch/fused_kernel ratio for batch size 8: 1.0240188872068685
pytorch_lora/fused_kernel ratio for batch size 8: 2.4916412086500785

pytorch baseline: 0.09513998031616211 msec
pytorch LORA baseline: 0.10854911804199219 msec
gptq: 0.04927778244018555 msec
gptq + pytorch for LORA: 0.05824875831604004 msec
gptq eora kernel: 0.06363630294799805 msec
gptq+pytorch/fused_kernel ratio for batch size 9: 0.9153385036154509
pytorch_lora/fused_kernel ratio for batch size 9: 1.7057734816979506
```


