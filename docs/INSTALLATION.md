# Installation

On Linux and Windows, GPTQModel can be installed through pre-built wheels for specific PyTorch versions:

| GPTQModel version | CUDA/ROCm version | Installation                                                                                               | Built against PyTorch |
|------------------|-------------------|------------------------------------------------------------------------------------------------------------|-----------------------|
| latest (0.7.1)   | CUDA 11.8         | `pip install gptqmodel --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/cu118/`          | 2.2.1+cu118           |
| latest (0.7.1)   | CUDA 12.1         | `pip install gptqmodel`                                                                                    | 2.2.1+cu121           |
| latest (0.7.1)   | ROCm 5.7          | `pip install gptqmodel --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/rocm571/`        | 2.2.1+rocm5.7         |
| 0.7.0   | CUDA 11.8         | `pip install gptqmodel --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/cu118/`          | 2.2.0+cu118           |
| 0.7.0   | CUDA 12.1         | `pip install gptqmodel`                                                                                    | 2.2.0+cu121           |
| 0.7.0   | ROCm 5.7          | `pip install gptqmodel --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/rocm571/`        | 2.2.0+rocm5.7         |
| 0.6.0            | CUDA 11.8         | `pip install gptqmodel==0.6.0 --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/cu118/`   | 2.1.1+cu118           |
| 0.6.0            | CUDA 12.1         | `pip install gptqmodel==0.6.0`                                                                             | 2.1.1+cu121           |
| 0.6.0            | ROCm 5.6          | `pip install gptqmodel==0.6.0 --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/rocm561/` | 2.1.1+rocm5.6         |
| 0.5.1            | CUDA 11.8         | `pip install gptqmodel==0.5.1 --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/cu118/`   | 2.1.0+cu118           |
| 0.5.1            | CUDA 12.1         | `pip install gptqmodel==0.5.1`                                                                             | 2.1.0+cu121           |
| 0.5.1            | ROCm 5.6          | `pip install gptqmodel==0.5.1 --extra-index-url https://huggingface.github.io/gptqmodel-index/whl/rocm561/` | 2.1.0+rocm5.6         |

GPTQModel is not available on macOS.