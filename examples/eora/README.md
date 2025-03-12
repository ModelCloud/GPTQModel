# EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
EoRA is a training-free method that uses a calibration dataset to build low-rank matrices aimed at mitigating quantization errors and enhancing the performance of quantized models. Its generation takes about the same time as GPTQ.
For more details, please refer to the paper: https://arxiv.org/abs/2410.21271.


## Calibration data
EoRA’s major advantage is that it can enhance the accuracy of quantized models on various downstream tasks without training, simply by using a small amount of task-specific data as a calibration set. For instance, to improve performance on MMLU, you can employ the MMLU validation set as calibration data when generating EoRA. Additionally, EoRA can boost a quantized model’s overall quality by using the same calibration data as GPTQ.

For examples of how to create these calibration sets, see `construct_c4` in `GPTQModel/examples/eora/eora_calibration_data_construction.py` for a general-purpose setup using the C4 dataset, and `construct_mmlu` in the same file for task-specific calibration data.

## EoRA generation
There are two ways to produce EoRA. The first is to generate it simultaneously with GPTQ during the quantization process. The second is to take an already GPTQ-quantized model and apply EoRA generation on top of it.

### First option: Generate EoRA and the GPTQ model together during quantization.
Below is an example of using C4 as calibration data for generating EoRA of rank 64 alongside 4-bits GPTQ quantization of meta-llama/Llama-3.2-3B. To further improve the accuracy on MMLU, set mmlu for eora_dataset.
```shell
python GPTQModel/examples/eora/eora_generation.py meta-llama/Llama-3.2-3B --bits 4 \
    --quant_save_path GPTQModel/examples/eora/Llama-3.2-3B-4bits \
    --eora_dataset c4 \
    --eora_save_path GPTQModel/examples/eora/Llama-3.2-3B-4bits-eora_rank64_c4 \
    --eora_rank 64
```

### Second option: If a GPTQ model is already available, run EoRA generation directly on the quantized model.
Below is an example of using C4 as calibration data for generating EoRA of rank 64 given a 4-bits GPTQ quantized meta-llama/Llama-3.2-3B. To further improve the accuracy on MMLU, set mmlu for eora_dataset.
```shell
python GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B c4 \
    --quantized_model sliuau/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path GPTQModel/examples/eora/Llama-3.2-3B-4bits-eora_rank64_c4 \
    --eora_rank 64
```

## EoRA Evaluation
To evaluate the GPTQ quantized model and the corresponding EoRA on ARC-C and MMLU run:
```shell
python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model sliuau/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path GPTQModel/examples/eora/Llama-3.2-3B-4bits-eora_rank64_c4 \
    --eora_rank 64
```

## EoRA Inference
Please refer to for `GPTQModel/examples/eora/eora_load_and_inference.py` how to load EoRA and the corresponding GPTQ quantized model for inference.
A simple example:
```python

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora 

eora = Lora(
    # for eora generation, path is adapter save path; for load, it is loading path
    path='GPTQModel/examples/eora/Llama-3.2-3B-4bits-eora_rank64_c4 ',
    rank=64,
)

model = GPTQModel.load(
    model_id_or_path='sliuau/Llama-3.2-3B_4bits_128group_size',
    adapter=eora,
)

tokens = model.generate("Capital of France is")[0]
result = model.tokenizer.decode(tokens)
print(f"Result: {result}")
```

## EoRA Kernel
We are working on improving the numerical stability of our EoRA kernel which can further speedup the EoRA + GPTQ inference up to 2.5x. Stay tuned! 


## EoRA results
We ran a series of experiments on meta-llama/Llama-3.2-3B. From the results, we see that EoRA substantially improves the accuracy of 3/4-bit quantized models on MMLU, and using the MMLU validation set as calibration data compared to using C4 can further increase MMLU accuracy.
|Model| Bit-width | EoRA Calibration Dataset | EoRA Rank | MMLU | MMLU Accuracy Boost(%) |
|---| ---|  ---|  ---|  ---| ---| 
|meta-llama/Llama-3.2-3B | Full-Precision (FP16) | - | - |  54.19 | - |
|meta-llama/Llama-3.2-3B | 4 | - | - | 	24.16 | - |
|meta-llama/Llama-3.2-3B | 4 | C4 | 32 |  52.53 | 217.43%|
|meta-llama/Llama-3.2-3B | 4 | C4 | 64 |  52.49 | 217.26%|
|meta-llama/Llama-3.2-3B | 4 | C4 | 128 |  52.93 | 219.08% |
|meta-llama/Llama-3.2-3B | 4 | MMLU | 32 | 	53.43 | 221.15% |
|meta-llama/Llama-3.2-3B | 4 | MMLU | 64 | 	53.32 | 220.70%|
|meta-llama/Llama-3.2-3B | 4 | MMLU | 128 | 	53.42 | 221.11% |
|meta-llama/Llama-3.2-3B | 3| - | - |  22.89 | - |
|meta-llama/Llama-3.2-3B | 3 | C4 | 32 | 	39.08 | 170.73% |
|meta-llama/Llama-3.2-3B | 3 | C4 | 64 | 38.83 |169.64% |
|meta-llama/Llama-3.2-3B | 3 | C4 | 128 |  39.68 | 173.35%|

In general, setting rank to 32 and use C4 as calibration data could be a good starting point when applying EoRA to improve the quantized model accuracy. 

## Citation
If you find our code useful for your research, please consider citing:
```bibtex
@article{liu2024eora,
  title={EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation},
  author={Liu, Shih-Yang and Yang, Huck and Wang, Chein-Yi and Fung, Nai Chit and Yin, Hongxu and Sakr, Charbel and Muralidharan, Saurav and Cheng, Kwang-Ting and Kautz, Jan and Wang, Yu-Chiang Frank and others},
  journal={arXiv preprint arXiv:2410.21271},
  year={2024}
}
```