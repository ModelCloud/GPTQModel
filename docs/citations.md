# Attributions and citations

[Project overview](../README.md)

## Attributions

* GPTQ: IST-DASLab, main-author: Elias Frantar, arXiv:2210.17323
* AWQ: main-authors: Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song
* ParoQuant: Z-Lab, main-authors: Yesheng Liang, Haisheng Chen, Song Han, and Zhijian Liu. [Official implementation](https://github.com/z-lab/paroquant), [Paper](https://openreview.net/forum?id=1USeVjsKau)
* EoRA: Nvidia, main-author: Shih-Yang Liu, arXiv preprint arXiv:2410.21271.
* GAR: Intel, main-author: T Gafni, A Karnieli, Y Hanani, [Paper](https://openaccess.thecvf.com/content/CVPR2025W/eLVM/html/Gafni_Dual_Precision_Quantization_for_Efficient_and_Accurate_Deep_Neural_Networks_CVPRW_2025_paper.html)
* GPTAQ: Yale Intelligent Computing Lab, main-author: Yuhang Li, arXiv:2504.02692.
* Fast Hadamard Transform: [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform), by Tri Dao, vendored into `gptqmodel_ext/hadamard/` under the BSD-3-Clause license. A copy of the license is included in `gptqmodel_ext/hadamard/LICENSE`.
* Swordfish Kernel: Blackwell (`>= sm100`) GPTQ/AWQ kernel from [AlpinDale](https://x.com/AlpinDale). [Paper](https://blog.alpindale.net/posts/swordfish/)
* NF12 / NyoomFloat12: [Together AI](https://www.together.ai/), from *"NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight Compression"* by Liberman, Fang, Zhang, Dao, and Fu (2026): the lossless 12-bit BF16 format, its escape coding, and the reference codec. NF12 inference kernels and the reference implementation are by [AlpinDale](https://x.com/AlpinDale) at [dphnAI/nf12](https://github.com/dphnAI/nf12), vendored into `gptqmodel_ext/nf12/` under the Apache-2.0 license with the original attribution headers. The AMD RDNA4 port in `gptqmodel_ext/nf12_hip/` is a GPT-QModel contribution that decodes the identical byte stream on gfx1200/gfx1201/gfx1250.
* QQQ: Meituan, main-author Ying Zhang, arXiv:2406.09904
* VPTQ: Microsoft, main-authors Yifei Liu et al., arXiv:2409.17066. [Official implementation](https://github.com/microsoft/VPTQ)
* MACaQ: Huawei CSL, main-authors Lorenz K. Muller, Philippe Bich, Chiara Boretti, Hyun-Min Chang, Jiawei Zhuang, and Lukas Cavigelli. [Official implementation](https://github.com/huawei-csl/MACaQ), [Paper](https://arxiv.org/abs/2611.10649)
* Norm Tweaking: Liang Li, Qingyuan Li, Bo Zhang, and Xiangxiang Chu, [paper](https://arxiv.org/abs/2309.02784)
* FOEM: Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/40123)
* GSQ: [IST-DASLab's GSQ paper](https://arxiv.org/abs/2604.18556) and [reference implementation](https://github.com/IST-DASLab/GSQ), by Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtić, Max Kleinegger, and Dan Alistarh. The optional local refinement and staged Llama training in GPT-QModel are adaptations of this work.
* RCO: [IST-DASLab's RCO paper](https://arxiv.org/abs/2605.00649) and [reference implementation](https://github.com/IST-DASLab/RCO), by Michael Helcig and Dan Alistarh. `gptqmodel/quantization/rco.py` and the `scripts/gsq_paper/rco_*.py` drivers are a port of this work.

## Citations

```bibtex
# GPT-QModel
@misc{qubitium2024gptqmodel,
  author = {ModelCloud.ai and qubitium@modelcloud.ai},
  title = {GPT-QModel},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/modelcloud/gptqmodel}},
  note = {Contact: qubitium@modelcloud.ai},
  year = {2024},
}

# GPTQ
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers},
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}

}

# MACaQ
@misc{muller2026macaq,
  title         = {{MACaQ}: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations},
  author        = {Muller, Lorenz K. and Bich, Philippe and Boretti, Chiara and Chang, Hyun-Min and Zhuang, Jiawei and Cavigelli, Lukas},
  year          = {2026},
  eprint        = {2611.10649},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.10649}
}

# GSQ
@article{dadgarnia2026gsq,
  title={GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling},
  author={Dadgarnia, Alireza and Tabesh, Soroush and Nikdan, Mahdi and Helcig, Michael and Kurti{\'c}, Eldar and Kleinegger, Max and Alistarh, Dan},
  journal={arXiv preprint arXiv:2604.18556},
  year={2026}
}

# RCO
@misc{helcig2026rco,
  title={Model Compression with Exact Budget Constraints via Riemannian Manifolds},
  author={Helcig, Michael and Alistarh, Dan},
  journal={arXiv preprint arXiv:2605.00649},
  year={2026}
}

# AWQ
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

# ParoQuant
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

# GGUF / llama.cpp
@misc{ggerganov2023gguf,
  author = {Georgi Gerganov and ggml-org contributors},
  title = {llama.cpp and the GGUF model format},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ggml-org/llama.cpp}},
  note = {Canonical GGUF implementation and format reference; see also \url{https://github.com/ggml-org/llama.cpp/wiki/dev-notes}},
  year = {2023}
}

# EoRA
@article{liu2024eora,
  title={EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation},
  author={Liu, Shih-Yang and Yang, Huck and Wang, Chien-Yi and Fung, Nai Chit and Yin, Hongxu and Sakr, Charbel and Muralidharan, Saurav and Cheng, Kwang-Ting and Kautz, Jan and Wang, Yu-Chiang Frank and others},
  journal={arXiv preprint arXiv:2410.21271},
  year={2024}
}

# GPTAQ
@article{li2025gptaq,
  title={GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration},
  author={Yuhang Li and Ruokai Yin and Donghyun Lee and Shiting Xiao and Priyadarshini Panda},
  journal={arXiv preprint arXiv:2504.02692},
  year={2025}
}

# FOEM
@inproceedings{zheng2026first,
  title={First-order error matters: Accurate compensation for quantized large language models},
  author={Zheng, Xingyu and Qin, Haotong and Li, Yuye and Chu, Haoran and Wang, Jiakai and Guo, Jinyang and Magno, Michele and Liu, Xianglong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={28883--28891},
  year={2026}
}

# QQQ
@article{zhang2024qqq,
      title={QQQ: Quality Quattuor-Bit Quantization for Large Language Models},
      author={Ying Zhang and Peng Zhang and Mincong Huang and Jingyang Xiang and Yujie Wang and Chao Wang and Yineng Zhang and Lei Yu and Chuan Liu and Wei Lin},
      journal={arXiv preprint arXiv:2406.09904},
      year={2024}
}

# VPTQ
@inproceedings{liu2024vptq,
  title={{VPTQ}: Extreme Low-bit Vector Post-Training Quantization for Large Language Models},
  author={Liu, Yifei and Wen, Jicheng and Wang, Yang and Ye, Shengyu and Zhang, Li Lyna and Cao, Ting and Li, Cheng and Yang, Mao},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}

# Norm Tweaking
@misc{li2023normtweakinghighperformancelowbit,
  title={Norm Tweaking: High-performance Low-bit Quantization of Large Language Models},
  author={Liang Li and Qingyuan Li and Bo Zhang and Xiangxiang Chu},
  year={2023},
  eprint={2309.02784},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2309.02784}
}

# ExLlama V3 / EXL3
@misc{turboderp2026exllamav3,
  author = {turboderp and exllamav3 contributors},
  title = {ExLlamaV3 and the EXL3 quantization format},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/turboderp-org/exllamav3}},
  note = {Project repository and EXL3 format documentation: \url{https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md}},
  year = {2026}
}

# Group Aware Reordering (GAR)
@article{gar,
  title={Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference, CVPRW 2025.},
  author={T. Gafni, A. Karnieli, Y. Hanani},
  journal={arXiv preprint arXiv:2505.14638},
  year={2025}
}

# Marlin Kernel
@article{frantar2024marlin,
  title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
  author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2408.11743},
  year={2024}
}

# Swordfish Kernel
@misc{alpindale2026swordfish,
  author = {AlpinDale},
  title = {Swordfish: A Weight-Quantized {GEMM} Family for {NVIDIA} Blackwell},
  howpublished = {\url{https://blog.alpindale.net/posts/swordfish/}},
  year = {2026}
}

# NF12 / NyoomFloat12
@article{liberman2026nyoomfloat12,
  title   = {NyoomFloat12: Accelerating LLM Inference via Lossless 12-bit Weight Compression},
  author  = {Liberman and Fang and Zhang and Dao and Fu},
  year    = {2026},
  note    = {Together AI}
}

@misc{alpindale2026nf12,
  author = {AlpinDale},
  title = {{NF12}: Lossless 12-bit {BF16} weight compression},
  howpublished = {\url{https://github.com/dphnAI/nf12}},
  year = {2026}
}

# Machete Kernel
@misc{vllm2024machete,
  author = {vLLM contributors},
  title = {Machete: Hopper-optimized mixed-precision {GEMM} kernels},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vllm-project/vllm/tree/main/csrc/quantization/machete}},
  note = {CUTLASS-based mixed-precision kernel implementation},
  year = {2024}
}

```
