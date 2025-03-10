
## meta-llama/Llama-3.2-1B
## meta-llama/Llama-3.2-3B
## meta-llama/Meta-Llama-3-8B
## meta-llama/Llama-3.1-8B
## meta-llama/Meta-Llama-3-70B

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank128_c4.txt

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     arc \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_arc \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_arc \
#     --eora_rank 128 \
#     --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank128_arc.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     mmlu \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_mmlu \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_mmlu \
#     --eora_rank 128 \
#     --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank128_mmlu.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     arc_c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_arc_c4 \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_arc_c4 \
#     --eora_rank 128 \
#     --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank128_arc_c4.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     mmlu_c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_mmlu_c4 \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_mmlu_c4 \
#     --eora_rank 128 \
#     --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank128_mmlu_c4.txt







# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_c4 \
#     --eora_rank 64

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_c4 \
#     --eora_rank 64|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank64_c4.txt

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     arc \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_arc \
#     --eora_rank 64

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_arc \
#     --eora_rank 64 \
#     --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank64_arc.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     mmlu \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_mmlu \
#     --eora_rank 64

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_mmlu \
#     --eora_rank 64 \
#     --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank64_mmlu.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     arc_c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_arc_c4 \
#     --eora_rank 64

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_arc_c4 \
#     --eora_rank 64 \
#     --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank64_arc_c4.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
#     mmlu_c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_mmlu_c4 \
#     --eora_rank 64

# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_mmlu_c4 \
#     --eora_rank 64 \
#     --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/rank64_mmlu_c4.txt


# CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank64_c4 \
#     --eora_rank 64


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
    c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size_eora_rank128_c4 \
    --eora_rank 128

CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
    c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size_eora_rank64_c4 \
    --eora_rank 64


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
    c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size_eora_rank128_c4 \
    --eora_rank 128


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-3B \
    c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size_eora_rank64_c4 \
    --eora_rank 64


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size_eora_rank128_c4 \
    --eora_rank 128|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/3_bits_rank128_c4.txt


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size_eora_rank64_c4 \
    --eora_rank 64|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/3_bits_rank64_c4.txt



CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size_eora_rank128_c4 \
    --eora_rank 128|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/2_bits_rank128_c4.txt


CUDA_VISIBLE_DEVICE=0 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size_eora_rank64_c4 \
    --eora_rank 64|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.2_3b/2_bits_rank64_c4.txt



python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_3bits_128group_size|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/quantized_weight/Llama-3.2-3B_3bits.txt


python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_2bits_128group_size|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/quantized_weight/Llama-3.2-3B_2bits.txt
