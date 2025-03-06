
## meta-llama/Llama-3.2-1B
## meta-llama/Llama-3.2-3B
## meta-llama/Meta-Llama-3-8B
## meta-llama/Llama-3.1-8B
## meta-llama/Meta-Llama-3-70B

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size 

# /mnt/home/shihyangl/llama3.2-1b-4bit-group128
# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py /mnt/home/shihyangl/llama3.2-1b-4bit-group128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
#     c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/eora_benchmark_speed.py --quantized_model /mnt/home/shihyangl/llama3.2-1b-4bit-group128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/eora_benchmark_speed.py --full_precision_model meta-llama/Llama-3.2-1B

#
# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/eora_benchmark_speed.py --quantized_model /mnt/home/shihyangl/llama3.2-1b-4bit-group128 \
#     --eora_save_path /mnt/home/shihyangl/llama3.2-1b-4bit-group128-eora-rank128-arc \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
#     c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank128_c4.txt

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
#     arc \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_arc \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_arc \
#     --eora_rank 128 \
#     --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank128_arc.txt


# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
#     mmlu \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_mmlu \
#     --eora_rank 128

# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_mmlu \
#     --eora_rank 128 \
#     --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank128_mmlu.txt


# CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
#     arc_c4 \
#     --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_arc_c4 \
#     --eora_rank 128

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_arc_c4 \
    --eora_rank 128 \
    --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank128_arc_c4.txt


CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    mmlu_c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_mmlu_c4 \
    --eora_rank 128

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank128_mmlu_c4 \
    --eora_rank 128 \
    --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank128_mmlu_c4.txt







CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_c4 \
    --eora_rank 64

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_c4 \
    --eora_rank 64|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank64_c4.txt

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    arc \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_arc \
    --eora_rank 64

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_arc \
    --eora_rank 64 \
    --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank64_arc.txt


CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    mmlu \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_mmlu \
    --eora_rank 64

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_mmlu \
    --eora_rank 64 \
    --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank64_mmlu.txt


CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    arc_c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_arc_c4 \
    --eora_rank 64

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_arc_c4 \
    --eora_rank 64 \
    --eval_task arc|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank64_arc_c4.txt


CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/post_quant_eora_generation.py meta-llama/Llama-3.2-1B \
    mmlu_c4 \
    --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_mmlu_c4 \
    --eora_rank 64

CUDA_VISIBLE_DEVICE=1 python /mnt/home/shihyangl/GPTQModel/examples/eora/evaluation.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
    --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size_eora_rank64_mmlu_c4 \
    --eora_rank 64 \
    --eval_task mmlu|tee -a /mnt/home/shihyangl/gptqmodel_saved_model/eval_results/llama3.1_1b/rank64_mmlu_c4.txt