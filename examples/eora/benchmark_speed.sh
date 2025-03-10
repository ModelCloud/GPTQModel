# python examples/eora/eora_benchmark_speed.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-1B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/llama-3.2-1b_4bits_128group_size_eora_rank64_c4 \
#     --eora_rank 64


## meta-llama/Llama-3.2-1B
## meta-llama/Llama-3.2-3B
## meta-llama/Meta-Llama-3-8B
## meta-llama/Llama-3.1-8B

# python examples/eora/eora_benchmark_speed.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size \
#     --eora_save_path /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size_eora_rank128_c4 \
#     --eora_rank 128


python examples/eora/eora_benchmark_speed.py --quantized_model /mnt/home/shihyangl/gptqmodel_saved_model/Llama-3.2-3B_4bits_128group_size

