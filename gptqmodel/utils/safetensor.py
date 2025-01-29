import os
from safetensors import safe_open

# debug print all safetensor files in a directory and print its properties
def inspect_safetensors(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            print(f"Safetensor File: {filename}")

            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    print(f"  Key: {key}")
                    print(f"    Shape: {tensor.shape}")
                    print(f"    Dtype: {tensor.dtype}")
            print("-" * 40)