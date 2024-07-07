import torch


def check_cuda(raise_exception: bool = True) -> bool:
    at_least_one_cuda_v6 = any(torch.cuda.get_device_capability(i)[0] >= 6 for i in range(torch.cuda.device_count()))

    if not at_least_one_cuda_v6:
        if raise_exception:
            raise EnvironmentError("GPTQModel requires at least one GPU device with CUDA compute capability >= `6.0`.")
        else:
            return False
    else:
        return True
