import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"


class BaseCudaQuantLinear(BaseQuantLinear):
    # override me
    QUANT_TYPE = "base-cuda"
