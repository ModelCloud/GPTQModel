import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"
