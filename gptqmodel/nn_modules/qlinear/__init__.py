import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"
    SUPPORTED_BITS = []
    SUPPORTED_GROUP_SIZES = []

    def validate_bits(self, bits: int):
        if bits not in self.SUPPORTED_BITS:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_BITS}` bits: actual bits = `{bits}`")

    def validate_group_size(self, group_size: int):
        if group_size not in self.SUPPORTED_GROUP_SIZES:
            raise NotImplementedError(
                f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_GROUP_SIZES}` group_size: actual group_size = `{group_size}`")



