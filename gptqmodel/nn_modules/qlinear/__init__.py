import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"
    SUPPORTED_BITS = []
    SUPPORTED_GROUP_SIZES = []
    SUPPORTED_DESC_ACT = [True, False]
    SUPPORTED_SYM = [True, False]

    def validate_bits(self, bits: int):
        if bits not in self.SUPPORTED_BITS:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_BITS}` bits: actual bits = `{bits}`")

    def validate_group_size(self, group_size: int):
        if group_size not in self.SUPPORTED_GROUP_SIZES:
            raise NotImplementedError(
                f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_GROUP_SIZES}` group_size: actual group_size = `{group_size}`")

    def validate_sym(self, sym: bool):
        if sym not in self.SUPPORTED_SYM:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_SYM}` bits: actual bits = `{sym}`")

    def validate_desc_act(self, desc_act: bool):
        if desc_act not in self.SUPPORTED_DESC_ACT:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_DESC_ACT}` bits: actual bits = `{desc_act}`")

    # override me
    def post_init(self):
        pass


