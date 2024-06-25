import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"

    SUPPORTED_BITS = []
    SUPPORTED_GROUP_SIZE = []
    SUPPORTED_DESC_ACT = [True, False]
    SUPPORTED_SYM = [True, False]

    def validate(self, bits: int, group_size: int, desc_act: bool, sym: bool):
        if self.SUPPORTED_BITS and bits not in self.SUPPORTED_BITS:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_BITS}` bits: actual bits = `{bits}`")

        if self.SUPPORTED_GROUP_SIZE and group_size not in self.SUPPORTED_GROUP_SIZE:
            raise NotImplementedError(
                f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_GROUP_SIZE}` group_size: actual group_size = `{group_size}`")

        if self.SUPPORTED_SYM and sym not in self.SUPPORTED_SYM:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_SYM}` bits: actual sym = `{sym}`")

        if self.SUPPORTED_DESC_ACT and desc_act not in self.SUPPORTED_DESC_ACT:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTED_DESC_ACT}` bits: actual desc_act = `{desc_act}`")

    # override me
    def post_init(self):
        pass


class BaseCudaQuantLinear(BaseQuantLinear):
    # override me
    QUANT_TYPE = "base-cuda"
