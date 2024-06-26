import torch.nn as nn


class BaseQuantLinear(nn.Module):
    # override me
    QUANT_TYPE = "base"

    SUPPORTS_BITS = []
    SUPPORTS_GROUP_SIZE = []
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDED: bool = True

    def validate(self, bits: int, group_size: int, desc_act: bool, sym: bool):
        if self.SUPPORTS_BITS and bits not in self.SUPPORTS_BITS:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTS_BITS}` bits: actual bits = `{bits}`")

        if self.SUPPORTS_GROUP_SIZE and group_size not in self.SUPPORTS_GROUP_SIZE:
            raise NotImplementedError(
                f"{self.QUANT_TYPE} only supports `{self.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`")

        if self.SUPPORTS_SYM and sym not in self.SUPPORTS_SYM:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTS_SYM}` bits: actual sym = `{sym}`")

        if self.SUPPORTS_DESC_ACT and desc_act not in self.SUPPORTS_DESC_ACT:
            raise NotImplementedError(f"{self.QUANT_TYPE} only supports `{self.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`")

    # override me
    def post_init(self):
        pass


class BaseCudaQuantLinear(BaseQuantLinear):
    # override me
    QUANT_TYPE = "base-cuda"
