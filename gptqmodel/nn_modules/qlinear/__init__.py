import torch.nn as nn


class BaseQuantLinear(nn.Module):

    SUPPORTED_BITS = []
    SUPPORTED_GROUP_SIZE = []
    SUPPORTED_DESC_ACT = [True, False]
    SUPPORTED_SYM = [True, False]
    SUPPORTED_SHARDS: bool = True

    @classmethod
    def validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, raise_error: bool = True) -> bool:
        validate = True
        err = ""
        if cls.SUPPORTED_BITS and bits not in cls.SUPPORTED_BITS:
            validate = False
            err = f"{cls} only supports `{cls.SUPPORTED_BITS}` bits: actual bits = `{bits}`"
        elif cls.SUPPORTED_GROUP_SIZE and group_size not in cls.SUPPORTED_GROUP_SIZE:
            validate = False
            err = f"{cls} only supports `{cls.SUPPORTED_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
        elif cls.SUPPORTED_SYM and sym not in cls.SUPPORTED_SYM:
            validate = False
            err = f"{cls} only supports `{cls.SUPPORTED_SYM}` bits: actual sym = `{sym}`"
        elif cls.SUPPORTED_DESC_ACT and desc_act not in cls.SUPPORTED_DESC_ACT:
            validate = False
            err = f"{cls} only supports `{cls.SUPPORTED_DESC_ACT}` bits: actual desc_act = `{desc_act}`"

        if not validate and raise_error:
            raise NotImplementedError(err)

        return validate

    # override me
    def post_init(self):
        pass
