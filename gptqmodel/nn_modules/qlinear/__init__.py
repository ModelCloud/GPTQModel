import torch.nn as nn

from ...models._const import DEVICE, get_device_by_type
from ...utils.device import check_cuda

class BaseQuantLinear(nn.Module):
    SUPPORTED_BITS = []
    SUPPORTED_GROUP_SIZE = []
    SUPPORTED_DESC_ACT = [True, False]
    SUPPORTED_SYM = [True, False]
    SUPPORTED_SHARDS: bool = True
    SUPPORTED_DEVICES = [DEVICE.CUDA]

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, *args, **kwargs):
        super().__init__()
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym)
        if err:
            raise NotImplementedError(err)

        if DEVICE.CUDA in self.SUPPORTED_DEVICES:
            check_cuda()

    @classmethod
    def validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool) -> bool:
        validate, _ = cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym)
        return validate

    @classmethod
    def _validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, ):
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
        return validate, err

    @classmethod
    def validate_device(cls, device_type: str):
        device = get_device_by_type(device_type)
        if cls.SUPPORTED_DEVICES and device not in cls.SUPPORTED_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTED_DEVICES}` bits: actual device = `{device}`")

    # override me
    def post_init(self):
        pass
