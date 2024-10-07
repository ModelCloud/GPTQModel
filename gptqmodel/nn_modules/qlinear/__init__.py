from typing import Tuple, Optional

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
    # empty which means all
    SUPPORT_INFEATURES_DIVISIBLE_BY = []
    # empty which means all
    SUPPORT_OUTFEATURES_DIVISIBLE_BY = []

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int, *args, **kwargs):
        super().__init__()
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, infeatures=infeatures,outfeatures=outfeatures)
        if err:
            raise err

        if DEVICE.CUDA in self.SUPPORTED_DEVICES:
            check_cuda()

    @classmethod
    def validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, dynamic=None) -> Tuple[
        bool, Optional[Exception]]:
        validate, err = cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, dynamic=dynamic)
        return validate, err

    @classmethod
    def _validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, dynamic=None, infeatures=None,
                  outfeatures=None) -> Tuple[bool, Optional[Exception]]:
        if cls.SUPPORTED_BITS and bits not in cls.SUPPORTED_BITS:
            err = f"{cls} only supports `{cls.SUPPORTED_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTED_GROUP_SIZE and group_size not in cls.SUPPORTED_GROUP_SIZE:
            err = f"{cls} only supports `{cls.SUPPORTED_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTED_SYM and sym not in cls.SUPPORTED_SYM:
            err = f"{cls} only supports `{cls.SUPPORTED_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTED_DESC_ACT and desc_act not in cls.SUPPORTED_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTED_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
            if cls.SUPPORTED_BITS:
                dynamic_bits = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_bits[pattern] = pattern_dict.get("bits", bits)
                if len(cls.SUPPORTED_BITS) == 1:
                    err = f"{cls} not supported dynamic_bits, only support `{cls.SUPPORTED_BITS}` bits"
                    return False, NotImplementedError(err)
                else:
                    for layer, bits in dynamic_bits.items():
                        if bits not in cls.SUPPORTED_BITS:
                            err = f"{cls} only supports `{cls.SUPPORTED_BITS}` bits: actual dynamic_bits = `{bits}` for layer `{layer}`"
                            return False, NotImplementedError(err)
            if cls.SUPPORTED_GROUP_SIZE:
                dynamic_group_size = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
                for layer, group_size in dynamic_group_size.items():
                    if group_size not in cls.SUPPORTED_GROUP_SIZE:
                        err = f"{cls} only supports `{cls.SUPPORTED_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                        return False, NotImplementedError(err)
            if cls.SUPPORTED_SYM:
                dynamic_sym = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_sym[pattern] = pattern_dict.get("sym", sym)
                for layer, sym in dynamic_sym.items():
                    if sym not in cls.SUPPORTED_SYM:
                        err = f"{cls} only supports `{cls.SUPPORTED_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                        return False, NotImplementedError(err)
            if cls.SUPPORTED_DESC_ACT:
                dynamic_desc_act = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_desc_act[pattern] = pattern_dict.get("desc_act", desc_act)
                for layer, desc_act in dynamic_desc_act.items():
                    if desc_act not in cls.SUPPORTED_DESC_ACT:
                        err = f"{cls} only supports `{cls.SUPPORTED_DESC_ACT}` bits: actual desc_act = `{desc_act}` for layer `{layer}`"
                        return False, NotImplementedError(err)
        if infeatures is not None:
            validate = all(infeatures % in_fea == 0 for in_fea in cls.SUPPORT_INFEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `infeatures` must be divisible by {cls.SUPPORT_INFEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        if outfeatures is not None:
            validate = all(outfeatures % out_fea == 0 for out_fea in cls.SUPPORT_OUTFEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `outfeatures` must be divisible by {cls.SUPPORT_OUTFEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)

        return True, None

    @classmethod
    def validate_device(cls, device_type: str):
        device = get_device_by_type(device_type)
        if cls.SUPPORTED_DEVICES and device not in cls.SUPPORTED_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTED_DEVICES}` bits: actual device = `{device}`")

    # override me
    def post_init(self):
        pass
