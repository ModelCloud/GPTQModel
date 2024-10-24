from typing import Optional, Tuple

import torch.nn as nn

from ...models._const import DEVICE, get_device_by_type
from ...utils.device import check_cuda


class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS = []
    SUPPORTS_GROUP_SIZE = []
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS: bool = True
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    # empty which means all
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = []
    # empty which means all
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = []

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int, *args, **kwargs):
        super().__init__()
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, infeatures=infeatures,outfeatures=outfeatures)
        if err:
            raise err

        if DEVICE.CUDA in self.SUPPORTS_DEVICES:
            check_cuda()

    @classmethod
    def validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, dynamic=None) -> Tuple[
        bool, Optional[Exception]]:
        validate, err = cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, dynamic=dynamic)
        return validate, err

    @classmethod
    def _validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, dynamic=None, infeatures=None,
                  outfeatures=None) -> Tuple[bool, Optional[Exception]]:
        if cls.SUPPORTS_BITS and bits not in cls.SUPPORTS_BITS:
            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTS_GROUP_SIZE and group_size not in cls.SUPPORTS_GROUP_SIZE:
            err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTS_SYM and sym not in cls.SUPPORTS_SYM:
            err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if cls.SUPPORTS_DESC_ACT and desc_act not in cls.SUPPORTS_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
            if cls.SUPPORTS_BITS:
                dynamic_bits = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_bits[pattern] = pattern_dict.get("bits", bits)
                if len(cls.SUPPORTS_BITS) == 1:
                    err = f"{cls} not supported dynamic_bits, only support `{cls.SUPPORTS_BITS}` bits"
                    return False, NotImplementedError(err)
                else:
                    for layer, bits in dynamic_bits.items():
                        if bits not in cls.SUPPORTS_BITS:
                            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual dynamic_bits = `{bits}` for layer `{layer}`"
                            return False, NotImplementedError(err)
            if cls.SUPPORTS_GROUP_SIZE:
                dynamic_group_size = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
                for layer, group_size in dynamic_group_size.items():
                    if group_size not in cls.SUPPORTS_GROUP_SIZE:
                        err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                        return False, NotImplementedError(err)
            if cls.SUPPORTS_SYM:
                dynamic_sym = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_sym[pattern] = pattern_dict.get("sym", sym)
                for layer, sym in dynamic_sym.items():
                    if sym not in cls.SUPPORTS_SYM:
                        err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                        return False, NotImplementedError(err)
            if cls.SUPPORTS_DESC_ACT:
                dynamic_desc_act = {}
                for pattern, pattern_dict in dynamic.items():
                    dynamic_desc_act[pattern] = pattern_dict.get("desc_act", desc_act)
                for layer, desc_act in dynamic_desc_act.items():
                    if desc_act not in cls.SUPPORTS_DESC_ACT:
                        err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}` for layer `{layer}`"
                        return False, NotImplementedError(err)
        if infeatures is not None:
            validate = all(infeatures % in_fea == 0 for in_fea in cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `infeatures` must be divisible by {cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        if outfeatures is not None:
            validate = all(outfeatures % out_fea == 0 for out_fea in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `outfeatures` must be divisible by {cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)

        return True, None

    @classmethod
    def validate_device(cls, device_type: str):
        device = get_device_by_type(device_type)
        if cls.SUPPORTS_DEVICES and device not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTS_DEVICES}` bits: actual device = `{device}`")

    # override me
    def post_init(self):
        pass
