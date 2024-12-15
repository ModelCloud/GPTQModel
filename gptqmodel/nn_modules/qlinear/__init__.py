from typing import List, Optional, Tuple, Union

import torch.nn as nn

from ...models._const import DEVICE, get_device_by_type

class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS: List[int] = None
    SUPPORTS_GROUP_SIZE: List[int] = None
    SUPPORTS_DESC_ACT: List[bool] = None
    SUPPORTS_SYM: List[bool] = None
    SUPPORTS_SHARDS: bool = None
    SUPPORTS_TRAINING: bool = None
    SUPPORTS_AUTO_PADDING: bool = None
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = None
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = None

    SUPPORTS_DEVICES: List[DEVICE] = None

    def __init__(self, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures: int, outfeatures: int, *args,
                 **kwargs):
        super().__init__()
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, infeatures=infeatures,outfeatures=outfeatures)
        if err:
            raise err

    @classmethod
    # custom quant linear class can override this and add custom checks
    def validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, infeatures:int=None,
                  outfeatures:int=None, dynamic:Optional[dict]=None, device:Optional[DEVICE]=None, trainable:Optional[bool]=None) -> Tuple[
        bool, Optional[Exception]]:
        validate, err = cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym,
                                      infeatures=infeatures, outfeatures=outfeatures, dynamic=dynamic,
                                      device=device, trainable=trainable)
        return validate, err

    @classmethod
    # internal method and should not be overriden
    def verify_supports_params(cls):
        """
        Validate that SUPPORTS parameters are not None or empty lists, raising an exception if the validation fails.
        """
        base_supports_variables = [
            (name, value) for name, value in BaseQuantLinear.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value)
        ]
        child_supports_variables = [
            (name, value) for name, value in cls.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value)
        ]

        base_supports_variables.sort(key=lambda x: x[0])
        child_supports_variables.sort(key=lambda x: x[0])

        base_variable_names = {name for name, value in base_supports_variables}
        child_variable_names = {name for name, value in child_supports_variables}

        missing_variables = base_variable_names - child_variable_names

        if missing_variables:
            raise ValueError(
                f"{cls.__name__} these SUPPORTS variables are not overridden: {', '.join(sorted(missing_variables))}")

        for name, value in child_supports_variables:
            if not name.startswith("SUPPORTS") or callable(value):
                continue
            if value is None or (isinstance(value, list) and not value):
                raise ValueError(f"{cls.__name__}.{name} cannot be None or an empty list.")

    @classmethod
    def _validate(cls, bits: int, group_size: int, desc_act: bool, sym: bool, dynamic:Optional[dict]=None, infeatures:int=None,
                  outfeatures:int=None, device:Optional[DEVICE]=None, trainable:Optional[bool]=None) -> Tuple[bool, Optional[Exception]]:
        cls.verify_supports_params()

        if device is not None:
            try:
                cls.validate_device(device)
            except NotImplementedError:
                e = f"{cls} does not support device: {device}"
                return False, NotImplementedError(e)

        if trainable and not cls.SUPPORTS_TRAINING:
            err = f"{cls} does not support training."
            return False, NotImplementedError(err)

        if bits not in cls.SUPPORTS_BITS:
            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        if group_size not in cls.SUPPORTS_GROUP_SIZE:
            err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if sym not in cls.SUPPORTS_SYM:
            err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if desc_act not in cls.SUPPORTS_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
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

            dynamic_group_size = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
            for layer, group_size in dynamic_group_size.items():
                if group_size not in cls.SUPPORTS_GROUP_SIZE:
                    err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_sym = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_sym[pattern] = pattern_dict.get("sym", sym)
            for layer, sym in dynamic_sym.items():
                if sym not in cls.SUPPORTS_SYM:
                    err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                    return False, NotImplementedError(err)

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

            validate = infeatures % group_size == 0 or cls.SUPPORTS_AUTO_PADDING
            if not validate:
                err = f"{cls}: `infeatures` must be divisible by `group_size: {group_size}`."
                return False, NotImplementedError(err)
        if outfeatures is not None:
            validate = all(outfeatures % out_fea == 0 for out_fea in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `outfeatures` must be divisible by {cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        return True, None

    @classmethod
    def validate_device(cls, device: Union[DEVICE, str]):
        device = get_device_by_type(device) if isinstance(device, str) else device

        if device not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTS_DEVICES}` bits: actual device = `{device}`")

    # override me
    def post_init(self):
        pass
