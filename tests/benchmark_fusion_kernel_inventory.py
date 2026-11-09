# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Export native fusion eligibility without substituting a portable backend."""

import argparse
import json
import platform
from pathlib import Path

import torch

from gptqmodel.models._const import DEVICE, normalize_device
from gptqmodel.models.fusion_kernels import kernel_specs
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND


def inventory():
    specs = kernel_specs()
    families = [DEVICE.CPU]
    if torch.cuda.is_available():
        families.append(normalize_device(torch.device("cuda:0")))
    rows, auto = [], []
    for family in families:
        device = torch.device("cpu" if family == DEVICE.CPU else "cuda:0")
        for cls in sorted(importer._registry_kernel_classes(), key=lambda cls: cls.__name__):
            spec = specs.get(cls)
            methods = set(importer.infer_quant_methods(cls))
            row = {"family": family.value, "kernel": cls.__name__, "module": cls.__module__,
                   "methods": sorted(method.value for method in methods),
                   "declared_devices": [value.value for value in cls.SUPPORTS_DEVICES],
                   "fusion_layout": spec.layout if spec else None}
            if DEVICE.ALL not in cls.SUPPORTS_DEVICES and family not in cls.SUPPORTS_DEVICES:
                row["status"] = "different_hardware_family"
            elif not methods & {METHOD.GPTQ, METHOD.AWQ}:
                row["status"] = "outside_gptq_awq_fusion_contract"
            else:
                # Native eight-bit backends are tested as eight-bit, never by
                # expanding the canonical four-bit checkpoint representation.
                bits = 8 if cls.SUPPORTS_BITS == [8] else 4
                group = -1 if cls.__name__ in ("AllSparkLinear", "AllSparkAwqLinear") else 128
                pack = torch.int16 if spec and spec.layout == "awq_fast" else torch.int32
                row.update(bits=bits, group_size=group, pack_dtype=str(pack), contracts={})
                for dtype in (torch.float16, torch.bfloat16):
                    try:
                        ok, error = cls.validate(bits=bits, group_size=group, sym=True, desc_act=False,
                                                 pack_dtype=pack, dtype=dtype, device=device,
                                                 in_features=512, out_features=1024)
                        row["contracts"][str(dtype)] = {"available": bool(ok), "reason": str(error) if error else ""}
                    except Exception as error:
                        row["contracts"][str(dtype)] = {"available": False, "reason": f"{type(error).__name__}: {error}"}
                available = any(value["available"] for value in row["contracts"].values())
                row["status"] = ("native_contract_available" if spec else "no_qualified_fusion_adapter") if available else "native_contract_unavailable"
            rows.append(row)
        for method, fmt in ((METHOD.GPTQ, FORMAT.GPTQ_V2), (METHOD.AWQ, FORMAT.GEMM)):
            for dtype in (torch.float16, torch.bfloat16):
                cls = importer.select_quant_linear(bits=4, group_size=128, sym=True, desc_act=False,
                                                   pack_dtype=torch.int32, dtype=dtype, device=device,
                                                   backend=BACKEND.AUTO, format=fmt, quant_method=method)
                auto.append({"family": family.value, "method": method.value, "dtype": str(dtype), "kernel": cls.__name__})
    # These format-specific loaders are outside the qlinear registry too.
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear

    specialized = [{"kernel": cls.__name__, "module": cls.__module__,
                    "declared_devices": [value.value for value in cls.SUPPORTS_DEVICES],
                    "status": "outside_gptq_awq_fusion_contract"}
                   for cls in (ExllamaV3Linear, ExllamaV3TorchLinear, TorchFP4Linear)]
    return {"torch": torch.__version__, "hip": torch.version.hip, "python": platform.python_version(),
            "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "registry_class_count": len(importer._registry_kernel_classes()),
            "rows": rows, "auto": auto, "specialized_loaders": specialized}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['rows'])} registry/device contracts to {args.output_json}")
