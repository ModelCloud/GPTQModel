import logging
import unittest

import torch

from gptqmodel.nn_modules.qlinear.machete import MacheteQuantLinear
from gptqmodel.utils.machete import (
    _validate_machete_device_support,
    machete_import_exception,
    pack_quantized_values_into_int32,
)


logger = logging.getLogger("test_machete")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class TestMacheteKernelLoad(unittest.TestCase):
    IN_FEATURES = 64
    OUT_FEATURES = 128

    device = torch.device("cuda")

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required to exercise the machete kernel")

        if machete_import_exception is not None:
            reason = f"machete extension import failed: {machete_import_exception}"
            logger.warning(reason)
            raise unittest.SkipTest(reason)

        if not _validate_machete_device_support():
            props = torch.cuda.get_device_properties()
            reason = (
                "Machete requires NVIDIA Hopper or newer (SM90+); "
                f"cuda is {props.name} (sm_{props.major}{props.minor})"
            )
            logger.warning(reason)
            raise unittest.SkipTest(reason)

        props = torch.cuda.get_device_properties()
        logger.info(
            "Running machete sanity test on cuda (%s, sm_%d%d, %.2f GiB)",
            props.name,
            props.major,
            props.minor,
            props.total_memory / (1024 ** 3),
        )

    def test_machete_linear_forward_zero_weights(self):
        module = self._build_linear()
        input_tensor = torch.ones((2, self.IN_FEATURES), dtype=torch.float16, device=self.device)
        expected = torch.zeros((2, self.OUT_FEATURES), dtype=torch.float16, device=self.device)

        try:
            output = module(input_tensor)
            torch.cuda.synchronize()
        except Exception as exc:
            self._fail_with_details("forward", exc)
            return

        self.assertTrue(
            torch.allclose(output, expected, atol=1e-5),
            "Machete linear layer did not produce the expected zero output for zero weights",
        )

    def _build_linear(self) -> MacheteQuantLinear:
        module = MacheteQuantLinear(
            bits=4,
            group_size=64,
            desc_act=False,
            sym=True,
            in_features=self.IN_FEATURES,
            out_features=self.OUT_FEATURES,
            bias=False,
        ).to(self.device)
        module.eval()

        with torch.no_grad():
            module.g_idx.copy_(
                torch.arange(self.IN_FEATURES, dtype=torch.int32, device=self.device)
            )
            module.scales.copy_(torch.ones_like(module.scales, device=self.device))

            quant_zero = torch.full(
                (self.IN_FEATURES, self.OUT_FEATURES),
                fill_value=module.weight_type.bias,
                dtype=torch.int32,
                device=self.device,
            )
            packed = pack_quantized_values_into_int32(
                quant_zero, module.weight_type, packed_dim=0
            )
            module.qweight.copy_(packed)

        try:
            module.post_init()
        except Exception as exc:
            self._fail_with_details("post_init", exc)

        return module

    def _fail_with_details(self, stage: str, exc: Exception) -> None:
        props = torch.cuda.get_device_properties()
        detail = (
            f"Machete {stage} failed on cuda "
            f"({props.name}, sm_{props.major}{props.minor}): "
            f"{type(exc).__name__}: {exc}"
        )
        logger.error(detail, exc_info=exc)
        hint = (
            "Suggested fix: confirm that `gptqmodel_machete_kernels.so` was built for this CUDA "
            "toolkit and that the NVIDIA driver runtime matches the compiled compute capability."
        )
        self.fail(f"{detail}\n{hint}")


if __name__ == "__main__":
    unittest.main()
