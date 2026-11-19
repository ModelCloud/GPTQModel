# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Calibration numeric options must fail at construction, not at quantize time.

``MSEConfig(norm=nan)`` used to construct, round-trip through the saved config,
and silently disable the MSE search; ``FOEMConfig(alpha=nan)`` crashed later
with ``KeyError: 'native_inp'``. ``NaN`` comparisons are always false, so every
range check has to reject non-finite values explicitly (#5307).
"""

import pytest

from gptqmodel.quantization import config as C


NAN = float("nan")
INF = float("inf")


CASES = [
    ("mse-nan", lambda: C.MSEConfig(norm=NAN), ("MSEConfig", "norm", "finite")),
    ("mse-inf", lambda: C.MSEConfig(norm=INF), ("MSEConfig", "norm", "finite")),
    ("mse-negative", lambda: C.MSEConfig(norm=-1.0), ("MSEConfig", "norm", "non-negative")),
    ("mse-bool", lambda: C.MSEConfig(norm=True), ("MSEConfig", "norm", "number")),
    ("gptaq-nan", lambda: C.GPTAQConfig(alpha=NAN), ("GPTAQConfig", "alpha", "finite")),
    ("gptaq-negative", lambda: C.GPTAQConfig(alpha=-5.0), ("GPTAQConfig", "alpha", "non-negative")),
    ("foem-alpha-nan", lambda: C.FOEMConfig(alpha=NAN), ("FOEMConfig", "alpha", "finite")),
    ("foem-alpha-negative", lambda: C.FOEMConfig(alpha=-0.1), ("FOEMConfig", "alpha", "non-negative")),
    ("foem-beta-nan", lambda: C.FOEMConfig(beta=NAN), ("FOEMConfig", "beta", "finite")),
    ("foem-beta-inf", lambda: C.FOEMConfig(beta=INF), ("FOEMConfig", "beta", "finite")),
    ("foem-beta-negative", lambda: C.FOEMConfig(beta=-0.2), ("FOEMConfig", "beta", "non-negative")),
    (
        "hessian-cap-nan",
        lambda: C.HessianConfig(sink_row_cap_median_multiple=NAN),
        ("HessianConfig", "sink_row_cap_median_multiple", "finite"),
    ),
    (
        "hessian-cap-negative",
        lambda: C.HessianConfig(sink_row_cap_median_multiple=-1.0),
        ("HessianConfig", "sink_row_cap_median_multiple", "non-negative"),
    ),
    ("hessian-chunk-size-bool", lambda: C.HessianConfig(chunk_size=True), ("HessianConfig", "chunk_size", "integer")),
    (
        "hessian-chunk-bytes-bool",
        lambda: C.HessianConfig(chunk_bytes=True),
        ("HessianConfig", "chunk_bytes", "integer"),
    ),
]


@pytest.mark.parametrize(
    "factory,fragments",
    [(factory, fragments) for _, factory, fragments in CASES],
    ids=[case_id for case_id, *_ in CASES],
)
def test_invalid_calibration_options_are_rejected(factory, fragments):
    with pytest.raises(ValueError) as excinfo:
        factory()

    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, message


LEGAL = [
    ("mse-zero-disables", lambda: C.MSEConfig(norm=0)),
    ("mse-float", lambda: C.MSEConfig(norm=2.4)),
    ("gptaq-zero", lambda: C.GPTAQConfig(alpha=0)),
    ("gptaq-default", lambda: C.GPTAQConfig(alpha=0.25)),
    ("foem-zero", lambda: C.FOEMConfig(alpha=0, beta=0)),
    ("foem-default", lambda: C.FOEMConfig(alpha=0.25, beta=0.2)),
    ("hessian-cap-zero", lambda: C.HessianConfig(sink_row_cap_median_multiple=0)),
    ("hessian-cap-none", lambda: C.HessianConfig(sink_row_cap_median_multiple=None)),
    ("hessian-chunk-size-one", lambda: C.HessianConfig(chunk_size=1)),
    ("hessian-chunk-bytes-one", lambda: C.HessianConfig(chunk_bytes=1)),
]


@pytest.mark.parametrize("factory", [factory for _, factory in LEGAL], ids=[case_id for case_id, _ in LEGAL])
def test_legal_calibration_values_still_construct(factory):
    assert factory() is not None


def test_config_dict_path_validates_the_same_bounds():
    with pytest.raises(ValueError, match="FOEMConfig: `alpha`"):
        C.QuantizeConfig.from_quant_config(
            {"quant_method": "gptq", "bits": 4, "group_size": 128, "foem": {"alpha": NAN}}
        )
    with pytest.raises(ValueError, match="MSEConfig: `norm`"):
        C.QuantizeConfig.from_quant_config(
            {"quant_method": "gptq", "bits": 4, "group_size": 128, "mse": {"norm": -1.0}}
        )


def test_valid_numbers_are_normalized_to_float():
    assert isinstance(C.MSEConfig(norm=2).norm, float)
    assert isinstance(C.GPTAQConfig(alpha=1).alpha, float)
    assert isinstance(C.FOEMConfig(alpha=0, beta=1).beta, float)
    hessian = C.HessianConfig(sink_row_cap_median_multiple=4)
    assert isinstance(hessian.sink_row_cap_median_multiple, float)
