import pytest
import torch

from gptqmodel.looper.exllamav3_processor import (
    HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN,
    resolve_hessian_owner_device,
)


DEVICES = [torch.device("cuda:0"), torch.device("cuda:1")]


def test_default_hessian_owner_remains_expert_modulo() -> None:
    owners = [
        resolve_hessian_owner_device(
            expert_index=expert,
            devices=DEVICES,
            policy=None,
        )
        for expert in range(6)
    ]

    assert owners == [DEVICES[0], DEVICES[1]] * 3


def test_weighted_hessian_owner_maps_one_third_to_primary() -> None:
    policy = {
        "contract": HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN,
        "device_weights": [1, 2],
    }
    owners = [
        resolve_hessian_owner_device(
            expert_index=expert,
            devices=DEVICES,
            policy=policy,
        )
        for expert in range(384)
    ]

    assert owners[:6] == [
        DEVICES[0],
        DEVICES[1],
        DEVICES[1],
        DEVICES[0],
        DEVICES[1],
        DEVICES[1],
    ]
    assert owners.count(DEVICES[0]) == 128
    assert owners.count(DEVICES[1]) == 256


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {
            "contract": HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN,
            "device_weights": [1],
        },
        {
            "contract": HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN,
            "device_weights": [0, 1],
        },
        {
            "contract": HESSIAN_OWNER_WEIGHTED_ROUND_ROBIN,
            "device_weights": [True, 1],
        },
    ],
)
def test_invalid_hessian_owner_policy_fails_closed(policy) -> None:
    with pytest.raises(ValueError, match="owner policy is invalid"):
        resolve_hessian_owner_device(
            expert_index=0,
            devices=DEVICES,
            policy=policy,
        )
