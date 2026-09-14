import json

import pytest

from gptqmodel.quantization import GPTQConfig, GSQConfig, QuantizeConfig


@pytest.mark.parametrize("alpha,beta", [(0, .2), (.5, .2), (.5, 0)])
@pytest.mark.parametrize("gsq", [None, GSQConfig(enabled=False), GSQConfig(enabled=True)])
def test_foem_only_config_preserves_coefficients(tmp_path, alpha, beta, gsq):
    config = GPTQConfig(bits=4, group_size=128, foem={"alpha": alpha, "beta": beta}, gsq=gsq)
    path = tmp_path / "quantize_config.json"
    path.write_text(json.dumps(config.to_dict()))
    restored = QuantizeConfig.from_quant_config(json.loads(path.read_text()))
    assert restored.gptaq is None
    assert restored.foem.alpha == alpha
    assert restored.foem.beta == beta
    assert restored.foem.device == config.foem.device
    assert restored.gsq == config.gsq
