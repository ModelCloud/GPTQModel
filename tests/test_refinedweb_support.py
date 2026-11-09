from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.rw import RwgQModel


# HF RefinedWeb checkpoints declare `RefinedWeb` / `RefinedWebModel`, while the
# registry keeps the historical `refinedWeb` / `refinedWebModel` spellings.
# Resolution must not depend on the config's casing.
@pytest.mark.parametrize(
    "model_type",
    (
        "RefinedWeb",
        "RefinedWebModel",
        "refinedWeb",
        "refinedWebModel",
        "refinedweb",
        "refinedwebmodel",
    ),
)
def test_refinedweb_model_type_resolves_case_insensitively(monkeypatch, model_type):
    fake_config = SimpleNamespace(model_type=model_type)

    monkeypatch.setattr(
        auto,
        "resolve_trust_remote_code",
        lambda path, trust_remote_code=False: trust_remote_code,
    )
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(
        auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config
    )

    assert (
        auto.check_and_get_model_definition(
            "/tmp/refinedweb", trust_remote_code=False
        )
        is RwgQModel
    )
