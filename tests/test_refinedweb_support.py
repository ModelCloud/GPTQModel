import json
from types import SimpleNamespace

import pytest

from gptqmodel.models import auto
from gptqmodel.models.definitions.rw import FalconQModel, RwgQModel
from tests.test_rw_local_port import legacy_payload


# HF RefinedWeb checkpoints declare `RefinedWeb` / `RefinedWebModel`, while the
# registry keeps the historical `refinedWeb` / `refinedWebModel` spellings.
# Resolution must not depend on the config's casing.
CASINGS = (
    "RefinedWeb",
    "RefinedWebModel",
    "refinedWeb",
    "refinedWebModel",
    "refinedweb",
    "refinedwebmodel",
)


@pytest.mark.parametrize("model_type", CASINGS)
def test_refinedweb_model_type_resolves_case_insensitively(model_type):
    assert auto.select_model_definition(model_type, SimpleNamespace(model_type=model_type)) is RwgQModel


@pytest.mark.parametrize("model_type", CASINGS)
def test_refinedweb_checkpoint_loads_as_native_falcon_definition(tmp_path, model_type):
    # The RW LocalModelLoader is selected case-insensitively from config.json and
    # canonicalizes the legacy alias to native Falcon, so the checkpoint resolves to
    # the Falcon definition (an RwgQModel subclass sharing its module tree).
    (tmp_path / "config.json").write_text(json.dumps(legacy_payload(model_type)))

    definition = auto.check_and_get_model_definition(str(tmp_path), trust_remote_code=False)

    assert definition is FalconQModel
    assert issubclass(definition, RwgQModel)
