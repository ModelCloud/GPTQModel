# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

from gptqmodel.models.writer import _prepare_embedding_save_metadata
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import META_FIELD, META_FIELD_QUANTIZER, META_FIELD_TIMESTAMP
from gptqmodel.version import __git_hash__, __local_version__, __version__


class TestQuantizeConfigVersion(unittest.TestCase):
    def test_local_version_appends_git_hash_when_in_git_checkout(self):
        # Local/editable installs should append the short git hash so saved configs
        # are traceable to an exact commit. Outside a git checkout it falls back.
        self.assertTrue(__local_version__.startswith(__version__))
        if __git_hash__:
            self.assertIn(f"-local-git-{__git_hash__}", __local_version__)
        else:
            self.assertEqual(__local_version__, __version__)

    def test_embedding_save_metadata_sets_quantizer_and_timestamp(self):
        qcfg = QuantizeConfig(bits=4, group_size=128)
        _prepare_embedding_save_metadata(qcfg, meta_quantizer=None)

        meta = qcfg.meta
        self.assertIsInstance(meta, dict)

        quantizers = meta.get(META_FIELD_QUANTIZER, [])
        self.assertEqual(len(quantizers), 1)
        self.assertTrue(quantizers[0].startswith("gptqmodel:"))
        self.assertIn(__local_version__, quantizers[0])

        self.assertIn(META_FIELD_TIMESTAMP, meta)
        # date/hour/minute format: YYYY-MM-DDTHH:MM
        self.assertRegex(meta[META_FIELD_TIMESTAMP], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")

    def test_config_round_trip_preserves_version_and_timestamp(self):
        qcfg = QuantizeConfig(bits=4, group_size=128)
        _prepare_embedding_save_metadata(qcfg, meta_quantizer=None)

        payload = qcfg.to_dict()
        self.assertIn(META_FIELD, payload)
        self.assertIn(META_FIELD_TIMESTAMP, payload[META_FIELD])
        self.assertIn(__local_version__, payload[META_FIELD][META_FIELD_QUANTIZER][0])

        restored = QuantizeConfig.from_quant_config(payload)
        self.assertEqual(restored.meta[META_FIELD_QUANTIZER], payload[META_FIELD][META_FIELD_QUANTIZER])
        self.assertEqual(restored.meta[META_FIELD_TIMESTAMP], payload[META_FIELD][META_FIELD_TIMESTAMP])


if __name__ == "__main__":
    unittest.main()
