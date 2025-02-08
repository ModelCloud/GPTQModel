# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -- do not touch
import os

from gptqmodel import QuantizeConfig
from gptqmodel.adapter.adapter import EoRA, normalize_adapter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402


class TestExtensionConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_extension_parse(self):
        ext = normalize_adapter(adapter={"eora": {"rank": 128}})

        assert isinstance(ext, EoRA)
        assert ext.rank == 128
        print(f"{ext}")

        ext = normalize_adapter(adapter={"eora": EoRA(rank=128)})

        assert isinstance(ext, EoRA)
        assert ext.rank == 128
        print(f"{ext}")

        try:
            normalize_adapter(adapter={"eora": {"rank": 128, "crash": 1}})
            raise RuntimeError("Non supported extension.property should crash on decode")
        except Exception:
            pass

        try:
            normalize_adapter(adapter={"CRASH": {"rank": 128}})
            raise RuntimeError("Non supported extension should crash on decode")
        except Exception:
            pass


    def test_extension_config(self):
        rank_field = "rank"
        rank = 2
        eora_config = EoRA(rank=rank)

        kv = eora_config.to_dict()
        print(f"eora config: {kv}")

        assert eora_config.rank == rank
        assert len(kv) == 1
        assert rank_field in kv.keys()
        assert kv[rank_field] == rank

    def test_extension_embed(self):
        bits = 4
        rank = 2

        eora_config = EoRA(rank=rank)

        qconfig = QuantizeConfig(
            bits=bits,
            adapter={"eora": eora_config},
        )

        print(f"qconfig: {qconfig}")
        get_eroa_config = qconfig.extension_get("eora")

        print(f"qconfig extract: {get_eroa_config}")
        assert qconfig.bits == bits
        assert len(qconfig.adapter) == 1
        assert qconfig.adapter.get("eora") == eora_config
        assert qconfig.adapter.get("eora").rank == rank
        assert get_eroa_config.rank == rank



