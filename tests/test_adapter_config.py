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
# GPU=-1
import os

from gptqmodel import QuantizeConfig
from gptqmodel.adapter.adapter import Lora, normalize_adapter


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402


lora = "lora"

class TestExtensionConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_extension_parse(self):
        ext = normalize_adapter(adapter={"name": lora, "rank": 128})

        assert isinstance(ext, Lora)
        assert ext.rank == 128
        print(f"{ext}")

        ext = normalize_adapter(adapter=Lora(rank=128))

        assert isinstance(ext, Lora)
        assert ext.rank == 128
        print(f"{ext}")

        try:
            normalize_adapter(adapter={"name": lora, "rank": 128, "crash": 1})
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
        lora_config = Lora(rank=rank)

        kv = lora_config.to_dict()
        print(f"{lora} config: {kv}")

        assert lora_config.rank == rank
        assert len(kv) == 3
        assert rank_field in kv.keys()
        assert kv[rank_field] == rank

    def test_extension_parse_does_not_mutate_serialized_payload(self):
        payload = Lora(path="/tmp/adapter", rank=128).to_dict()

        first = normalize_adapter(adapter=payload)
        second = normalize_adapter(adapter=payload)

        assert isinstance(first, Lora)
        assert isinstance(second, Lora)
        assert payload["name"] == lora
        assert payload["rank"] == 128

    def test_to_dict_does_not_mutate_dynamic_adapter_overrides(self):
        qconfig = QuantizeConfig(
            bits=4,
            group_size=128,
            adapter=Lora(rank=8),
            dynamic={
                "model.layers.0.self_attn.q_proj": {
                    "bits": 8,
                    "adapter": {"rank": 16},
                }
            },
        )
        expected_dynamic = {
            "model.layers.0.self_attn.q_proj": {
                "bits": 8,
                "adapter": {"rank": 16},
            }
        }

        payload = qconfig.to_dict()

        assert payload["dynamic"] == {"model.layers.0.self_attn.q_proj": {"bits": 8}}
        assert qconfig.dynamic == expected_dynamic
        assert qconfig.extract_adapter_rank_patterns() == {"model.layers.0.self_attn.q_proj": 16}

    def test_extension_embed(self):
        bits = 4
        rank = 2

        eora_config = Lora(rank=rank)

        qconfig = QuantizeConfig(
            bits=bits,
            adapter=eora_config,
        )

        print(f"qconfig: {qconfig}")

        assert qconfig.bits == bits
        assert qconfig.adapter == eora_config
        assert qconfig.adapter.rank == rank

