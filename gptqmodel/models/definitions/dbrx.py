# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


# placer=holder only as dbrx original models are not supported
# supported dbrx_converted models can be found on https://hf.co/ModelCloud
class DbrxQModel(BaseQModel):
    info = {"notes": "Dbrx is only supported using defused/converted models on https://hf.co/ModelCloud with `trust_remote_code=True`"}
