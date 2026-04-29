# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from transformers.utils import hub as transformers_hub
from transformers.utils import logging as transformers_logging


cached_file = transformers_hub.cached_file
create_repo = transformers_hub.create_repo
has_file = transformers_hub.has_file
hf_hub_download = transformers_hub.hf_hub_download
list_repo_tree = transformers_hub.list_repo_tree
snapshot_download = transformers_hub.snapshot_download

disable_progress_bar = transformers_logging.disable_progress_bar

# Reuse the hub client instance that transformers already exposes so GPT-QModel
# does not need to import huggingface_hub directly.
_HF_API = list_repo_tree.__self__


def list_repo_files(*args, **kwargs):
    return _HF_API.list_repo_files(*args, **kwargs)


def model_info(*args, **kwargs):
    return _HF_API.model_info(*args, **kwargs)


def repo_info(*args, **kwargs):
    return _HF_API.repo_info(*args, **kwargs)
