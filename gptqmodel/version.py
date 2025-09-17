# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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

def get_project_version() -> str:
    import pathlib
    import tomllib
    _pyproject_path = pathlib.Path(__name__).parent.parent / "pyproject.toml"
    with _pyproject_path.open("rb") as f:
        _pyproject_data = tomllib.load(f)

    _project_version = _pyproject_data.get("project", {}).get("version")

    return _project_version

__version__ = get_project_version()
