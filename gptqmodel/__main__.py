# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""`python -m gptqmodel` runs the command line."""

from .cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
