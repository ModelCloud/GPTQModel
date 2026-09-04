#!/usr/bin/env bash
set -euo pipefail

exec "${1:?missing runtime smoke binary}"
