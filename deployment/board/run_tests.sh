#!/usr/bin/env bash
# Build and run every x86 golden test of the C ports (the same as `make test`).  From deployment/board:
#     wsl bash run_tests.sh
set -u
cd "$(dirname "$0")"
make test "$@"
