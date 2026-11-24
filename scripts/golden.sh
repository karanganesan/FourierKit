#!/bin/zsh
# runs the golden test, then rebuilds docs/performance.html from the log.
# usage: scripts/golden.sh ["what this run changed"]
set -euo pipefail

# run from the repo root, whatever the current directory is
cd "${0:A:h}/.."

if [[ $# -gt 0 && -n "$1" ]]; then
  export FOURIER_RUN_LABEL="$1"
fi

# the test appends its line to the log even when it fails, so the page is rebuilt either way
result=0
swift test -c release || result=$?
swift scripts/performance.swift
exit $result
