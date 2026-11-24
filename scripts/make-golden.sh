#!/bin/zsh
# remakes the golden tensors with the python reference. needs the model prepared and uv.
# takes several minutes. it also rewrites golden/manifest.json, which is frozen:
# do not commit a changed manifest without the owner.
set -euo pipefail

# run from the repo root, whatever the current directory is
cd "${0:A:h}/.."

exec uv run --project reference/python python reference/python/make_golden.py "$@"
