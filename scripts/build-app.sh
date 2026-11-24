#!/bin/zsh
# builds FourierPlayground in release, with derived data under .build/app.
set -euo pipefail

# run from the repo root, whatever the current directory is
cd "${0:A:h}/.."

project="FourierPlayground/FourierPlayground.xcodeproj"
if [[ ! -d "$project" ]]; then
  echo "$project does not exist yet. the app is still being built." >&2
  exit 1
fi

# apple silicon only: the kit uses 16-bit floats, which swift does not have on intel macs
xcodebuild \
  -project "$project" \
  -scheme FourierPlayground \
  -configuration Release \
  -derivedDataPath .build/app \
  -destination 'platform=macOS,arch=arm64' \
  ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  build
