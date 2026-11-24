#!/bin/zsh
# archives FourierPlayground for the app store and testflight, and uploads it when asked.
#   scripts/archive.sh ios            archive only
#   scripts/archive.sh macos upload   archive, then upload to app store connect
# needs FourierPlayground/Local.xcconfig with your team id, and that team's account signed in to xcode.
set -euo pipefail
cd "${0:A:h}/.."

platform="${1:-}"
upload="${2:-}"
project="FourierPlayground/FourierPlayground.xcodeproj"
team="$(sed -n 's/^DEVELOPMENT_TEAM *= *//p' FourierPlayground/Local.xcconfig 2>/dev/null | tr -d ' ')"
if [[ -z "$team" ]]; then
  echo "set DEVELOPMENT_TEAM in FourierPlayground/Local.xcconfig first. see Local.example.xcconfig." >&2
  exit 1
fi

out=".build/archives"
mkdir -p "$out"
archive="$out/FourierPlayground-$platform.xcarchive"

case "$platform" in
  ios)
    xcodebuild -project "$project" -scheme FourierPlayground -configuration Release \
      -derivedDataPath .build/archive-ios -destination 'generic/platform=iOS' \
      -archivePath "$archive" -allowProvisioningUpdates archive
    ;;
  macos)
    # the mac app store requires the sandbox. apple silicon only: the kit uses 16-bit floats.
    xcodebuild -project "$project" -scheme FourierPlayground -configuration Release \
      -derivedDataPath .build/archive-macos -destination 'generic/platform=macOS' \
      -archivePath "$archive" -allowProvisioningUpdates \
      ARCHS=arm64 ONLY_ACTIVE_ARCH=NO \
      CODE_SIGN_IDENTITY="Apple Development" DEVELOPMENT_TEAM="$team" \
      FOURIER_MAC_ENTITLEMENTS=FourierPlayground-Store.entitlements FOURIER_MAC_SANDBOX=YES \
      archive
    ;;
  *)
    echo "usage: scripts/archive.sh ios|macos [upload]" >&2
    exit 1
    ;;
esac

if [[ "$upload" == "upload" ]]; then
  plist="$out/export-$platform.plist"
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key><string>app-store-connect</string>
	<key>destination</key><string>upload</string>
	<key>signingStyle</key><string>automatic</string>
	<key>teamID</key><string>$team</string>
</dict>
</plist>
PLIST
  xcodebuild -exportArchive -archivePath "$archive" -exportOptionsPlist "$plist" \
    -exportPath "$out/export-$platform" -allowProvisioningUpdates
fi
