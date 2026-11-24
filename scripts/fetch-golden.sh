#!/bin/zsh
# downloads the golden tensors from the github release golden-v1 and checks their hashes
# against tensors_sha256 in golden/manifest.json. needs the gh cli and a published release.
set -euo pipefail

# run from the repo root, whatever the current directory is
cd "${0:A:h}/.."

tag="golden-v1"
clips=(short medium long)
manifest="golden/manifest.json"

# the repo comes from the origin remote, in ssh or https form
remote="$(git remote get-url origin)"
repo="${remote%.git}"
repo="${repo#git@github.com:}"
repo="${repo#https://github.com/}"
if [[ "$repo" == */*/* || "$repo" != */* ]]; then
  echo "cannot read a github owner/repo from the origin remote" >&2
  exit 1
fi

mkdir -p golden/tensors
for clip in $clips; do
  gh release download "$tag" --repo "$repo" --pattern "$clip.safetensors" --dir golden/tensors --clobber
done

# a hash that differs from the manifest is a failure, never a warning
failed=0
for clip in $clips; do
  file="golden/tensors/$clip.safetensors"
  want="$(plutil -extract "clips.$clip.tensors_sha256" raw -o - "$manifest")"
  have="$(shasum -a 256 "$file" | cut -d' ' -f1)"
  if [[ "$want" == "$have" ]]; then
    echo "ok       $file"
  else
    echo "mismatch $file" >&2
    echo "  manifest $want" >&2
    echo "  file     $have" >&2
    failed=1
  fi
done

if (( failed )); then
  echo "the golden tensors do not match golden/manifest.json. delete golden/tensors and try again." >&2
  exit 1
fi
echo "all three golden tensors match the manifest"
