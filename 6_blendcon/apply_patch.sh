#!/usr/bin/env bash
# PASS stage 6 — fetch upstream BlendCon at the pinned commit and apply our
# camera-placement patch. Upstream is NOT vendored in this repo.
set -euo pipefail

UPSTREAM_URL="https://github.com/Ali-Tohidifar/BlendCon"
PINNED_COMMIT="fb6784945b261aaee007b3e93d23346ebfc5eee2"
PATCH="$(cd "$(dirname "$0")" && pwd)/blendcon_pass.patch"
DEST="${1:-BlendCon}"

if [ -e "$DEST" ]; then
    echo "ERROR: '$DEST' already exists — remove it or pass another target dir." >&2
    exit 1
fi

echo "Cloning BlendCon (upstream: $UPSTREAM_URL) ..."
git clone "$UPSTREAM_URL" "$DEST"
cd "$DEST"

echo "Checking out pinned commit $PINNED_COMMIT ..."
git checkout --quiet "$PINNED_COMMIT"

echo "Applying PASS camera-placement patch ..."
git apply --stat "$PATCH"
git apply "$PATCH"

echo
echo "Done. Patched BlendCon is in: $DEST"
echo "Configure config.yaml, place the stage-5 .blend scene, then run:"
echo "  blender --background --python DataGenerator.py"
