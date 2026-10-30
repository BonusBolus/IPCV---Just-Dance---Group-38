#!/usr/bin/env bash
# Makes the ZIP for the final submission: all files in git (code, songs, docs, tests), the two YOLO models and,
# when given, the report and the poster. Run from anywhere:
#   tools/make_zip.sh [report.pdf] [poster.pdf]
# The result is Group38_JustDance.zip in the project folder. Not in the ZIP: .venv, build-env, dist, recordings,
# measurements and the *_openvino_model folders (the game makes those itself at the first start).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=Group38_JustDance
OUT="$PWD/$NAME.zip"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

if [ -n "$(git status --porcelain)" ]; then
    echo "Warning: there are changes that are not committed; the ZIP only has the committed version." >&2
fi

git archive --format=tar --prefix="$NAME/" HEAD | tar -x -C "$STAGE"
rm -f "$STAGE/$NAME/.gitignore"      # only needed for git, not in the submission

for model in yolo26n-pose.pt yolo26n-seg.pt; do
    if [ ! -f "$model" ]; then
        echo "Missing $model: start the game once (python main.py), then Ultralytics downloads it." >&2
        exit 1
    fi
    cp "$model" "$STAGE/$NAME/"
done

for pdf in "$@"; do
    cp "$pdf" "$STAGE/$NAME/"
done

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" "$NAME")
echo "Made $OUT ($(du -h "$OUT" | cut -f1); the limit is 500 MB)"
