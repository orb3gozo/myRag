#!/usr/bin/env bash
# Fetches the Spanish FastAPI docs (sparse checkout) into data/fastapi-es/.
set -euo pipefail

TAG="${FASTAPI_DOCS_TAG:-0.162.5}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/fastapi-es"
# Example code that the .md files include with {* ../../docs_src/... *}
DEST_SRC="$ROOT/data/fastapi-docs_src"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse \
  https://github.com/fastapi/fastapi.git "$TMP/fastapi"
git -C "$TMP/fastapi" sparse-checkout set docs/es/docs docs_src

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$TMP/fastapi/docs/es/docs/." "$DEST/"
rm -rf "$DEST_SRC"
cp -r "$TMP/fastapi/docs_src" "$DEST_SRC"
echo "$TAG" > "$DEST/.fastapi-tag"

echo "FastAPI docs ($TAG) in $DEST: $(find "$DEST" -name "*.md" | wc -l) .md files"
