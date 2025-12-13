#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

submodule_name=$1
SUBMODULE_PATH="submodules/$submodule_name"

if [ ! -d "$SUBMODULE_PATH" ]; then
  echo "Error: $SUBMODULE_PATH does not exist." >&2
  exit 1
fi

echo "Updating submodule at $SUBMODULE_PATH..."
git -C "$SUBMODULE_PATH" fetch --all
git -C "$SUBMODULE_PATH" pull --ff-only
git add "$SUBMODULE_PATH"

echo "Submodule updated and staged."
