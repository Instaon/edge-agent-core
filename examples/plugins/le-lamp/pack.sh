#!/usr/bin/env bash
# Build le-lamp.wasm and copy it into the on-disk plugin layout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CORE="$(cd "$ROOT/../../.." && pwd)"
DEPLOY="$CORE/../device-deployment/services/agent/plugins/le-lamp/1.0.0"
cd "$ROOT"
cargo build --release --target wasm32-unknown-unknown
WASM="$ROOT/target/wasm32-unknown-unknown/release/le_lamp.wasm"
for dest in "$CORE/plugins/le-lamp/1.0.0" "$DEPLOY"; do
    mkdir -p "$dest"
    cp "$WASM" "$dest/plugin.wasm"
    cp "$ROOT/manifest.json" "$dest/manifest.json"
    echo "packed $dest"
done
