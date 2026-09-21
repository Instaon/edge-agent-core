#!/usr/bin/env bash
# Build spider-bot.wasm and copy it into the on-disk plugin layout.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CORE="$(cd "$ROOT/../../.." && pwd)"
DEPLOY="$(cd "$CORE/../device-deployment/services/agent/plugins/spider-bot/1.0.0" && pwd)"
cd "$ROOT"
cargo build --release --target wasm32-unknown-unknown
WASM="$ROOT/target/wasm32-unknown-unknown/release/spider_bot.wasm"
for dest in "$CORE/plugins/spider-bot/1.0.0" "$DEPLOY"; do
    mkdir -p "$dest"
    cp "$WASM" "$dest/plugin.wasm"
    cp "$ROOT/manifest.json" "$dest/manifest.json"
    echo "packed $dest"
done
