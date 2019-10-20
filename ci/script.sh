#!/bin/bash

set -ex

if [ ! -z "$CROSS_TARGET" ]; then
  rustup target add "$CROSS_TARGET"
  cargo install cross --force
  export CARGO_CMD="cross"
  export TARGET_PARAM="--target $CROSS_TARGET"
else
  export CARGO_CMD="cargo"
  export TARGET_PARAM=""
fi

"$CARGO_CMD" build --verbose $TARGET_PARAM
"$CARGO_CMD" test --verbose $TARGET_PARAM

# Only run fmt and clippy against stable Rust 
if [ "$TRAVIS_RUST_VERSION" = "stable" ] && [ -z "$CROSS_TARGET" ]; then
  "$CARGO_CMD" fmt --all -- --check
  "$CARGO_CMD" clippy -- -D warnings
fi


