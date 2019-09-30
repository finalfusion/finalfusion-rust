#!/bin/bash

set -ex

cargo build --verbose
cargo test --verbose

# Modern glibc versions align on 16 byte boundaries, so align on
# 32-byte boundaries to shake out errors. Unfortunately, these
# errors can be non-deterministic (accidental allocation on a
# boundary).
cargo test --verbose --features "align64"

# On Rust 1.32.0, we only care about passing tests.
if [ "$TRAVIS_RUST_VERSION" = "stable" ]; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


