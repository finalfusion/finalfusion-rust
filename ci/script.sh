#!/bin/bash

set -ex

cargo build
cargo test

# Only run fmt and clippy against stable Rust 
if [ "$TRAVIS_RUST_VERSION" = "stable" ]; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


