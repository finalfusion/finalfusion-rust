#!/bin/bash

set -ex

cargo build
cargo test

# On Rust 1.31.0, we only care about passing tests.
if rustc --version | grep -v "^rustc 1.31.0"; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


