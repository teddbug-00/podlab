#!/bin/sh
cd "$(dirname "$0")" || exit 1
if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust is not installed yet. Install it from https://rustup.rs (or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh),"
  echo "then open a new terminal and run ./run.sh again."
  exit 1
fi
cargo run --release -- "$@"
echo
echo "Done. The table above is also saved in results.txt - send me that file."
