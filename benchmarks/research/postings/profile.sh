#!/bin/sh
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

deps_dir=${1:?usage: profile.sh <cargo dependency directory> <output directory> [baseline revision]}
output_dir=${2:?supply an output directory}
baseline=${3:-d6a38eb11ab0ff320929249d0ec9e26f16e46dad}
repo_dir=$(git rev-parse --show-toplevel)
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
fxhash=$(rg --files "$deps_dir" -g 'libfxhash-*.rlib' | sort | head -1)
rayon=$(rg --files "$deps_dir" -g 'librayon-*.rlib' | sort | head -1)
test -n "$fxhash"
test -n "$rayon"
rustc -Vv > "$output_dir/rustc.txt"
git show "$baseline:src/core/documents/postings.rs" > "$output_dir/baseline.rs"
cp "$repo_dir/src/core/documents/postings.rs" "$output_dir/candidate.rs"
for variant in baseline candidate; do
    # include! cannot expand module-level inner doc comments.
    sed 's@^//!@//@' "$output_dir/$variant.rs" > "$output_dir/$variant-include.rs"
    HMS_POSTINGS_SOURCE="$output_dir/$variant-include.rs" rustc --edition=2021 -C opt-level=3 -C lto=thin \
        -L "dependency=$deps_dir" --extern "fxhash=$fxhash" --extern "rayon=$rayon" \
        "$repo_dir/benchmarks/research/postings/profile.rs" -o "$output_dir/$variant-profile"
    "$output_dir/$variant-profile" > "$output_dir/$variant.jsonl"
done
