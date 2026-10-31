#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the whole versioned documentation site into one directory, with no gh-pages branch:
#   <out>/dev/       the current checkout (main in CI)
#   <out>/vX.Y.Z/    one site per release tag that carries a mkdocs.yml
#   <out>/latest/    its own build of the newest tag (of dev until a tag carries docs), so its
#                    URLs, sitemap and canonical links say /latest/ and not a version
#   <out>/versions.json, <out>/index.html (redirect to latest/), <out>/404.html, <out>/robots.txt
# Every version names the matching page under latest/ as its canonical URL; scripts/docs_finalize.py
# repairs the pages that have no counterpart there and writes the root files.
# versions.json uses the format the Material version selector reads, so no mike is needed.
#
# Usage: scripts/build-docs-site.sh [out-dir]     (needs mkdocs and the packages in docs/requirements.txt)
# Environment: WARM_SITE_BASE (public base URL), MKDOCS and PYTHON (executables), WARM_DOCS_TAGS
# (space-separated tag list that replaces `git tag --list 'v*'`, used to test the tag path).
set -euo pipefail
export DISABLE_MKDOCS_2_WARNING=true
export WARM_DOCS_API_TOOLS="${WARM_DOCS_API_TOOLS:-$(git rev-parse --show-toplevel)/scripts/docs-api}"

out="${1:-_site}"
base="${WARM_SITE_BASE:-https://writerslogic.com/warm/docs}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
case "$out" in /*) ;; *) out="$repo_root/$out" ;; esac
rm -rf "$out"
mkdir -p "$out"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

# rustdoc for one source tree, built once and shared with the `latest` build of the same tree.
# WARM_BUILD_RUSTDOC=1 (CI) enables it; the API reference page needs it, so CI also sets
# WARM_DOCS_REQUIRE_API=1.
rustdoc() { # key source-dir
  [ "${WARM_BUILD_RUSTDOC:-0}" = 1 ] || return 0
  [ -d "$scratch/rustdoc/$1" ] || "$repo_root/scripts/build-rustdoc.sh" "$2" "$scratch/rustdoc/$1"
}

build() { # name ref source-dir is-latest rustdoc-key
  WARM_SITE_URL="$base/$1/" WARM_DOCS_REF="$2" WARM_CANONICAL_BASE="$base/latest/" WARM_IS_LATEST="${4:-0}" \
    WARM_RUSTDOC_DIR="$scratch/rustdoc/${5:-$1}" \
    "${MKDOCS:-mkdocs}" build --quiet --site-dir "$out/$1" --config-file "$3/mkdocs.yml"
}

# The current checkout is dev. A failure here is a real docs error and fails the build.
rustdoc dev "$repo_root"
build dev main "$repo_root" 0 dev

# Release tags, newest first. A tag cut before the docs site existed has no mkdocs.yml and is skipped.
# A tag whose own docs no longer build is reported and left out so it cannot block every later deploy.
if [ -n "${WARM_DOCS_TAGS:-}" ]; then
  tags="$(printf '%s\n' $WARM_DOCS_TAGS)"
else
  tags="$(git tag --list 'v*' --sort=-v:refname)"
fi
versions=()
while IFS= read -r entry; do
  [ -n "$entry" ] || continue
  tag="${entry%%=*}" # the version name; WARM_DOCS_TAGS may say name=rev to build any revision as a version
  rev="${entry#*=}"
  if ! git cat-file -e "$rev:mkdocs.yml" 2>/dev/null; then
    echo "skip $tag: no mkdocs.yml at that tag"
    continue
  fi
  if ! git show "$rev:Cargo.toml" | grep -q '^name = "writerslogic-warm"$'; then
    echo "skip $tag: legacy HMS documentation redirects to latest; tagged source is preserved"
    continue
  fi
  src="$scratch/$tag"
  mkdir -p "$src"
  git archive "$rev" | tar -x -C "$src"
  if rustdoc "$tag" "$src" && build "$tag" "$tag" "$src" 0 "$tag"; then
    versions+=("$tag")
  else
    echo "::warning::docs for $tag do not build and are left out of the site"
    rm -rf "${out:?}/$tag"
  fi
done <<< "$tags"

if [ "${#versions[@]}" -gt 0 ]; then
  build latest "${versions[0]}" "$scratch/${versions[0]}" 1 "${versions[0]}"
  latest_title="${versions[0]}"
else
  build latest main "$repo_root" 1 dev
  latest_title="dev"
fi

# versions.json for the selector. The newest tag carries the `latest` alias; before the first
# docs-bearing tag, dev does.
{
  printf '[\n'
  sep=''
  for v in ${versions[@]+"${versions[@]}"}; do
    alias='[]'
    [ "$v" = "${versions[0]}" ] && alias='["latest"]'
    printf '%s  {"version": "%s", "title": "%s", "aliases": %s}' "$sep" "$v" "$v" "$alias"
    sep=$',\n'
  done
  dev_alias='[]'
  [ "${#versions[@]}" -eq 0 ] && dev_alias='["latest"]'
  printf '%s  {"version": "dev", "title": "dev", "aliases": %s}\n]\n' "$sep" "$dev_alias"
} > "$out/versions.json"

touch "$out/.nojekyll"
mkdir -p "$out/ns/warm"
cp "$repo_root/docs/reference/fixtures/warm-v1-context.jsonld" "$out/ns/warm/v1.jsonld"
mkdir -p "$out/ns/hms"
cp "$repo_root/docs/reference/fixtures/hms-v1-context.jsonld" "$out/ns/hms/v1.jsonld"
"${PYTHON:-python3}" "$repo_root/scripts/docs_finalize.py" "$out" "$base"
echo "built dev ${versions[*]:-} latest=$latest_title into $out"
