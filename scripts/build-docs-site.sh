#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the whole versioned documentation site into one directory, with no gh-pages branch:
#   <out>/dev/       the current checkout (main in CI)
#   <out>/vX.Y.Z/    one site per release tag that carries a mkdocs.yml
#   <out>/latest/    a copy of the newest tag's site (of dev until a tag carries docs)
#   <out>/versions.json, <out>/index.html (redirect to latest/)
# versions.json uses the format the Material version selector reads, so no mike is needed.
#
# Usage: scripts/build-docs-site.sh [out-dir]     (needs mkdocs and the packages in docs/requirements.txt)
set -euo pipefail

out="${1:-_site}"
base="${HMS_SITE_BASE:-https://writerslogic.github.io/holographic-memory}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
case "$out" in /*) ;; *) out="$repo_root/$out" ;; esac
rm -rf "$out"
mkdir -p "$out"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

build() { # name ref source-dir
  HMS_SITE_URL="$base/$1/" HMS_DOCS_REF="$2" "${MKDOCS:-mkdocs}" build --quiet --site-dir "$out/$1" --config-file "$3/mkdocs.yml"
}

# The current checkout is dev. A failure here is a real docs error and fails the build.
build dev main "$repo_root"

# Release tags, newest first. A tag cut before the docs site existed has no mkdocs.yml and is skipped.
# A tag whose own docs no longer build is reported and left out so it cannot block every later deploy.
versions=()
while IFS= read -r tag; do
  [ -n "$tag" ] || continue
  if ! git cat-file -e "$tag:mkdocs.yml" 2>/dev/null; then
    echo "skip $tag: no mkdocs.yml at that tag"
    continue
  fi
  src="$scratch/$tag"
  mkdir -p "$src"
  git archive "$tag" | tar -x -C "$src"
  if build "$tag" "$tag" "$src"; then
    versions+=("$tag")
  else
    echo "::warning::docs for $tag do not build and are left out of the site"
    rm -rf "${out:?}/$tag"
  fi
done < <(git tag --list 'v*' --sort=-v:refname)

if [ "${#versions[@]}" -gt 0 ]; then
  cp -R "$out/${versions[0]}" "$out/latest"
  latest_title="${versions[0]}"
else
  cp -R "$out/dev" "$out/latest"
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

cat > "$out/index.html" <<'HTML'
<!doctype html>
<meta charset="utf-8">
<title>Holographic Memory System</title>
<link rel="canonical" href="latest/">
<meta http-equiv="refresh" content="0; url=latest/">
<a href="latest/">Holographic Memory System documentation</a>
HTML
touch "$out/.nojekyll"
echo "built dev ${versions[*]:-} latest=$latest_title into $out"
