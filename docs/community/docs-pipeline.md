---
description: How the WARM documentation site is built, versioned and verified, and the commands to run every check locally.
---

# How these docs are built and verified

The site is MkDocs with the Material theme, built into one static directory by
`scripts/build-docs-site.sh` and published to GitHub Pages by `.github/workflows/docs.yml`.
There is no `gh-pages` branch: the workflow uploads the directory as a Pages artifact.

## What is generated, so it cannot drift

| Part | Generated from |
|---|---|
| Landing page copy, install tabs, feature cards | `docs/_data/site.yml`; code is read from the quickstart examples that CI runs |
| Headline benchmark and the vector-search chart | the committed files under `benchmarks/results/`, each time the site is built |
| Comparison table | `docs/_data/comparison.json` when the benchmark generator writes one, else the committed SciFact result |
| Python reference | mkdocstrings reads the SDK source; a stub written from `src/bindings/python/mod.rs` covers the PyO3 module |
| Node and TypeScript reference | typedoc reads every `types` entry in `package.json` exports |
| Rust reference | rustdoc, embedded under `api/rust/doc/` |
| HTTP and C reference | rendered from the OpenAPI document and the C header when the tree has them |
| Changelog | `CHANGELOG.md` |
| Social cards | one image per page, drawn at build time |

## Versions

`dev` is `main`, each `v*` tag that carries a `mkdocs.yml` gets its own site, and `latest` is a
separate build of the newest tag (of `dev` until one exists). Every version names the matching page
under `latest/` as its canonical URL, old versions show a banner, and `versions.json` feeds the
selector in the header. To test the tag path without creating a tag, build any revision as a version:

```sh
WARM_DOCS_TAGS="v0.0.1-test=HEAD" bash scripts/build-docs-site.sh _site
```

## Checks

A build is strict: a warning, a broken link, a missing anchor or a page missing from the
navigation fails it. Pull requests also run, against a local build:

- **axe**: zero serious or critical violations on six pages, at three widths, in light and dark.
- **Lighthouse**: at least 95 in performance, accessibility, best practices and SEO, on the landing page and a documentation page, mobile and desktop.
- **Search**: every query in `scripts/docs-qa/queries.txt` finds its page in the first five. Add a line when a question has no answer.
- **Screenshots** of the landing page and three more pages at desktop, tablet and phone widths in both color schemes, uploaded as the `docs-qa-output` artifact for a visual review.

## Run it locally

```sh
uv venv .venv-docs && uv pip install --python .venv-docs -r docs/requirements.txt
npm ci --prefix scripts/docs-api && npm ci --prefix scripts/docs-qa
WARM_SITE_BASE=http://127.0.0.1:8123 MKDOCS=.venv-docs/bin/mkdocs PYTHON=.venv-docs/bin/python \
  bash scripts/build-docs-site.sh _site
VERSION=latest node scripts/docs-qa/axe.mjs _site
VERSION=latest node scripts/docs-qa/lighthouse.mjs _site
VERSION=latest node scripts/docs-qa/search.mjs _site
```

Set `WARM_BUILD_RUSTDOC=1` to build the Rust reference as well (CI does, and sets
`WARM_DOCS_REQUIRE_API=1` so a missing generator output fails the build).

## Add or change a page

Pages are listed by folder in `docs/.nav.yml`; a page under an existing area appears without
editing it. A page that is in no folder of the navigation fails the build. Put examples in
`examples/quickstart/` and include them, so CI runs them.
