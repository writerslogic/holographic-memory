# Docs polish: handoff

Branch `feat/docs-polish`. Verified locally in headless Chromium: landing page, documentation
pages and API pages at 1440, 820 and 390 px in light and dark; axe 0 serious or critical on six
pages x 6 views; Lighthouse landing 99 mobile / 100 desktop, doc page 99 / 100, other categories
100 (measured on a gzip-serving local server); search 43 of 43 queries.

## Done

- Identity (logo mark from two interfering ring sets, Geist fonts self-hosted, light/dark palettes), landing page with install tabs read from the quickstart examples, headline SciFact result and vector-search chart from `benchmarks/results/`, architecture diagram, bind/bundle/resonate demo, comparison table slot (`docs/_data/comparison.json`), data-driven copy (`docs/_data/site.yml`).
- Seven-tab navigation (`docs/.nav.yml`), breadcrumbs, footer, 404, announcement slot, version banner, Open Graph cards per page, canonical URLs to `latest`, robots, sitemap, `latest` built separately.
- API reference generators (`scripts/docs_api.py`): Python (mkdocstrings + PyO3 stub), Node (typedoc), Rust (rustdoc), HTTP (OpenAPI renderer), C (header renderer).
- Changelog, glossary, FAQ, Community, Install, "how these docs are built" pages.
- CI (`.github/workflows/docs.yml`): strict build, internal links, axe, Lighthouse >= 95, search queries, screenshots as artifacts, SHA-pinned actions.

## Left

- Run the Rust reference and the generated HTTP and C pages in CI once: the HTTP page needs the server's OpenAPI file (`docs/server/openapi.json`, `schemas/openapi.json` or `openapi.json`) and the C page needs `bindings/c/include/*.h` to be on main.
- Add Go, Java, .NET, Swift, Kotlin and WASM to `runtimes:` in `docs/_data/site.yml` as each lands (needs a quickstart snippet per runtime), and wire `runtimes/` pages.
- Pixel-diff visual regression against stored baselines (screenshots are uploaded, not compared).
- Test the tag path with a real tag after the release: `HMS_DOCS_TAGS="v0.0.1-test=HEAD" bash scripts/build-docs-site.sh _site`, then check the banner and selector with Playwright.
- Cache built tag sites between deploys once several tags exist.
- Next steps footers: add `next_steps` front matter (or a folder `.meta.yml`) to pages as areas settle.
