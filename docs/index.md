---
template: home.html
title: WARM
description: Hybrid search, graph indexing, and vector-symbolic memory. Local-first retrieval and associative memory with crash-safe storage and no service calls.
hide:
  - navigation
  - toc
  - path
---

## How to read this site

The document pipeline is conventional retrieval: stemmed BM25 fused with exact cosine over
embeddings you supply. The vector-symbolic layer powers the meaning memory, composition and the
experimental vector index. Where WARM loses to another system on a measured benchmark, the
[benchmark pages](PUBLIC-BENCHMARKS.md) say so, and every number on this site traces to a
committed file under
[`benchmarks/results/`](https://github.com/writerslogic/warm/tree/main/benchmarks/results).

New here? Start with the [quickstart](quickstart/index.md), then read
[what "holographic" means in this code](concepts/holographic.md). Looking something up? Use the
search box, the [glossary](reference/glossary.md) or the [FAQ](reference/faq.md). The version
selector in the header switches between `latest` (the newest tagged release that carries these
docs), `dev` (the `main` branch) and each tagged release. How the site is built and checked is on
[its own page](community/docs-pipeline.md).

The [standards and provenance page](reference/standards.md) names the W3C, C2PA, CAWG, DIF,
and Trust over IP features, their interoperability checks, and the limits on current claims.
