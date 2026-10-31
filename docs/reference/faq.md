---
description: Answers to the questions people ask about WARM, each traced to a documentation page, the code or an open issue.
---

# FAQ

Each answer points to the page or issue that backs it. If your question is not here, open an
[issue](https://github.com/writerslogic/warm/issues) or ask in
[Discussions](https://github.com/writerslogic/warm/discussions).

## Is the document search actually holographic?

No. Document search chunks text, scores it with stemmed BM25 and exact cosine over embeddings
you supply, and fuses the two lists. It does not rank with hypervectors. The holographic part
is the vector-symbolic layer: the meaning memory, composition, sequences and the experimental
vector index. [What "holographic" means here](../concepts/holographic.md) lists which is which
and shows how to check it in the code.

## Do I need embeddings?

No. Text search works from the text alone, with stemmed BM25. Supply an embedding per chunk and
WARM also scores exact cosine and blends the two, which is where it can beat a lexical engine on
paraphrases. The [quickstart](../quickstart/index.md) runs both ways.

## Why does the Python package need a server?

`warm-sdk` is an HTTP client for `warm-server`. The server wraps one store per tenant and
holds the engine, so the Python process stays small and the store has one writer. The
`warm-python` wheel on PyPI provides native vector search and phase vectors, and
does not include document search. See the [server page](../SERVER.md).

## Does it send anything over the network?

The library makes no network calls and sends no telemetry. Two opt-in Cargo features touch a
network: `server` listens for HTTP requests, and `provenance-scitt` uses an HTTP client to submit
signed statements to a transparency service you configure. Neither is on by default
(`default = []`).

## Is my data encrypted on disk?

Only if you turn it on. The `security` feature adds AES-256-GCM with an Argon2id-derived key.
What the encryption covers, and what a stored hypervector still reveals, is on
[Privacy limits](../concepts/privacy-limits.md); read it before relying on encryption at rest.

## Opening an encrypted store with the wrong passphrase says "cannot replay arena frame at 0"

That is the current error: the real cause, an authenticated-decryption failure, is wrapped in
the replay context. It means the passphrase is wrong or the store is damaged. A clearer message
is tracked in [issue 91](https://github.com/writerslogic/warm/issues/91).

## How many items fit in one bundle?

There is no single safe number. The capacity depends on the dimension and density, and the
false-positive rate climbs well before the bundle saturates. The
[capacity page](../concepts/capacity.md) reports what was swept, where the sweep stops, and that
earlier "hard wall" figures were wrong and are retired.

## Where does WARM lose?

The benchmark pages say so next to the wins. On vector search the sparse vector path is far
below exact search on recall, and the quantized graph index does not reach 0.95 recall@10 on
nytimes-256 where FAISS, hnswlib, Lucene and RaBitQ do. On SciFact the hybrid search is the
best WARM configuration. Start at [Public benchmarks](../PUBLIC-BENCHMARKS.md).

## Which Rust version do I need?

Rust 1.89 or newer (the MSRV). Node 22 or newer for the bindings, and Node 24 for the
quickstart, which uses `node:sqlite`.

## Why is `npm run build` changing `index.d.ts`?

The napi build regenerates it, and the file is both tracked and listed in `.gitignore`. Do not
commit that diff by accident. The cleanup is tracked in
[issue 104](https://github.com/writerslogic/warm/issues/104).

## Why do the docs have three versions?

`latest` is the newest tagged release that carries these docs, `dev` is the `main` branch, and
each release tag keeps its own copy. The selector in the header switches between them, and an
old version shows a banner. [How the docs are built](../community/docs-pipeline.md) explains it.

## Which license is it under?

Apache-2.0. There is no CLA; you certify your right to contribute with a sign-off on each
commit. See [Contributing](../community/index.md).
