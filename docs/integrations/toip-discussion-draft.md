# ToIP collaboration introduction

Published on 2026-10-28 in the prototype's
[Ideas discussion 21](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/discussions/21).

Discussion title: **WARM as an optional backend for organizational knowledge**.
Integration sources: [bf46965a](https://github.com/writerslogic/warm/commit/bf46965a).

## Discussion body

I'm working on [WARM (WARM)](https://github.com/writerslogic/warm),
an Apache-2.0 Rust library with Node bindings for local search and agent memory.
I'd like to explore whether it could be a useful optional backend for your
organizational knowledge prototype, particularly
[Knowledge Indexing](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/issues/11)
and [MCP Retrieval](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/issues/12).

The [released engine](https://github.com/writerslogic/warm/releases/tag/v0.6.4)
supports BM25 search, hybrid search with supplied embeddings, metadata filters,
source text offsets, and document replacement and deletion. Its separate
[memory API](https://github.com/writerslogic/warm/blob/v0.6.4/docs/memory-api/README.md#forgetting)
supports supersession history and explicit forgetting, including a purge option
with documented storage limits. Deleting an indexed document and forgetting
derived facts remain separate operations.

I've also prepared [Rust and JavaScript integration examples](https://github.com/writerslogic/warm/blob/main/docs/integrations/organizational-knowledge.md)
for the prototype's current flat curated JSON. They preserve source metadata and
content identifiers, keep stable identities separate across applications, and
preview explicit deletion before applying it. Local tests cover provenance,
Unicode curated-text byte offsets, replacement, scoped retrieval, and deletion across
reopens and languages. This is new checkout integration work: it is not included
in v0.6.4, does not yet support the nested model proposed in
[issue #8](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/issues/8),
and is not wired into your MCP server.

The proposed fit would preserve your collection and normalization pipeline,
authoritative provenance, and existing MCP tools. WARM would hold a rebuildable
index alongside the existing backend option. The next integration step would
agree how to preserve canonical chunk IDs, working-group filter semantics, and
the retrieval-time provenance join. Your indexer already replaces changed
documents and supports pruning; the comparison would focus on retrieval and
lifecycle behavior using the same corpus, chunks, embeddings, filters, and
relevance judgments.

WARM also has vector-symbolic binding and bundling operations and associative
memory. Exploring relationships across applications through that representation
is a possible research direction; the lexical examples do not exercise it, and
its benefit on this corpus remains unmeasured.

Would a small optional-backend experiment be useful? I'd welcome input on the
provenance and forgetting cases that matter most, and collaboration on the
interface and evaluation. Shared stewardship could be a later conversation if
the collaboration proves useful.
