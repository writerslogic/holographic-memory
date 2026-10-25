---
description: Evaluate document search with known answers, inspect source passages and separate lexical retrieval from embedding quality.
search:
  boost: 2
---

# Evaluate search quality on your documents

A search integration is ready when it returns useful evidence for the questions your application
actually asks. Start with a small set you can inspect, then measure a representative workload.

## 1. Establish a repeatable baseline

Run the [quickstart](../quickstart/index.md). Its public-domain corpus contains 30 articles,
eight queries and expected document IDs. Rust and Node use the same engine; the Python example
uses a toy embedding and has its own expected output.

For example, `right to a fair trial` returns `udhr-10`. The result identifies the original
article, so you can inspect the text instead of relying only on a score. The website's
interactive walkthrough displays these recorded results; it does not run the engine in your browser.

## 2. Make a small evaluation set

For each question, record the documents or passages that answer it. Include exact phrases,
paraphrases, rare terms, queries that should return nothing, and queries restricted by metadata.
Keep evaluation questions separate from any examples used to tune retrieval settings.

Run against a fixed snapshot. Record the engine commit, enabled features, embedding model and
revision, dimensions, filters and query settings so another person can reproduce the result.

## 3. Inspect what comes back

For document-search results, inspect the document ID, source URI, passage and byte offsets.
Verify that the passage supports the answer and that its source is still current. Byte offsets
are offsets in the source bytes, not JavaScript character indices; take care with UTF-8 text.

Check empty results and filtered results explicitly. A query that returns a plausible passage
from the wrong tenant, document version or time period is not a successful retrieval.

## 4. Add semantic retrieval deliberately

First test BM25 with words that occur in the source. Then supply document and query embeddings
from the same model, revision and preprocessing pipeline. Compare lexical, semantic and hybrid
results on the same questions. Keep a mode only when your evaluation supports it.

Use the [embedding guide](../embeddings/README.md) for local models. The Python quickstart's
`toy_embed` is an installation demonstration, not a semantic model.

## 5. Measure quality and cost together

Use recall@k when you have relevant-document labels, and nDCG@k when relevance is graded.
Report latency, throughput, memory and indexing cost alongside quality. Keep hardware,
thread counts, corpus size and concurrency explicit. A higher throughput at lower recall
is a different operating point, not an unconditional improvement.

The [public benchmarks](../PUBLIC-BENCHMARKS.md) provide measured results and limitations;
the [glossary](../reference/glossary.md) defines the metrics. The eight-query quickstart is
an API example and is too small to establish general relevance quality.
