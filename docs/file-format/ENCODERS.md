<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Encoders and embedding spaces

A store records the encoders that produced its vectors in `store.json`; the engine refuses to
open it with another configuration, because queries must be encoded exactly as documents were.

## Text encoders (`text_encoder`)

| Name | Status | Notes |
|---|---|---|
| `multiscale-words-v1` | stable, default | character 2-5 grams, word tokens with position, word bigrams, fixed density `dim/256` |
| `multiscale-capped-union-v1` | experimental (`experimental` feature to create) | union within each level, capped by cross-level frequency |

An encoder name is a permanent contract: the same text and dimension produce the same indices
forever. Changing output means a new name. `tests/fixtures/text-encoder-v1.json` pins vectors
at the density boundaries (D = 1, 255, 256, 257, 1024, 4096, 16384).

`vector_encoder` is `signed-projection-v2` and `metric` is `jaccard`.

## Lexical terms

Chunk `terms` are lowercase tokens passed through the English Snowball stemmer (`rust-stemmers`)
when `stemmed` is true. The query side stems the same way, so a stemmer change would silently
degrade old stores. The golden search expectations catch that.

## Embedding space

`embedding_space` is `{model, revision, dimensions, normalization, metric}` or null. Chunk
`embedding` arrays must have `dimensions` elements. It is part of the schema equality check.

## Encoder versions added later

<!-- Section reserved for the embeddings work: add each new encoder name, its status, and a
golden store under tests/golden/ before it can be written by default. -->

(None beyond the table above at the time of writing.)
