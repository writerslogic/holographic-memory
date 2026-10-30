<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Encoders and embedding spaces

A store records the encoders that produced its vectors in `store.json`; the engine refuses to
open it with another configuration, because queries must be encoded exactly as documents were.

## Text encoders (`text_encoder`)

| Name | Status | Notes |
|---|---|---|
| `multiscale-words-v1` | stable, default | character 2-5 grams, word tokens with position, word bigrams, fixed density `dim/256` |
| `multiscale-words-v2` | opt-in | same multiscale construction; majority ties use frequency descending, coordinate ascending |
| `multiscale-capped-union-v1` | experimental (`experimental` feature to create) | union within each level, capped by cross-level frequency |

An encoder name is a permanent contract: the same text and dimension produce the same indices
forever. Changing output means a new name. `tests/fixtures/text-encoder-v1.json` pins vectors
at the density boundaries (D = 1, 255, 256, 257, 1024, 4096, 16384).

V1 preserves its existing equal-frequency selection, which depends on the standard library's
unstable-selection algorithm; the original goldens remain unchanged. V2 defines a total order
at each majority selection and cross-level selection, so equal counts no longer depend on
selection-algorithm ties. A new schema-5 golden store pins the V2 journal. Creating a V2 store
requires explicitly selecting `multiscale-words-v2`; the default remains V1. Existing V1 stores
reject a V2 configuration. Reencode original sources into a new destination:

```sh
hms-admin reencode original.jsonl new-store --text-encoder multiscale-words-v2
```

`vector_encoder` is `signed-projection-v2` and `metric` is `jaccard`.

## Lexical terms

Chunk `terms` are lowercase tokens passed through the English Snowball stemmer (`rust-stemmers`)
when `stemmed` is true. The query side stems the same way, so a stemmer change would silently
degrade old stores. The golden search expectations catch that.

## Embedding space

`embedding_space` is `{model, revision, dimensions, normalization, metric}` or null. Chunk
`embedding` arrays must have `dimensions` elements. It is part of the schema equality check.

## Encoder versions added later

`local-embed-v1` uses a sparse sketch of a managed local embedding and requires `local-embed`.
Its complete effective embedding space is also part of store compatibility; see
[Local embeddings](../embeddings/README.md#effective-model-identity-and-migration).
