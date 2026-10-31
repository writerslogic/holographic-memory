# Local embeddings

Semantic search with no API key, no service and no manual model setup. A pure-Rust CPU engine runs a
sentence encoder (BERT, RoBERTa or XLM-R family) with int8 or f16 weights; the tokenizer ships inside
the model file; nothing touches the network at inference.

Everything here is behind cargo features that are **off by default**: the default build, its
dependencies and its serving memory are unchanged (see [Default build](#default-build)).

| Feature | Adds |
|---|---|
| `local-embed` | the engine, `Embedder`, `SemanticIndex`, `TextEncoder::LocalEmbedV1`, the `warm-embed` tool. No network code. |
| `local-embed-download` | `ureq`; first-use download of the pinned upstream files. |

Python wheels enable both. The default npm addon build enables `node-api,security` and does
not export `LocalEmbedder`; build it with `node-api,security,local-embed-download` to use the
optional native Node API below. The JavaScript `semantic` wrapper has its own embedder adapter.

## Use

Rust:

```rust
use writerslogic_warm::core::embed::{Embedder, EmbedOptions, SemanticIndex};
use std::sync::Arc;

let embedder = Arc::new(Embedder::new(&EmbedOptions::default())?);   // default model
let mut index = SemanticIndex::new(embedder);
index.add(&["a", "b"], &["The cat sat on the mat.", "Quarterly revenue grew."])?;
let hits = index.search("feline on a rug", 3)?;                     // [{id, score, text}]
```

Python (`uv pip install warm-py`; add `[embed]` for the pinned default offline pack):

```python
import writerslogic_warm as hm
index = hm.SemanticIndex()                       # default model
index.add(["a", "b"], ["The cat sat on the mat.", "Quarterly revenue grew."])
index.search("feline on a rug", k=3)             # [(id, score, text), ...]
hm.Embedder(threads=4, max_length=256).embed_query("what is WARM?")   # list[float]
```

Node (`npm i writerslogic-warm`; `npm i writerslogic-warm-models` adds offline packs):

```js
const { LocalEmbedder, SemanticIndex } = require('writerslogic-warm')
const embedder = await LocalEmbedder.load()
const index = SemanticIndex.create(embedder)
await index.add(['a', 'b'], ['The cat sat on the mat.', 'Quarterly revenue grew.'])
await index.search('feline on a rug', 3)
```

### In the document store

One line selects the embedder for ingest and search of a store:

```rust
let config = WarmConfig { text_encoder: TextEncoder::LocalEmbedV1, ..WarmConfig::default() };
let core = WarmCore::new(4096, Some(path), Some(config))?;
core.memorize_documents(docs)?;             // chunks are embedded in batches
core.search_documents("feline on a rug", &SearchOptions::default())?;   // hybrid: BM25 + cosine
```

`local-embed-v1` loads the embedder from `WarmConfig::local_embed` (`EmbedOptions`, defaults when
absent), fills `embedding_space` from the model, embeds chunks that arrive without embeddings, embeds
text queries, and encodes `encode_text` as a sparse sketch of the embedding. The store schema records
model, upstream revision, quantization and dimension, so a store opened with a different model is
rejected. Existing encoders (`multiscale-words-v1`, `multiscale-capped-union-v1`) and their stores are
untouched; to add dense vectors to a lexical store, install an embedder with
`WarmCore::set_text_embedder` (the store's `embedding_space` must equal `Embedder::space()`).
Callers that pass their own `embeddings` / `embedding` are never overridden.

## Options

All fields of `EmbedOptions` (camelCase in JSON / Node, keyword arguments in Python) are optional.

| Option | Default | Meaning |
|---|---|---|
| `model` | `default` | registry id (`warm-embed list`) or the path of a `.hmsemb` pack |
| `sha256` | none | expected hash of a pack given by path (registry models always use their pin) |
| `maxLength` | model's (512, or 256) | tokens per text, specials included; longer texts are truncated |
| `pooling`, `normalize`, `queryPrefix`, `passagePrefix` | model's | override the model conventions |
| `batchSize`, `maxBatchTokens` | 32, 2048 | sequences / soft token budget per forward pass; one sequence may exceed the token budget up to `maxLength` |
| `limits` | see below | hard call-level text, input-byte, retained-token, output-byte and concurrent-call limits |
| `threads` | all cores | size of the embedder's private thread pool |
| `offline` | `WARM_EMBED_OFFLINE` | never touch the network |
| `modelDir`, `cacheDir` | env / platform cache | where packs are searched / downloaded to |

Determinism: output is bit-identical for any `threads`, any `batchSize` and any batch composition on
one build (no padding; each output element has one fixed reduction order). Across CPU architectures
the last bits can differ because aarch64 and FMA-enabled x86-64 builds use fused multiply-add.
Each forward pass admits at most `max(batch budget, maxLength)` tokens, also limited by the
call's retained-token cap. Working memory additionally depends on model dimensions, thread-local
tokenizer work, tokenized inputs and output vectors. These are allocation/work limits, not a
measured process RSS ceiling; model weights, caller-owned inputs, foreign-runtime object copies
and retained results are separate.

`limits` defaults to `maxTexts: 4096`, `maxInputBytes: 67108864` (64 MiB),
`maxTokens: 1048576`, `maxOutputBytes: 67108864` (including flat output plus row copies),
and `maxConcurrentCalls: 4`. All limits must be positive. Input bytes count the entire supplied
text plus the selected prefix for every text, before truncation. Tokens count retained tokens,
including specials and prefixes. Calls over capacity fail immediately; the embedder has no
pending queue. Admission covers tokenization, inference and output construction, and releases
on errors. Separate embedders have separate budgets; hosts must bound their number and their
own transport/task queues. Split larger workloads into bounded calls. Limits do not change
the effective embedding identity or successful vectors.

Node requires finite integer limits in `1..=4294967295`; fractions and JavaScript integer-wrap
inputs are rejected before loading a model.

`Embedder::embed_with_report(texts, kind)` returns `{embeddings, inputs}`. Each input reports
`inputBytes`, `retainedBytes`, `tokens`, `byteTruncated` and `tokenTruncated`, in input order.
Node exposes `embedQueryWithReport` and `embedPassagesWithReport`; Python exposes
`embed_query_with_report` and `embed_passages_with_report` (both take lists and return the
same fields with snake_case input keys). Python's `limits` dictionary accepts snake_case or
camelCase keys. Existing vector-only methods retain their output shapes and truncation policy.

Rust store callers can use `WarmCore::try_encode_text` and `try_find_analogy` to propagate
model and resource failures. Their legacy infallible counterparts panic on failure instead of
logging an error and returning an empty vector. Batch writes embed together under one admission;
encoding failures return before the journal transaction, preserving existing records.

## Model packs, hashes and air-gapped installs

A pack is one file (`<id>-int8.hmsemb`: header, tokenizer, quantized tensors). The registry in
`src/core/embed/registry.rs` pins, per model id, the upstream commit, the sha256 and size of each
upstream file, and the sha256 of the converted pack. Resolution order for a registry model:

1. `WARM_EMBED_PACK` (path of the pack file)
2. `modelDir` option / `WARM_EMBED_MODEL_DIR` (directory containing the pack)
3. the cache (`WARM_EMBED_CACHE`, else the platform cache directory)
4. `local-embed-download` builds only, and never when offline: download the pinned upstream files from
   `https://huggingface.co` (override the host with `WARM_EMBED_MIRROR`), verify every file's size and
   sha256, convert to a pack, verify the pack's sha256, then rename into the cache.

Every load re-hashes exactly the bytes it parses; a mismatch fails before any weight is used
(`Sha256Mismatch`). A damaged cache entry is replaced by a fresh verified download when allowed; a file
you named (path, env, model directory) is never modified. An unreachable model fails with the exact
variables to set. Air-gapped:

```sh
warm-embed fetch default --out ./packs            # on a connected machine
export WARM_EMBED_MODEL_DIR=./packs WARM_EMBED_OFFLINE=1   # on the offline one
```

`warm-py[embed]` installs `warm-models`, which carries the pinned
default Granite embedding pack. The Python `Embedder` also uses packs from
`WARM_EMBED_MODEL_DIR` or the local cache. Fetch other models on a connected machine with
`warm-embed fetch`, then copy them to an offline machine if needed.
`warm-embed pack` converts a local BERT-family Hugging Face directory.

Why not bundle weights in the Rust crate: crates.io caps a crate at 10 MB and the smallest good int8
pack is about 25 MB. Python wheels (100 MB limit) and npm can carry them, so they do.

## Model selection and results

See [SELECTION.md](SELECTION.md) (written from `benchmarks/results/embedding_selection_v1.json`) for
the candidates, the splits, the engine-versus-reference agreement, and the retrieval gain, cost and
throughput measured against the lexical default (`benchmarks/results/embedding_store_v1.json`).
Reproduce with `benchmarks/embedding/` (`select_model.py`, `run_engine_eval.py`, `agreement.py`,
`run_store_eval.py`).

## Default build

`default = []`. Without `local-embed*` no new dependency is compiled (`cargo tree` identical), no
embedder code is linked, and the only changes to default code paths are one enum variant
(`TextEncoder::LocalEmbedV1`, rejected by config validation unless the feature is on) and one branch in
document preparation that default encoders never take.

## Effective model identity and migration

Native embedders bind the complete embedding function to the store schema. The revision is a
versioned fingerprint of the pack or model-file bytes, effective pooling, normalization,
query/passage instruction, token and UTF-8 prefix truncation, and inference profile. Disabling
normalization records `none`. Worker counts and batch sizes do not change the pure-Rust identity.
Candle query tasks and numeric precision are part of its identity. The core and MCP backend
validate the installed embedder against that identity, including same-dimensional replacements.

Previously created managed-model stores with nominal revisions cannot establish their original
embedding function. Preserve them with the original binary and configuration; re-embed original
source documents into a new destination. Rewriting `store.json` to a new fingerprint does not
migrate vectors. External embeddings supplied with an explicit application-managed space retain
their existing schema contract. Standalone semantic-index format 2 persists the effective space;
format 1 must likewise be rebuilt from source texts, because its effective options were not saved.

Texts are limited to `max_length` tokens after a UTF-8 prefix cap of
`max(256, 32 * max_length)` bytes; report methods expose each limit separately. Chunk sources
before embedding when complete coverage is needed. Quantization agreement checks validate
numerical drift on their stated fixtures; they do not establish equivalent downstream task quality.
Model fetches in one process share a per-cache-file reservation; independent processes use unique
staging directories and publish only verified complete pack bytes atomically.
