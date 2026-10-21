# Quickstart

Five minutes, one small public corpus, four steps: **ingest** documents, **search** them,
**compare** with SQLite FTS5, and **encrypt** the store. Pick Rust, Python or Node.

The corpus is the 30 articles of the Universal Declaration of Human Rights, a United Nations
document in the public domain (`examples/quickstart/corpus.jsonl`). The queries are eight
questions in `examples/quickstart/queries.json`, each with the article that answers it.

!!! info "This page cannot rot"
    Every code block below is included from `examples/quickstart/`, and CI runs those files on
    every pull request and checks their output against the tables shown here. If a snippet stops
    working, the build fails before the page does.

## Set up

=== "Rust"

    ```sh
    git clone https://github.com/writerslogic/holographic-memory
    cd holographic-memory
    cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml
    ```

    The example crate depends on the library by path, with the `security` feature, plus
    `rusqlite` (bundled SQLite) for the comparison. In your own project:
    `cargo add holographic-memory --features security`.

=== "Python"

    The Python package is an HTTP client for `hms-server`, which stores vectors you supply.
    Start the server, then run the script:

    ```sh
    git clone https://github.com/writerslogic/holographic-memory
    cd holographic-memory
    cargo build --release --features server --bin hms-server
    HMS_API_KEY=quickstart target/release/hms-server --input-dim 384 &
    HMS_API_KEY=quickstart uv run --with bindings/python/holographic-sdk \
        python examples/quickstart/python/quickstart.py
    ```

    The script embeds text with `toy_embed`, a small hashed bag of words so it needs no model
    download. It is lexical and only a stand-in: use a real embedding model for semantic
    search. The `holographic-vsa` wheel on PyPI is a different thing (the quantized-phase
    vector substrate) and does not include document search.

=== "Node"

    ```sh
    git clone https://github.com/writerslogic/holographic-memory
    cd holographic-memory
    npm ci --omit=optional && npm run build
    node examples/quickstart/node/quickstart.mjs
    ```

    The native module is built from source (Rust toolchain required). The script uses
    `node:sqlite`, so it needs Node 24 or newer.

## 1. Ingest

A store is a directory; it persists there and reopens with the same arguments. Each document has
an ID, its text, a source URI, and optional metadata and embeddings. Text is split into chunks
(256 words by default) and written in one atomic batch.

=== "Rust"

    ```rust
    --8<-- "examples/quickstart/rust/src/main.rs:ingest"
    ```

=== "Python"

    ```python
    --8<-- "examples/quickstart/python/quickstart.py:ingest"
    ```

=== "Node"

    ```js
    --8<-- "examples/quickstart/node/quickstart.mjs:ingest"
    ```

## 2. Search, and compare with SQLite FTS5

The baseline is the same corpus in an FTS5 virtual table ranked by `bm25`. A bare multi-word FTS5
query means AND, which would unfairly return nothing for most of these questions, so the
baseline ORs the words.

=== "Rust"

    ```rust
    --8<-- "examples/quickstart/rust/src/main.rs:compare"
    ```

=== "Python"

    ```python
    --8<-- "examples/quickstart/python/quickstart.py:compare"
    ```

=== "Node"

    ```js
    --8<-- "examples/quickstart/node/quickstart.mjs:compare"
    ```

Output (identical in Rust and Node, which call the same engine; CI diffs it):

```text
--8<-- "examples/quickstart/expected-search.txt"
```

The Python script prints this table, which differs because `toy_embed` and the engine's text
analysis are not the same thing:

```text
--8<-- "examples/quickstart/expected-search-python.txt"
```

Read this honestly. Eight hand-written questions on 30 short passages show how the API behaves.
They are not a benchmark and say nothing about relevance at scale. HMS's default text search is
stemmed BM25, the same family as FTS5, so on keyword-style queries they largely agree. Supply embeddings (see below) and HMS fuses exact
cosine with BM25, which is where it can beat a lexical engine on paraphrases. Measured retrieval
results on public datasets are on the [benchmarks](../PUBLIC-BENCHMARKS.md) page, including where
HMS loses.

Each hit also carries the passage text, its byte range in the source and the score, so you can
show a snippet and link back to the source:

=== "Rust"

    ```rust
    --8<-- "examples/quickstart/rust/src/main.rs:hit"
    ```

=== "Python"

    ```python
    --8<-- "examples/quickstart/python/quickstart.py:hit"
    ```

=== "Node"

    ```js
    --8<-- "examples/quickstart/node/quickstart.mjs:hit"
    ```

## 3. Encrypt

=== "Rust"

    ```rust
    --8<-- "examples/quickstart/rust/src/main.rs:encrypt"
    ```

=== "Python"

    ```python
    --8<-- "examples/quickstart/python/quickstart.py:encrypt"
    ```

=== "Node"

    ```js
    --8<-- "examples/quickstart/node/quickstart.mjs:encrypt"
    ```

Rust and Node enable AES-256-GCM with an Argon2id-derived key (the `security` feature, which
the npm build includes). The script then proves two things: a wrong passphrase cannot open the
store, and the right one can. Prefer `encryptionPassphraseEnv` (Node) or
`encryption_passphrase_env` (Rust) over a literal passphrase in real code.

The script also scans every file in the encrypted store for a known passage and asserts none
contains it, the text segment included. Read the [privacy limits](../concepts/privacy-limits.md)
for what encryption does not cover (file names, sizes, access patterns).

The Python client cannot encrypt, because `hms-server` has no encryption at rest. It offers
client-side vector masking with `zero_trust_key`, which is obfuscation: the server still learns
every pairwise similarity and stores text and metadata in plaintext
([server limits](../SERVER.md)).

## Where next

- [Offline agent demo](offline-agent.md): a personal agent memory with no embeddings and no network.
- [Concepts](../concepts/index.md): what is lossy, and the tested capacity range.
- [Production readiness](../production-readiness.md): storage format, locking, migration, limits.

<!-- COORDINATOR-FILL: embeddings-and-memory-api
     The bundled embeddings and the memory API are being added by other work. When both have
     merged, add the following here, with executed code in examples/quickstart/ like the rest:
     (a) a "Add embeddings" step that swaps toy_embed for the bundled model, and
     (b) a "Memory API" step. Then delete this comment and the notice below. -->

!!! note "Embeddings and the memory API"
    The bundled embedding models and the high-level memory API are not part of this quickstart
    yet. This section will show them, with executed code, after they are released.
