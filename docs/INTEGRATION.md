# Integration guide

## Rust

Use `HmsCore` for vectors, documents, and structural knowledge. The default feature set is
empty; enable `security` for encryption/signing and `provenance` for provenance APIs. Optional
experimental modules have separate compatibility guarantees. Rust 1.89 or newer is required.

```rust
use holographic_memory::HmsCore;

fn main() -> anyhow::Result<()> {
    let memory = HmsCore::new(16384, Some("./memory".into()), None)?;
    memory.memorize("greeting".into(), memory.encode_text("hello world"))?;
    memory.flush()?;
    Ok(())
}
```

## Node.js

CommonJS and ESM both export `HolographicMemorySystem`. Native operations return promises when
work is scheduled off the event loop. For documents, import `DocumentMemory` from
`holographic-memory/semantic`; it serializes model work, bounds its queue, and supports async
iterable ingestion. The default document search is lexical BM25. Add an embedder for cosine
semantic candidates and optional reranking. See [the example](../examples/local-documents.mjs)
and the README for the complete setup.

Install `@huggingface/transformers` separately to use the local ONNX factories. Supply an
existing model directory and explicit model/revision identity. The store schema must match
the adapter's embedding space. Keep models alive while operations are pending and dispose
of them after ingestion/search finishes. No model is downloaded by these factories.

`memorizeFile` now ingests a bounded UTF-8 file as chunks. Use `searchDocuments` to retrieve
passages and `deleteDocument` to remove it. The earlier whole-file vector behavior can be
expressed explicitly with `memorizeText` for bounded inputs if needed.

## Python

The Python wheel exposes `PhaseHVec` and `PhaseResonator`, not the persistent document/search
engine. Build with `maturin develop --features python`:

```python
from holographic_memory import PhaseHVec

a = PhaseHVec.random(1024, 8, 1)
b = PhaseHVec.random(1024, 8, 2)
assert -1.0 <= a.similarity(b) <= 1.0
```

## Portability

Use the Rust or native Node APIs for interoperable encodings. The current text encoder is
versioned as `multiscale-words-v1`; a generic character-trigram implementation does not produce
compatible vectors. Dense projection version 2 preserves sign, and old vectors must be
re-encoded from their original inputs. See [migration](production-readiness.md).

The persistent engine depends on native file locking and mmap. This repository does not
provide a supported browser/Wasm binding. Applications in other languages can expose their
own service around the native engine with authentication and limits appropriate to their use.
