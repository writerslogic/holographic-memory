<a href="https://holographic-memory.com/"><img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/crystal-halo/crystal-detailed-static.png" alt="Holographic Memory logo" width="96" align="left"></a>

# Holographic Memory for Python

Native Python bindings for Holographic Memory's vector search and vector-symbolic algebra.

[Website](https://holographic-memory.com/) · [Documentation](https://docs.holographic-memory.com/) · [Source](https://github.com/writerslogic/holographic-memory) · [License](https://github.com/writerslogic/holographic-memory/blob/main/LICENSE)

<br clear="left">

## Install

Python 3.9 or newer is required. Install the native binding and NumPy for the vector-index API:

```sh
uv add holographic-memory-py numpy
```

The distribution is named `holographic-memory-py`; the import is `holographic_memory`.

For the pinned default embedding model with no first-use download, install
`holographic-memory-py[embed]`. That extra adds the separate
[`holographic-memory-models`](https://pypi.org/project/holographic-memory-models/) wheel.

## Vector search

`VectorIndex` builds an approximate nearest-neighbor index over vectors you supply. It supports
angular, Euclidean, and inner-product search. The returned IDs are row numbers in the input matrix.

```python
import numpy as np
from holographic_memory import VectorIndex

vectors = np.random.default_rng(7).normal(size=(128, 32)).astype(np.float32)
index = VectorIndex("angular", degree=16, build_ef=64)
index.fit(vectors)
index.set_query_arguments(ef=64, rerank=32)
matches = index.query(vectors[0], 5)
print(matches.tolist())
```

The binding also provides `UpdatableIndex` for inserts and deletes, `FilteredIndex` for restricted
searches, and `SparseIndex` for sparse vectors.

With the `[embed]` extra, the pinned default text embedding model is available offline:

```python
from holographic_memory import Embedder

embedder = Embedder(offline=True)
print(len(embedder.embed_query("Where is my note?")))
```

## Phase-vector algebra

`PhaseHVec` supports deterministic phase-vector binding, unbinding, bundling, and similarity.

```python
from holographic_memory import PhaseHVec

key = PhaseHVec.random(1024, 64, seed=1)
value = PhaseHVec.random(1024, 64, seed=2)
stored = key.bind(value)
assert stored.unbind(key).phases() == value.phases()
```

`resonator_factorize` searches codebooks for factors of a bound phase vector.

## Choosing the right Python package

This native binding runs vector search and phase-vector operations in your Python process. It does
not expose the persistent document-search and agent-memory APIs. For an HTTP client to `hms-server`,
use [`holographic-memory-sdk`](https://pypi.org/project/holographic-memory-sdk/). The Rust and Node.js
packages expose the full embedded engine.
