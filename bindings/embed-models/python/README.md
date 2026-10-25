<a href="https://holographic-memory.com/"><img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/crystal-halo/crystal-detailed-static.png" alt="Holographic Memory logo" width="96" align="left"></a>

# Holographic Memory Models

The default offline embedding model pack for
[`holographic-memory-py`](https://pypi.org/project/holographic-memory-py/).

[Website](https://holographic-memory.com/) · [Documentation](https://docs.holographic-memory.com/latest/embeddings/) · [Source](https://github.com/writerslogic/holographic-memory) · [License](https://github.com/writerslogic/holographic-memory/blob/main/LICENSE)

<br clear="left">

Install it through the native binding's `embed` extra:

```sh
uv add "holographic-memory-py[embed]"
```

The wheel contains the pinned int8 pack for
[`granite-embedding-30m-english`](https://huggingface.co/ibm-granite/granite-embedding-30m-english/tree/9b5b096411652ec1189c68fcfb90d0a82c5b45af)
(about 34 MB). The model is Apache-2.0 licensed. The binding verifies the pack against its pinned
SHA-256 before use.
With `offline=True`, embedding does not make a first-use network request:

```python
from holographic_memory import Embedder

embedder = Embedder(offline=True)
vector = embedder.embed_query("Where is my note?")
```

Other registered models can be fetched into the cache on first use, or supplied through
`HMS_EMBED_MODEL_DIR`. This wheel includes the default model only. See the
[embedding guide](https://docs.holographic-memory.com/latest/embeddings/) for model selection and
offline setup.
