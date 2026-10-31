<a href="https://writerslogic.com/warm/"><img src="https://raw.githubusercontent.com/writerslogic/warm/main/assets/crystal-halo/crystal-detailed-static.png" alt="WARM logo" width="96" align="left"></a>

# WARM Models

The default offline embedding model pack for
[`warm-python`](https://pypi.org/project/warm-python/).

[Website](https://writerslogic.com/warm/) · [Documentation](https://writerslogic.com/warm/docs/latest/embeddings/) · [Source](https://github.com/writerslogic/warm) · [License](https://github.com/writerslogic/warm/blob/main/LICENSE)

<br clear="left">

Install it through the native binding's `embed` extra:

```sh
uv add "warm-python[embed]"
```

The wheel contains the pinned int8 pack for
[`granite-embedding-30m-english`](https://huggingface.co/ibm-granite/granite-embedding-30m-english/tree/9b5b096411652ec1189c68fcfb90d0a82c5b45af)
(about 34 MB). The model is Apache-2.0 licensed. The binding verifies the pack against its pinned
SHA-256 before use.
With `offline=True`, embedding does not make a first-use network request:

```python
from writerslogic_warm import Embedder

embedder = Embedder(offline=True)
vector = embedder.embed_query("Where is my note?")
```

Other registered models can be fetched into the cache on first use, or supplied through
`WARM_EMBED_MODEL_DIR`. This wheel includes the default model only. See the
[embedding guide](https://writerslogic.com/warm/docs/latest/embeddings/) for model selection and
offline setup.
