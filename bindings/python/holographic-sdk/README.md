# Holographic SDK

The official Python SDK for the Holographic Memory System, featuring Zero-Trust FHE-Lite cryptographic privacy.

## Installation

Install the base SDK:
```bash
pip install holographic-sdk
```

Or install with specific framework integrations:
```bash
pip install "holographic-sdk[langchain]"
pip install "holographic-sdk[llamaindex]"
pip install "holographic-sdk[haystack]"
```

## Integrations

```python
# LangChain
from holographic_sdk.langchain import HolographicVectorStore

# LlamaIndex
from holographic_sdk.llamaindex import HolographicVectorStore

# Haystack
from holographic_sdk.haystack import HolographicDocumentStore
```
