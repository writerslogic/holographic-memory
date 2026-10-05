# Holographic SDK

HTTP client and agent-framework adapters for a Holographic Memory System service.

The client talks to a service exposing `POST /api/v1/documents/batch`, `POST /api/v1/documents`, `DELETE /api/v1/documents/{id}` and `POST /api/v1/query`. The `holographic-memory` repository does not ship that service; you must provide one.

## Vector masking

Passing `zero_trust_key` masks every embedding client-side with a keyed signed permutation (`VectorMask`, scrypt-derived). The server can still rank vectors because inner products and cosine similarity are preserved exactly. This is obfuscation, **not encryption**: the server learns all pairwise similarities, and known plain/masked pairs reveal the key for the coordinates they cover. Pass a per-collection `mask_salt` (at least 16 bytes) shared by all clients of that collection. Masks produced by 0.1.0/0.1.1 are not compatible and were not secure.

## Installation

Install the base SDK:
```bash
pip install holographic-sdk
```

Or install with specific framework integrations:
```bash
pip install "holographic-sdk[all]"
```

## Supported Integrations

```python
# LangChain
from holographic_sdk.langchain import HolographicVectorStore

# LlamaIndex
from holographic_sdk.llamaindex import HolographicVectorStore

# Haystack
from holographic_sdk.haystack import HolographicDocumentStore

# Microsoft Semantic Kernel
from holographic_sdk.semantic_kernel import HolographicMemoryStore

# DSPy
from holographic_sdk.dspy import HolographicRM

# CrewAI
from holographic_sdk.crewai import HolographicSearchTool

# Embedchain
from holographic_sdk.embedchain import HolographicDB

# Phidata
from holographic_sdk.phidata import HolographicVectorDb

# Pydantic AI
from holographic_sdk.pydantic_ai import get_holographic_tool

# SmolAgents
from holographic_sdk.smolagents import HolographicSearchTool
```
