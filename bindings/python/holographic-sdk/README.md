# Holographic SDK

The official Python SDK for the Holographic Memory System, featuring Zero-Trust FHE-Lite cryptographic privacy.

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
```

# Phidata
from holographic_sdk.phidata import HolographicVectorDb
