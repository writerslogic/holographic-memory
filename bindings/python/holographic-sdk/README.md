<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo-black.svg">
  <img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo.png" width="120" alt="Holographic Memory System" align="left">
</picture>

<h3>Holographic SDK</h3>
<p><strong>Python HTTP client and agent-framework adapters for a Holographic Memory System service.</strong></p>

<br clear="left">

<p align="center">
  <a href="https://pypi.org/project/holographic-sdk/"><img src="https://img.shields.io/pypi/v/holographic-sdk?style=flat-square&amp;color=007ec6&amp;labelColor=20232a&amp;logo=pypi&amp;logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/holographic-sdk/"><img src="https://img.shields.io/badge/python-3.10%2B-007ec6?style=flat-square&amp;labelColor=20232a&amp;logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/writerslogic/holographic-memory/ci.yml?branch=main&amp;style=flat-square&amp;label=CI&amp;labelColor=20232a" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/writerslogic/holographic-memory"><img src="https://img.shields.io/ossf-scorecard/github.com/writerslogic/holographic-memory?style=flat-square&amp;labelColor=20232a" alt="OpenSSF Scorecard"></a>
  <a href="https://crates.io/crates/holographic-memory"><img src="https://img.shields.io/crates/v/holographic-memory?style=flat-square&amp;color=007ec6&amp;labelColor=20232a&amp;logo=rust" alt="crates.io version"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://github.com/sponsors/dcondrey"><img src="https://img.shields.io/badge/sponsor-dcondrey-EA4AAA?style=flat-square&amp;labelColor=20232a&amp;logo=githubsponsors&amp;logoColor=white" alt="Sponsor dcondrey"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#integrations">Integrations</a> &middot;
  <a href="#vector-masking">Vector Masking</a> &middot;
  <a href="https://github.com/writerslogic/holographic-memory">Core Engine</a>
</p>

---

The client talks to a service exposing `POST /api/v1/documents/batch`, `POST /api/v1/documents`, `DELETE /api/v1/documents/{id}` and `POST /api/v1/query`. The `holographic-memory` repository ships a reference implementation, `hms-server` (build with `--features server`); see `docs/SERVER.md`.

## Installation

```bash
pip install holographic-sdk
```

Framework adapters are optional extras. Install one, or all of them:

```bash
pip install "holographic-sdk[langchain]"
pip install "holographic-sdk[all]"
```

Requires Python 3.10 or newer.

## Quick Start

```python
from holographic_sdk import HolographicClient

client = HolographicClient(url="http://localhost:8080", api_key="...")

client.add_documents([
    {"id": "doc-1", "text": "Paris is the capital of France.", "vector": [0.12, -0.03, 0.88], "metadata": {"lang": "en"}},
])

response = client.query(query_vector=[0.10, -0.01, 0.90], top_k=3, filter={"lang": "en"})
client.delete_document("doc-1")
client.close()
```

`AsyncHolographicClient` has the same methods as coroutines; close it with `await client.aclose()`. Embeddings come from your own model: the SDK sends vectors, it does not compute them.

## Integrations

| Framework | Extra | Import |
|-----------|-------|--------|
| LangChain | `langchain` | `from holographic_sdk.langchain import HolographicVectorStore` |
| LlamaIndex | `llamaindex` | `from holographic_sdk.llamaindex import HolographicVectorStore` |
| Haystack | `haystack` | `from holographic_sdk.haystack import HolographicDocumentStore` |
| Semantic Kernel | `semantic-kernel` | `from holographic_sdk.semantic_kernel import HolographicMemoryStore` |
| DSPy | `dspy` | `from holographic_sdk.dspy import HolographicRM` |
| CrewAI | `crewai` | `from holographic_sdk.crewai import HolographicSearchTool` |
| Embedchain | `embedchain` | `from holographic_sdk.embedchain import HolographicDB` |
| Phidata | `phidata` | `from holographic_sdk.phidata import HolographicVectorDb` |
| Pydantic AI | `pydantic-ai` | `from holographic_sdk.pydantic_ai import get_holographic_tool` |
| SmolAgents | `smolagents` | `from holographic_sdk.smolagents import HolographicSearchTool` |

## Vector Masking

Passing `zero_trust_key` masks every embedding client-side with a keyed signed permutation (`VectorMask`, scrypt-derived). The server can still rank vectors because inner products and cosine similarity are preserved exactly. This is obfuscation, **not encryption**: the server learns all pairwise similarities, and known plain/masked pairs reveal the key for the coordinates they cover. Pass a per-collection `mask_salt` (at least 16 bytes) shared by all clients of that collection. Masks produced by 0.1.0/0.1.1 are not compatible and were not secure.

## License

AGPL-3.0-or-later. A commercial license is available from WritersLogic; see the [core repository](https://github.com/writerslogic/holographic-memory#licensing-dual-license-model).
