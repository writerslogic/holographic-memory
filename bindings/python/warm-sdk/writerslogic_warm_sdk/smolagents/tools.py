from typing import Any, Optional
from smolagents.tools import Tool
from writerslogic_warm_sdk.client import WarmClient

class WarmSearchTool(Tool):
    name = "writerslogic_warm_search"
    description = "Searches the highly-secure Zero-Trust Holographic vector database for relevant context."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string to look up in the secure memory."
        },
        "top_k": {
            "type": "integer",
            "description": "The number of top results to return.",
            "default": 3
        }
    }
    output_type = "string"
    
    def __init__(
        self,
        embedder: Any,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedder = embedder
        self._client = WarmClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def forward(self, query: str, top_k: int = 3) -> str:
        if not self.embedder:
            return "Error: Embedder function must be provided to WarmSearchTool."
        
        embedding = self.embedder(query)
        res = self._client.query(query_vector=embedding, top_k=top_k)
        
        results = [match.get("text", "") for match in res.get("matches", [])]
        return "\n\n---\n\n".join(results) if results else "No matches found."
