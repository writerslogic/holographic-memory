from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from holographic_sdk.client import HolographicClient

class HolographicSearchSchema(BaseModel):
    query: str = Field(description="The query string to search for in the secure holographic memory.")

class HolographicSearchTool(BaseTool):
    name: str = "Holographic Secure Memory Search"
    description: str = "Search the zero-trust FHE-lite vector database for highly relevant secure context."
    args_schema: Type[BaseModel] = HolographicSearchSchema
    
    url: str = "http://localhost:8080"
    zero_trust_key: str = ""
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    timeout: int = 30
    embedder: Any = None
    top_k: int = 3

    def _run(self, query: str) -> str:
        if not self.embedder:
            return "Error: Embedder function must be provided to HolographicSearchTool."
        
        client = HolographicClient(
            url=self.url,
            zero_trust_key=self.zero_trust_key,
            api_key=self.api_key,
            tenant_id=self.tenant_id,
            timeout=self.timeout
        )
        
        embedding = self.embedder(query)
        res = client.query(query_vector=embedding, top_k=self.top_k)
        
        results = []
        for match in res.get("matches", []):
            results.append(match.get("text", ""))
            
        return "\n\n---\n\n".join(results) if results else "No matches found."
