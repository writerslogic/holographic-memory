from typing import Any, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from holographic_sdk.client import HolographicClient

class HolographicDependencies(BaseModel):
    url: str = "http://localhost:8080"
    zero_trust_key: str = ""
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    embedder: Any = None

def get_holographic_tool():
    """Returns a Pydantic AI Tool function for holographic memory search."""
    
    def search_holographic_memory(ctx: RunContext[HolographicDependencies], query: str, top_k: int = 3) -> str:
        """Search the highly-secure Zero-Trust Holographic memory for context."""
        deps = ctx.deps
        if not deps.embedder:
            return "Error: embedder function required in context dependencies."
            
        client = HolographicClient(
            url=deps.url,
            zero_trust_key=deps.zero_trust_key,
            api_key=deps.api_key,
            tenant_id=deps.tenant_id
        )
        
        embedding = deps.embedder(query)
        res = client.query(query_vector=embedding, top_k=top_k)
        
        results = [match.get("text", "") for match in res.get("matches", [])]
        return "\n\n---\n\n".join(results) if results else "No matches found."
        
    return search_holographic_memory
