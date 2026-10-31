from typing import Any, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext
from writerslogic_warm_sdk.client import WarmClient

class WarmDependencies(BaseModel):
    url: str = "http://localhost:8080"
    zero_trust_key: str = ""
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    embedder: Any = None

def get_warm_tool():
    """Return a Pydantic AI tool function for WARM search."""
    
    def search_writerslogic_warm(ctx: RunContext[WarmDependencies], query: str, top_k: int = 3) -> str:
        """Search WARM for relevant context."""
        deps = ctx.deps
        if not deps.embedder:
            return "Error: embedder function required in context dependencies."
            
        with WarmClient(
            url=deps.url,
            zero_trust_key=deps.zero_trust_key,
            api_key=deps.api_key,
            tenant_id=deps.tenant_id
        ) as client:
            embedding = deps.embedder(query)
            res = client.query(query_vector=embedding, top_k=top_k)
        
        results = [match.get("text", "") for match in res.get("matches", [])]
        return "\n\n---\n\n".join(results) if results else "No matches found."
        
    return search_writerslogic_warm
