# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Custom Letta tools backed by WARM.

Letta runs custom tools in its own sandbox, so each tool is a self-contained function whose source is uploaded with
`client.tools.upsert`. It receives the injected `agent_state`, builds an `ArchivalMemory` from environment secrets
(`WARM_URL`, `WARM_API_KEY`, `WARM_TENANT_ID`, `WARM_INPUT_DIM`, `WARM_ZERO_TRUST_KEY`, `WARM_TIMEZONE`) and uses the agent id as the
collection. The sandbox needs the `writerslogic-warm-integrations[letta]` package and network access to your warm-server.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from writerslogic_warm_integrations.core import WarmConfig

from .archival import ArchivalMemory

TOOL_NAMES = ("warm_archival_insert", "warm_archival_search")

INSERT_SOURCE = '''
from typing import Optional


def warm_archival_insert(agent_state: "AgentState", content: str, tags: Optional[list[str]] = None) -> Optional[str]:
    """
    Add information to long-term archival memory for later retrieval.

    Use this tool to store facts, knowledge, or context that you want to remember across all future conversations.
    Store self-contained facts or summaries, not conversational fragments, and add descriptive tags.

    Args:
        content: The information to store. Should be clear and self-contained.
        tags: Optional list of category tags (e.g., ["meetings", "project-updates"])

    Returns:
        None. The memory is stored.
    """
    from writerslogic_warm_integrations.letta.tools import archival_from_env

    with archival_from_env() as memory:
        return memory.archival_memory_insert(agent_state.id, content, tags)
'''

SEARCH_SOURCE = '''
import json
from typing import Literal, Optional


def warm_archival_search(
    agent_state: "AgentState",
    query: str,
    tags: Optional[list[str]] = None,
    tag_match_mode: Literal["any", "all"] = "any",
    top_k: Optional[int] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
) -> Optional[str]:
    """
    Search archival memory using semantic similarity to find relevant information.

    Query by concept or meaning. Use tags to narrow results when you know the category.

    Args:
        query: What you're looking for, described naturally (e.g., "meetings about API redesign")
        tags: Filter to memories with these tags. Use tag_match_mode to control matching.
        tag_match_mode: "any" = match memories with ANY of the tags, "all" = match only memories with ALL tags
        top_k: Maximum number of results to return (default: 5)
        start_datetime: Only return memories created after this time (ISO 8601: "2024-01-15" or "2024-01-15T14:30")
        end_datetime: Only return memories created before this time (ISO 8601 format)

    Returns:
        A JSON list of relevant memories with IDs, timestamps, content and tags, ranked by similarity.
    """
    from writerslogic_warm_integrations.letta.tools import archival_from_env

    with archival_from_env() as memory:
        results = memory.archival_memory_search(agent_state.id, query, tags, tag_match_mode, top_k, start_datetime, end_datetime)
    return json.dumps(results)
'''


def archival_from_env(environ: Optional[Mapping[str, str]] = None) -> ArchivalMemory:
    """Build an `ArchivalMemory` from the `WARM_*` environment (what the uploaded tools run)."""
    env = os.environ if environ is None else environ
    dim = env.get("WARM_INPUT_DIM")
    config = WarmConfig(
        url=env.get("WARM_URL", "http://127.0.0.1:8080"),
        api_key=env.get("WARM_API_KEY") or None,
        tenant_id=env.get("WARM_TENANT_ID") or None,
        zero_trust_key=env.get("WARM_ZERO_TRUST_KEY", ""),
        input_dim=int(dim) if dim else None,
    )
    return ArchivalMemory(config, timezone=env.get("WARM_TIMEZONE", "UTC"))


def agent_secrets(url: str, *, api_key: Optional[str] = None, tenant_id: Optional[str] = None, input_dim: Optional[int] = None,
                  zero_trust_key: Optional[str] = None, timezone: Optional[str] = None) -> Dict[str, str]:
    """Environment for `client.agents.create(secrets=...)` so the uploaded tools can reach your service."""
    pairs = {"WARM_URL": url, "WARM_API_KEY": api_key, "WARM_TENANT_ID": tenant_id,
             "WARM_INPUT_DIM": None if input_dim is None else str(input_dim), "WARM_ZERO_TRUST_KEY": zero_trust_key, "WARM_TIMEZONE": timezone}
    return {k: v for k, v in pairs.items() if v}


def register_tools(client: Any, *, requirement: str = "writerslogic-warm-integrations", version: Optional[str] = None) -> List[Any]:
    """Upload both tools with `client.tools.upsert` and return the Letta `Tool` objects (attach by id)."""
    pip = {"name": requirement} if version is None else {"name": requirement, "version": version}
    return [
        client.tools.upsert(source_code=source.strip() + "\n", pip_requirements=[pip], tags=["writerslogic-warm"])
        for source in (INSERT_SOURCE, SEARCH_SOURCE)
    ]
