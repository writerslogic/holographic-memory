# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""CrewAI adapter: `HolographicMemoryStorage` for `Memory(storage=...)` and
`HolographicKnowledgeStorage` for `Knowledge(storage=...)`. Requires `crewai`."""

from .knowledge import HolographicKnowledgeStorage
from .storage import HolographicMemoryStorage, similarity

__all__ = ["HolographicKnowledgeStorage", "HolographicMemoryStorage", "similarity"]
