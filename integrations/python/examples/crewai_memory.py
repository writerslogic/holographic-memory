# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""CrewAI crew whose memory lives in WARM.

Start the service with the dimension of the embedder below:
    WARM_API_KEY=demo warm-server --input-dim 64 --dim 1024

Runs offline: the LLM is a scripted stand-in that reports what it saw in its prompt, and the
embedder is the dependency-free HashingEmbedder. Swap in a real `llm=` and `embedder=` (and
`--input-dim` to match) for real use.
"""

import os
from typing import Any, List, Optional

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.memory.unified_memory import Memory

from writerslogic_warm_integrations.core import HashingEmbedder, WarmConfig
from writerslogic_warm_integrations.crewai import WarmMemoryStorage

URL = os.environ.get("WARM_URL", "http://127.0.0.1:8080")
EMBED = HashingEmbedder(64)


class ScriptedLLM(BaseLLM):
    """Answers with a fixed final answer that quotes whether its prompt carried memory."""

    def call(self, messages: Any, tools: Optional[List[Any]] = None, callbacks: Any = None,
             available_functions: Any = None, from_task: Any = None, from_agent: Any = None,
             response_model: Any = None) -> str:
        text = messages if isinstance(messages, str) else " ".join(str(m.get("content", "")) for m in messages)
        seen = "launch codename is Bluebird" in text and "Relevant memories" in text
        return f"Thought: I know this.\nFinal Answer: memory in prompt={seen}"

    def supports_function_calling(self) -> bool:
        return False


def main() -> None:
    config = WarmConfig(url=URL, api_key=os.environ.get("WARM_API_KEY"), tenant_id="crewai_demo", input_dim=64)
    storage = WarmMemoryStorage(config)
    llm = ScriptedLLM(model="scripted")
    memory = Memory(storage=storage, embedder=EMBED.embed_documents, llm=llm, consolidation_threshold=1.0)
    storage.reset()
    memory.remember("The launch codename is Bluebird", scope="/project", categories=["fact"], importance=0.9)

    agent = Agent(role="Analyst", goal="Answer questions", backstory="Knows the project.", llm=llm)
    task = Task(description="What is the launch codename?", expected_output="the codename", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], memory=memory)
    print(crew.kickoff())
    for match in memory.recall("launch codename", depth="shallow", limit=3):
        print(f"{match.score:.3f}  {match.record.scope}  {match.record.content}")


if __name__ == "__main__":
    main()
