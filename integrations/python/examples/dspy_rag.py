# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""A DSPy RAG program whose retriever is Holographic Memory.

Start a service first:  HMS_API_KEY=dev hms-server --input-dim 384
Then:                   HMS_URL=http://127.0.0.1:8080 HMS_API_KEY=dev python dspy_rag.py

The language model is DSPy's DummyLM so the example runs offline; replace it with
`dspy.LM("openai/gpt-4o-mini")` (or any LiteLLM model) to answer for real.
"""

import os

import dspy
from dspy.utils import DummyLM

from holographic_integrations.dspy import HolographicRetriever

retriever = HolographicRetriever(
    k=2,
    url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
    api_key=os.environ.get("HMS_API_KEY"),
    tenant_id=os.environ.get("HMS_TENANT", "dspy-example"),
    input_dim=int(os.environ.get("HMS_INPUT_DIM", "384")),
)
retriever.add(
    [
        "Holographic Memory runs on your machine and makes no service calls.",
        "DSPy compiles declarative programs into prompts and weights.",
        "Pasta needs ripe tomato and basil.",
    ],
    ids=["hms", "dspy", "pasta"],
)


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        return self.answer(context=self.retrieve(question).passages, question=question)


with dspy.context(lm=DummyLM([{"reasoning": "the first passage says so", "answer": "on your machine"}])):
    out = RAG()("Where does Holographic Memory run?")
print(out.answer)
print(retriever("Where does Holographic Memory run?").ids)
retriever.delete(["hms", "dspy", "pasta"])
