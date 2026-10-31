# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Generic OpenAI-format memory tools for LiteLLM (or any OpenAI-compatible client).

LiteLLM has no memory or vector-store interface, so "adapter" here means:

- `tools_schema()`: OpenAI-format function-tool definitions (`remember`, `recall`,
  `forget`) to pass as `tools=`;
- `handle_tool_call(tool_call)` / `ahandle_tool_call`: execute one tool call and return the
  `{"role": "tool", ...}` message to append. It accepts LiteLLM and OpenAI SDK tool-call
  objects and plain dicts; malformed arguments or unknown tools come back as an error
  message for the model rather than raising;
- `run` / `arun`: a bounded completion loop that executes the model's memory tool calls
  until it answers.

Nothing here imports litellm except the default `completion` of `run`/`arun`, so the schema
and dispatcher also serve the OpenAI, Azure, Mistral or any other SDK with the same format.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from ..core import AsyncWarmBackend, WarmBackend, WarmConfig, Record

_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a fact or note in long-term memory. Returns the memory id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact or note to remember."},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search long-term memory. Returns the best matches, one `[id] text` per line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "k": {"type": "integer", "description": "How many matches to return."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete one memory by id.",
            "parameters": {
                "type": "object",
                "properties": {"memory_id": {"type": "string", "description": "Id from remember or recall."}},
                "required": ["memory_id"],
            },
        },
    },
]


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _parse_call(tool_call: Any) -> tuple[str, str, Any]:
    """(call id, function name, raw arguments) from an SDK object or a dict."""
    fn = _field(tool_call, "function")
    if fn is None:
        raise ValueError("tool call has no function")
    return str(_field(tool_call, "id") or ""), str(_field(fn, "name") or ""), _field(fn, "arguments")


def _args(raw: Any) -> Dict[str, Any]:
    if raw in (None, ""):
        return {}
    value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return dict(value)


def _format(hits) -> str:
    if not hits:
        return "No matching memories."
    return "\n".join(f"[{h.id}] {h.text}" for h in hits)


class MemoryTools:
    """The memory tools bound to one store.

    `user_id` is stored as metadata and restricts `recall` and `forget` to that user's
    memories. Pass `config` (an `WarmConfig`) or ready `backend`/`async_backend` objects; the
    sync and async backends are created lazily so either path can be used alone.
    """

    def __init__(
        self,
        config: Optional[WarmConfig] = None,
        *,
        backend: Optional[WarmBackend] = None,
        async_backend: Optional[AsyncWarmBackend] = None,
        user_id: Optional[str] = None,
        default_k: int = 4,
        max_k: int = 20,
    ):
        self.config = config or (backend.config if backend else async_backend.config if async_backend else WarmConfig())
        self._backend, self._abackend = backend, async_backend
        self.scope = {"user_id": user_id} if user_id is not None else {}
        self.default_k, self.max_k = default_k, max_k

    @property
    def backend(self) -> WarmBackend:
        if self._backend is None:
            self._backend = WarmBackend(self.config)
        return self._backend

    @property
    def async_backend(self) -> AsyncWarmBackend:
        if self._abackend is None:
            self._abackend = AsyncWarmBackend(self.config)
        return self._abackend

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    async def aclose(self) -> None:
        if self._abackend is not None:
            await self._abackend.aclose()

    @staticmethod
    def tools_schema() -> List[Dict[str, Any]]:
        """OpenAI-format tool definitions. A fresh deep copy, safe to mutate."""
        return json.loads(json.dumps(_SCHEMA))

    # -- dispatch ----------------------------------------------------------------------------

    def _clamp(self, k: Any) -> int:
        return max(1, min(int(k) if k else self.default_k, self.max_k))

    def _owned(self, record: Optional[Record]) -> bool:
        return record is not None and all(record.metadata.get(a) == b for a, b in self.scope.items())

    def _record(self, args: Dict[str, Any]) -> Record:
        text = args.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("remember requires a non-empty string 'text'")
        meta = dict(self.scope)
        if args.get("tags"):
            meta["tags"] = [str(t) for t in args["tags"]]
        return Record(uuid.uuid4().hex, text, meta)

    @staticmethod
    def _query(args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("recall requires a non-empty string 'query'")
        return query

    @staticmethod
    def _id(args: Dict[str, Any]) -> str:
        memory_id = args.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("forget requires a string 'memory_id'")
        return memory_id

    @staticmethod
    def _message(call_id: str, name: str, content: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}

    def handle_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Run one tool call and return the `role: tool` message answering it."""
        call_id, name, raw = _parse_call(tool_call)
        try:
            args = _args(raw)
            if name == "remember":
                return self._message(call_id, name, self.backend.remember([self._record(args)])[0])
            if name == "recall":
                hits = self.backend.recall(self._query(args), k=self._clamp(args.get("k")), filter=self.scope or None)
                return self._message(call_id, name, _format(hits))
            if name == "forget":
                memory_id = self._id(args)
                if not self._owned(self.backend.get([memory_id])[0]):
                    return self._message(call_id, name, "No such memory.")
                self.backend.forget([memory_id])
                return self._message(call_id, name, "Deleted.")
            return self._message(call_id, name, f"Error: unknown tool {name!r}")
        except (ValueError, TypeError) as exc:  # bad arguments are the model's to fix; service errors propagate
            return self._message(call_id, name, f"Error: {exc}")

    async def ahandle_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        call_id, name, raw = _parse_call(tool_call)
        try:
            args = _args(raw)
            if name == "remember":
                return self._message(call_id, name, (await self.async_backend.remember([self._record(args)]))[0])
            if name == "recall":
                hits = await self.async_backend.recall(
                    self._query(args), k=self._clamp(args.get("k")), filter=self.scope or None
                )
                return self._message(call_id, name, _format(hits))
            if name == "forget":
                memory_id = self._id(args)
                if not self._owned((await self.async_backend.get([memory_id]))[0]):
                    return self._message(call_id, name, "No such memory.")
                await self.async_backend.forget([memory_id])
                return self._message(call_id, name, "Deleted.")
            return self._message(call_id, name, f"Error: unknown tool {name!r}")
        except (ValueError, TypeError) as exc:
            return self._message(call_id, name, f"Error: {exc}")

    # -- loops -------------------------------------------------------------------------------

    @staticmethod
    def _assistant_message(message: Any) -> Dict[str, Any]:
        dump = getattr(message, "model_dump", None)
        data = dump(exclude_none=True) if dump else dict(message)
        data = {k: v for k, v in data.items() if k in ("role", "content", "tool_calls")}
        data.setdefault("role", "assistant")
        return data

    def run(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        completion: Optional[Callable[..., Any]] = None,
        max_steps: int = 8,
        **kwargs: Any,
    ) -> Any:
        """Call `completion` (default `litellm.completion`) until the reply has no tool calls.

        `messages` is extended in place with every assistant and tool message. Returns the
        final response; raises `RuntimeError` after `max_steps` calls that all asked for tools.
        """
        if completion is None:
            import litellm

            completion = litellm.completion
        kwargs.setdefault("tools", self.tools_schema())
        for _ in range(max_steps):
            response = completion(model=model, messages=messages, **kwargs)
            message = response.choices[0].message
            calls = _field(message, "tool_calls")
            if not calls:
                return response
            messages.append(self._assistant_message(message))
            messages.extend(self.handle_tool_call(c) for c in calls)
        raise RuntimeError(f"model still requested tools after {max_steps} steps")

    async def arun(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        completion: Optional[Callable[..., Awaitable[Any]]] = None,
        max_steps: int = 8,
        **kwargs: Any,
    ) -> Any:
        """Async `run`; the default `completion` is `litellm.acompletion`."""
        if completion is None:
            import litellm

            completion = litellm.acompletion
        kwargs.setdefault("tools", self.tools_schema())
        for _ in range(max_steps):
            response = await completion(model=model, messages=messages, **kwargs)
            message = response.choices[0].message
            calls = _field(message, "tool_calls")
            if not calls:
                return response
            messages.append(self._assistant_message(message))
            for call in calls:
                messages.append(await self.ahandle_tool_call(call))
        raise RuntimeError(f"model still requested tools after {max_steps} steps")
