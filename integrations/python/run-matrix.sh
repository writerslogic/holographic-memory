#!/usr/bin/env bash
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs each adapter's tests in its own isolated environment (framework dependency trees conflict).
# Usage: WARM_SERVER_BIN=/path/to/warm-server ./run-matrix.sh [extra ...]
set -u
cd "$(dirname "$0")"
: "${WARM_SERVER_BIN:?set WARM_SERVER_BIN to an warm-server binary}"

# extra | extra to install | dependency groups | test paths
matrix=(
  "langchain|langchain|test-langchain|tests/langchain_adapter"
  "langgraph|langgraph|test-langgraph|tests/langgraph_adapter"
  "llamaindex|llamaindex||tests/llamaindex"
  "haystack|haystack||tests/haystack"
  "dspy|dspy||tests/dspy"
  "openai-agents|openai-agents||tests/openai_agents"
  "pydantic-ai|pydantic-ai||tests/pydantic_ai"
  "smolagents|smolagents||tests/smolagents"
  "litellm|litellm||tests/litellm"
  "crewai|crewai||tests/crewai"
  "autogen|autogen||tests/autogen/test_autogen_memory.py"
  "ag2|ag2||tests/autogen/test_ag2_toolkit.py"
  "semantic-kernel|semantic-kernel||tests/semantic_kernel"
  "mem0|mem0||tests/mem0"
  "letta|letta||tests/letta"
  "core|||tests/test_core.py tests/test_paged.py"
)

failed=()
for row in "${matrix[@]}"; do
  IFS='|' read -r name extra groups paths <<<"$row"
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
  args=(--isolated --python 3.12)
  [ -n "$extra" ] && args+=(--extra "$extra")
  for g in $groups; do args+=(--group "$g"); done
  echo "== $name"
  # shellcheck disable=SC2086
  uv run "${args[@]}" pytest -q $paths || failed+=("$name")
done
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "all passed"
