---
name: holographic-memory
description: Use persistent memory across sessions. Load when the user asks to remember, recall, forget or review something, when earlier decisions may matter for the task, or when a durable fact, preference or decision is worth keeping.
---

# Holographic Memory

Memory lives in an MCP server (`holographic-memory`) that stores everything on this machine. Its tools are `remember`, `recall`, `search`, `get`, `list`, `forget`, `stats` and `export`.

## Recall before work

At the start of a task, call `recall` with a short natural-language description of the task. Use what comes back; say so when a memory looks out of date and update it.

## What to remember

Remember facts that will still matter in a later session: decisions and the reason for them, conventions, user preferences, environment quirks, open questions. Do not remember what the code or git history already shows, secrets, or step-by-step logs.

- One topic per memory. Keep text short and self-contained.
- Give a stable `id` such as `decisions/auth-model` or `prefs/testing`. Writing the same id replaces the memory, which is how you keep it current.
- Add `tags` for grouping (for example `decision`, `convention`) and `title` for a readable name.
- Use `if_version` (from `get`) when you read, change and write back, so a concurrent change is not overwritten.

## Reviewing and forgetting

- `list` with a `prefix` or `tag` pages through memories; `get` reads one in full.
- `forget` removes by id, or by `prefix` with `confirm: true`. Check with `list` first.
- `export` returns memories as json, jsonl or markdown for backup or sharing.

## Search and recall

`recall` returns the best matches with full text. `search` adds filters (`tags`, `filter` on metadata), weights and fusion, and returns previews with the score breakdown. Without an embedding model the ranking is lexical (BM25), so use the words a memory would contain; `stats` reports the active mode.

## Limits

A memory is at most the server's `max_text_bytes` (default 32 KiB). Split longer notes into several memories with related ids.
