# Letta importer (`letta`)

```sh
LETTA_API_KEY=... hms-import letta --hms-url http://127.0.0.1:8080 --embedder mypkg.enc:embed
hms-import letta --base-url http://localhost:8283 --kinds passages,blocks,messages --agent-id agent-1234
```

Requires `pip install 'holographic-sdk[import-letta]'`. Credentials: `LETTA_API_KEY` (optional for an unauthenticated
self-hosted server). Without `--base-url` the SDK default (Letta Cloud) is used.

The importer uses the real `letta_client.Letta` SDK. Agents come from `GET /v1/agents/` (or `--agent-id`,
repeatable); per agent, `--kinds` selects:

| kind (default) | Endpoint | HMS id | text | metadata |
|---|---|---|---|---|
| `passages` (yes) | `GET /v1/agents/{id}/archival-memory` (`after`, `limit`) | `passage:<id>` | `text` | `tags`, `archive_id`, `file_id`, `file_name`, `source_id`, `letta_metadata`, `agent_id`, `agent_name` |
| `blocks` (yes) | `GET /v1/agents/{id}/core-memory/blocks` | `block:<agent>:<block id>` | `value` | `label`, `description`, `limit`, `read_only`, `tags`, `block_id`, `letta_metadata` |
| `messages` (no) | `GET /v1/agents/{id}/messages` | `message:<id>` | message content, or the reasoning, tool return or tool call payload | `message_type`, `otid`, `name`, `run_id` |

Namespace is the agent id. Timestamps: passage `created_at`/`updated_at`, message `date`; blocks carry none.
Messages with no text at all are skipped and listed under `skipped.empty` in the report.

Vectors: passage embeddings are not carried by default (they use the agent's own embedding model). `--with-vectors` carries
the ones the server returns; they must have the HMS dimension, and a passage with no embedding is then rejected rather than
encoded in another space (see the `auto` vector policy).

## Verification status

MOCK: a stateful HTTP mock of the endpoints above (`after` cursor paging, `Authorization: Bearer`), driven by the
real `letta-client` 1.12.1 SDK (<https://docs.letta.com/api-reference>; response shapes from the SDK models `Passage`,
`BlockResponse`, message types). Not run against Letta Cloud or a self-hosted server.

## Limits

- Files and folders attached to an agent, tools, identities and run history are not imported; their passages appear only if exposed as archival memory.
- Shared blocks are imported once per agent that has them (id includes the agent).
- Deleted-at-source items stay in HMS (no prune).
