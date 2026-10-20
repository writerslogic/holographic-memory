# Zep importer (`zep`)

```sh
ZEP_API_KEY=... hms-import zep --hms-url http://127.0.0.1:8080 --embedder mypkg.enc:embed
hms-import zep --base-url https://zep.internal.example/api/v4 --kinds facts,entities ...
```

Requires `pip install 'holographic-sdk[import-zep]'`. Credentials: `ZEP_API_KEY` only. Zep memory is text, so the
server must embed text or you pass `--embedder`.

The importer uses the real `zep_cloud.Zep` client against the v4 API (cursor pagination, ordered by
`created_at` ascending) and imports four kinds (`--kinds` selects a subset):

| kind | Zep object | HMS id | text | metadata |
|---|---|---|---|---|
| `messages` | thread messages (`GET /threads`, `GET /threads/{uuid}/messages`) | `message:<uuid>` | `content` | `thread_id`, `thread_uuid`, `role`, `name`, `processed`, `reference_time`, `zep_metadata` (the message metadata) |
| `episodes` | graph episodes not tied to a thread (`POST /graphs/{uuid}/episodes/list`) | `episode:<uuid>` | `content` | `source`, `source_description`, `role`, `role_name`, `reference_time`, `document_id`, `processed`, `zep_metadata` |
| `facts` | graph edges (`.../edges/list`) | `fact:<uuid>` | `fact` | `relation`, source and target node uuid and name, `valid_at`, `invalid_at`, `expired_at`, `episode_uuids`, `attributes` |
| `entities` | graph nodes (`.../nodes/list`) | `entity:<uuid>` | `name: summary` | `name`, `labels`, `attributes` |

All records also get `kind` and `graph_uuid`. `created_at` maps to `hms_created_at`; a fact's `expired_at` to
`hms_updated_at`. Namespace: the user's `user_id` for a thread or a user graph, `graph:<graph_id>` for a standalone graph.

Enumeration: all users (`POST /users/list`), then all threads, then every graph (each user's `graph_uuid`
plus `POST /graphs/list`, deduplicated by uuid) in uuid order.

Episodes that belong to a thread (`thread_uuid` set) are the same text as that thread's messages and are skipped to avoid
importing it twice; messages are the authoritative copy. Items without text (a fact without `fact`, a node without name or summary) are rejected with their id.

## Verification status

MOCK: a stateful HTTP mock of the Zep v4 endpoints used (cursor paging, `Authorization: Api-Key`), driven by
the real `zep-cloud` SDK. Response shapes were taken from the SDK's generated models
(`UserPage`, `MessagePage`, `Edge`, `Episode`, `Node`; see <https://help.getzep.com/sdk-reference>). Not run
against Zep Cloud or a self-hosted Zep.

## Limits

- Zep ontology, summaries (`/threads/{uuid}/summary`) and context blocks are derived data and are not exported.
- Ordering by `created_at` is the server's; items created during a run may be missed and appear on the next (idempotent) run.
- Deleted-at-source items stay in HMS (no prune).
