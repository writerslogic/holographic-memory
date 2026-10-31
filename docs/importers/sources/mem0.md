# mem0 importers (`mem0`, `mem0-platform`)

```sh
# open-source library: point at the same config the application used
warm-import mem0 --config mem0-config.json --with-vectors --warm-url http://127.0.0.1:8080
# hosted platform
MEM0_API_KEY=... warm-import mem0-platform --warm-url http://127.0.0.1:8080 --embedder mypkg.enc:embed
```

Requires `uv add 'writerslogic-warm-sdk[import-mem0]'`. mem0 memories are text, so unless `--with-vectors`
carries the stored embeddings (OSS only), the server must embed text or you pass `--embedder`
(see the vectors section of the main importer docs). With `--vectors auto` a source is carried or re-encoded as a whole, never mixed.

## `mem0` (open-source library, `mem0ai`)

`--config` is the dictionary given to `Memory.from_config` (JSON file or inline). The importer opens the
same `Memory`, then scans its vector store directly, because `Memory.get_all` has no pagination and
needs a scope id.

| Store | Reading | Resume |
|---|---|---|
| qdrant (local path or server) | `scroll` over the whole collection | seekable: page offset plus index in page |
| chroma | `collection.get(limit, offset)` | seekable: offset plus index |
| any other store | `Memory.get_all(user_id/agent_id/run_id)` for each `--user-id/--agent-id/--run-id` | not seekable |

For the other stores a scope that returns `--scope-limit` (default 100000) or more memories fails the
import instead of truncating.

Field mapping (mem0 payload to WARM):

| mem0 | WARM |
|---|---|
| point id | document id (`<namespace>::<id>` for a non-empty namespace) |
| `data` | `text` |
| `created_at`, `updated_at` | `warm_created_at`, `warm_updated_at` (UTC ISO 8601; unparsable values kept raw in `warm_*_raw`) |
| `user_id` / `agent_id` / `run_id` | namespace `alice`, `agent:a1`, `run:r7` (first present of user, agent, run); the raw ids stay in metadata |
| `hash`, `actor_id`, `role` and every custom metadata key | metadata, flat, so filters work |
| stored embedding | `vector`, only with `--with-vectors` |

Plus `kind: "memory"`. A payload without `data` is rejected by id (`source_unreadable`), never dropped.

## `mem0-platform` (hosted API)

Uses the real `mem0.MemoryClient`. Credentials: `MEM0_API_KEY` only (a flag would show in the process
list); `--host`, `--org-id`, `--project-id` as in the SDK. Entities come from `GET /v1/entities/`
(following `next`), or `--user-id/--agent-id/--app-id/--run-id` restrict to given scopes. For each entity
memories are listed with `POST /v2/memories/` (`filters={<scope>: <id>}`, `page`, `page_size` 1..200).

A memory with several scopes (for example a user and an agent) is returned by several queries; it is
imported once, under the first scope present in the order user, agent, app, run. The v2 listing reports the run as `session_id`.

| Platform field | WARM |
|---|---|
| `id` | document id |
| `memory` | `text` |
| `created_at`, `updated_at` | `warm_created_at`, `warm_updated_at` (ISO strings and epoch seconds both handled) |
| `user_id`, `agent_id`, `app_id`, `session_id` | namespace as above; also kept in metadata |
| `metadata` | flattened into metadata and kept whole as `mem0_metadata` |
| `categories`, `immutable`, `expiration_date`, `hash`, `structured_attributes`, `replaced_by`, `synthesized` | metadata, as returned |

## Verification status

| What | Verified against |
|---|---|
| `mem0` with a local qdrant store | REAL: mem0ai 0.1.115 `Memory` populated with `add(infer=False)` (embedder stubbed, no network), then imported and checked against a real warm-server, including carried-vector recall 1.0 and mid-page resume |
| `mem0` with a chroma store | REAL: mem0ai with chromadb |
| `mem0-platform` | MOCK: a stateful HTTP mock of the documented API driven by the real `MemoryClient` (pagination, entity paging, auth header, scope dedupe). Not run against api.mem0.ai |

Documentation used: <https://docs.mem0.ai/api-reference/memory/v2-get-memories> and
<https://docs.mem0.ai/api-reference/entities/get-users>. The current docs describe a v3 listing
(`POST /v3/memories/`); the installed SDK (0.1.115) calls `/v2/memories/` with the same filter body and
envelope (`count`, `next`, `previous`, `results`), which is what is implemented and mocked. A newer SDK that moves to
v3 needs `version="v3"` in `Mem0PlatformSource.records` and a re-check of the mock.

## Limits

- Platform memories deleted or expired (`show_expired` false) after the run started may or may not appear.
- mem0 graph-store relations (`enable_graph`) are not exported.
- mem0's history table (change log) is not imported.
- Re-running replaces owned documents by id; memories deleted at the source stay in WARM. The importer does not mirror source deletions.
