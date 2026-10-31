---
search:
  boost: 1.6
---

# warm-server authentication and authorization

## Modes

| Mode | How | Who it is for |
|---|---|---|
| Per-tenant keys | `--keys-file keys.json` (`WARM_KEYS_FILE`) | Production. Keys are bound to tenants and scopes, stored only as SHA-256 digests, expire and rotate without a restart. |
| One shared key | `WARM_API_KEY` (environment only) | Small deployments and the original SDK setup. The key is valid for every tenant with every scope. May be combined with a key file. |
| No authentication | `--no-auth` (`WARM_NO_AUTH`) | Local development. Every caller has every scope. Refused unless the listener is loopback or a unix socket, refused together with credentials, and logged as a warning at startup. Requests from web pages are refused with 403 unless their `Origin` is in `--cors-origins`, and on TCP the `Host` must be a loopback name (`localhost`, `*.localhost`, a loopback address), which stops DNS rebinding. |

With neither a key nor `--no-auth` the server does not start. There is no implicit open mode.

## Tokens and storage

A token looks like `warm_<id>.<64 hex digits>`: the key id, then 256 bits from the operating system's random generator. It is shown once, by `warm-server keygen`. The server keeps `sha256(token)` only. A presented token is hashed and the digests are compared in constant time, so neither the comparison time nor the token length reveals anything about a stored value. SHA-256 is the right primitive here because tokens carry 256 bits of entropy; a slow password hash would only cost throughput.

The key id appears in access logs (`key_id`), never the secret.

## Creating keys

```sh
warm-server keygen --id acme-reader --tenants acme --scopes read --keys-file keys.json
# prints the token on stdout; the entry is appended to keys.json (mode 0600, written atomically)

warm-server keygen --id ops --tenants '*' --scopes admin --expires-in-days 90 --keys-file keys.json
```

Without `--keys-file` the entry is printed on stderr for you to place yourself.

## Key file format

```json
{
  "keys": [
    {
      "id": "acme-reader",
      "hash": "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
      "tenants": ["acme"],
      "scopes": ["read"],
      "not_before": 1767225600,
      "expires_at": 1798761600,
      "disabled": false
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | 1 to 32 characters of `[A-Za-z0-9_-]`, unique. |
| `hash` | `sha256:` plus 64 hex digits of the full token. |
| `tenants` | Tenant ids the key may use, or `["*"]` for every tenant (not combinable with ids). |
| `scopes` | Any of `read`, `write`, `admin`. |
| `not_before`, `expires_at` | Optional Unix seconds. Outside the window the key is rejected. |
| `disabled` | Optional. A disabled key is rejected. |

An invalid file is refused at startup. On reload, an invalid file is rejected and the previous keys stay in force.

## Scopes

| Scope | Allows |
|---|---|
| `read` | Query, list, stats, compaction status. |
| `write` | Add, replace and delete documents (single, batch, stream), index maintenance. |
| `admin` | Everything `read` and `write` allow, plus snapshots, restore, compaction runs, and `/metrics` (unless `--metrics-public`; a key limited to some tenants sees only their per-tenant series). Key reload and tenant listing additionally need a key valid for every tenant. |

`write` does not include `read`; give a service that ingests and queries both. The required scope of every operation is in the OpenAPI document as its `security` requirement, and a test checks the server against it.

## Tenants

A key bound to one tenant needs no `X-Tenant-ID`; its tenant is the default. A key bound to several tenants, or to `*`, selects with `X-Tenant-ID` (default `default`). Naming a tenant outside the key's list is 403 `tenant_forbidden`. Because a key can only reach its own tenants, a leaked key cannot create stores for tenants it was not issued for.

## Rotation

1. `warm-server keygen` a new key for the same tenants and scopes; add it to the key file.
2. Reload: `kill -HUP <pid>` or `POST /api/v1/admin/keys/reload` with an all-tenant admin key. Both keys are now valid.
3. Roll the new token out to clients.
4. Set `"disabled": true` (or remove the entry) on the old key and reload. Or issue keys with `--expires-in-days` so the old one lapses by itself.

Rotating `WARM_API_KEY` needs a restart; prefer a key file where rotation matters.

## What this does not do

- No per-request signing and no mutual TLS; transport security is TLS (`--tls-cert`) or a terminating proxy.
- No lockout of repeated failures. `warm_auth_failures_total` counts them; apply network-level controls for guessing attacks (the tokens are 256-bit random, so online guessing is not practical).
- No per-key rate limits; limits are per tenant (`--rate-limit-rps`).
