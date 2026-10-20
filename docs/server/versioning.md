# hms-server API versioning policy

The HTTP contract is versioned by path prefix. Everything under `/api/v1` is the v1 contract. The unprefixed operational endpoints (`/healthz`, `/readyz`, `/metrics`, `/openapi.json`) are not versioned and follow the compatibility rules below.

The contract is the OpenAPI 3.1 document at `docs/server/openapi.json`, also served at `/openapi.json`. It is generated from the server's own route table and request and response types, and a test fails when the committed file differs from the build.

## Within v1 (additive only)

These changes can ship in any release without a new version:

- new endpoints, new optional request fields, new optional query parameters and headers;
- new response fields (clients must ignore fields they do not know);
- new error `code` values for new situations (clients must treat an unknown `code` by its HTTP status);
- tighter limits only when the previous value was unsafe, called out in the release notes;
- new scopes or tenants rules that only add access.

These changes never ship in v1:

- removing or renaming an endpoint, field, header or error `code`;
- changing the type, units or meaning of a field, or the score semantics (`score` is exact cosine for vector queries);
- making an optional request field required, or rejecting input v1 accepted;
- changing the status code of an existing outcome;
- changing the body shape of an error. The `error: {code, message}` member is deprecated and stays until v2; `type`, `title`, `status`, `detail`, `instance`, `code` and `request_id` are the supported members.

## Breaking changes

A breaking change is introduced as `/api/v2`. v1 and v2 are served side by side for at least two minor releases of the crate after v2 first ships, and the deprecation of v1 is announced in the release notes and, from then on, by a `Deprecation` header on v1 responses. The SDK pins its API version.

## Stored data

Store files are versioned separately by the engine. A server release that needs a store migration says so in its release notes and refuses to open a store it cannot read rather than altering it silently. The embedding model, revision and dimension recorded in a store are checked on every open.

## Configuration

Flags and environment variables follow the crate's semver: a flag or variable is not removed or repurposed in a patch or minor release; defaults that change behavior are listed in the release notes. `docs/server/configuration.md` is the reference.
