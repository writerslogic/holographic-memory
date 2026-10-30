<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Supported remote trust profiles

Implementation is provisional; authored vectors have not yet executed. The historical standards matrix remains historical. These profiles do not establish general VC, CAWG, C2PA, SCITT or OAuth certification.

## MCP resource server

The selected profile follows [MCP HTTP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization), [RFC 9728](https://www.rfc-editor.org/rfc/rfc9728.html), [RFC 6750](https://www.rfc-editor.org/rfc/rfc6750.html) and [RFC 7662](https://www.rfc-editor.org/rfc/rfc7662.html). The external authority implements authorization, consent, client registration, PKCE and [RFC 8414 discovery](https://www.rfc-editor.org/rfc/rfc8414.html). HMS publishes resource metadata and validates access tokens; it never issues or forwards client tokens to application services. Its authenticated introspection request is the token-validation protocol.

| Requirement | Implementation | Rejection/evidence vector |
|---|---|---|
| Discover protected resource and authority | Configured resource URL, well-known path and challenge; no Host-derived trust | Real HTTP unauthenticated metadata and 401 challenge |
| Correct token authority and resource | Pinned HTTPS introspection endpoint on issuer origin; exact `iss`, audience member, subject and bearer type | Wrong/missing issuer, audience, subject, type |
| Least privilege and tenant scope | `hms:read`, write-tool classification shared with exposure policy; exact authority-issued `hms_namespace`; fixed store namespace | Read-only token write returns 403; another namespace returns 401; override configuration refused |
| Expiry/revocation | Fresh introspection per request, integer expiry, optional not-before/issued-at; recheck expiry after body read | Expiry minus one/equal/plus one; revoked second request fails |
| Session/replay safety | rmcp stateless service for all revisions; session headers rejected | Stolen session ID fails; initialized response has no session ID |
| Bound outbound work and parsing | Eight retained worker permits, five-second client deadline, 8 KiB token, 16 KiB response, no retries/cache/redirects/proxy; fixed endpoint | Token boundaries, oversized response, existing transport cancellation/admission vectors; final slow-authority gate |
| No ambiguous sender proof | Reject `cnf` and non-bearer types | Sender-constrained token cannot silently become bearer |

Bearer replay during valid authorization is inherent in this selected profile. Revocation is observed on the next introspection; a mutation already admitted may complete. HTTPS authority configuration and DNS/network administration remain deployment trust. Introspection keys/rotation/status are managed by that authority, avoiding a local JWKS or status cache. Standard HTTP Host/Origin checks and the existing connection/body/response/cancellation limits remain active. Authorization adds one authority round-trip and bounded body parsing per request; latency and throughput are **unmeasured**.

## SCITT detached receipt

`scitt::receipt::verify_receipt` implements a restricted [RFC 9942 inclusion receipt](https://www.rfc-editor.org/rfc/rfc9942.html#section-5.2.1) with [RFC 9162 SHA-256 tree semantics](https://www.rfc-editor.org/rfc/rfc9162.html#section-2.1.3.2) and [RFC 9943 statement/receipt identities](https://www.rfc-editor.org/rfc/rfc9943.html#section-6.3). Supported inputs are Ed25519 HMS `application/cbor` fact statements and detached COSE Sign1 receipts with VDS 1, one inclusion proof, tree size greater than one, and protected CWT issuer/subject. Unknown critical headers fail closed. Consistency proofs, alternate algorithms/VDSs, embedded receipts and general claim schemas are not supported by this function.

| Requirement | Implementation | Rejection/evidence vector |
|---|---|---|
| Independently trusted statement issuer | Caller issuer TrustStore, exact key ID, did:key CWT issuer and subject, COSE strict Ed25519 verification | Untrusted issuer; altered statement with same subject |
| Independently trusted service | Separate service TrustStore, caller service issuer binding, exact service key ID and subject, signed reconstructed root | Wrong service trust; altered signature; unknown critical header |
| Inclusion of supplied bytes | Empty statement unprotected headers, RFC 9162 leaf/node domain separation and path reconstruction | Recursive tree oracle for every leaf in sizes 2–17; missing/excess path nodes; wrong leaf position |
| Current expected checkpoint | Root equality plus caller-provisioned validity interval checked against caller time | Wrong/stale root; time before interval and at expiry |
| Bounded decode | Receipt ≤4096 B, proof ≤2304 B and ≤64 SHA-256 nodes; CBOR depth 32 (proof 8); statement ≤1 MiB | Receipt overflow/truncation; path 63/64/65; hash size 31/32/33 |
| Distinct verification result | Private fields of `VerifiedReceipt`; only verifier constructs it | Unverified submission remains `UnverifiedReceipt`; no implicit promotion |

The relying party must provision issuer/service keys and the service-issuer binding, authenticated current root, validity interval and trustworthy current time. Those policy values must come from a separately authenticated service checkpoint or administrative distribution channel, not from the receipt being checked. Removing a key from its trust store revokes that local anchor. This function does not discover checkpoints, prove append-only consistency/non-equivocation, or claim that receipt metadata is fresh merely because a signature verifies. The experimental `/entries` submission transport remains explicitly unverified and is not a general RFC 9943 registration client.

Independent review inputs are the exact signed statement, receipt bytes, pinned root and validity interval, both public keys and issuer identities, the relevant trust-store configuration, and source/artifact hashes. Deterministic tests use fixed nonproduction signing keys and separate recursive tree construction. An independently produced cross-implementation signed receipt remains a final interoperability/external evidence requirement; no certification follows from local tests.

## Deferred execution

Use the final session's repository-owned `CARGO_TARGET_DIR`. Run `cargo test --offline --locked --features mcp --bin hms-mcp`, `cargo test --offline --locked --features provenance-scitt --lib core::provenance::scitt::`, and the retained COSE malformed-header tests. The combined gate must include slow/failed authority cancellation, admission saturation, no redirects, TLS peer validation, wrong-trust and stale-checkpoint artifacts, the existing TypeScript MCP interop suite, and all-feature lint. No remote authority, paid service, model call or broad test was run during implementation.
