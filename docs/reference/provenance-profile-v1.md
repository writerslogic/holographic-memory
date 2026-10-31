<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WARM provenance profile v2 (draft)


**Identifier:** `urn:hms:provenance:v2`
**Status:** Experimental draft. The profile describes WARM artifacts; it is not a certification from W3C, C2PA, CAWG, DIF, or Trust over IP.

The established `provenance-profile-v1.md` documentation URL is retained. The current verification profile is v2; the v1 JSON-LD context and fact-subject field namespace remain unchanged.

## Artifacts and verification

| Artifact | Producer | Consumer checks |
|---|---|---|
| Fact credential | WARM `provenance` feature | Parse VC Data Model 2.0 JSON, resolve the Ed25519 `did:key` verification method, verify the `eddsa-jcs-2022` proof, check the WARM fact credential's semantic constraints, and apply an independent trust decision to the issuer. These appear as separate `vc_proof`, `vc_semantics`, and `trust_anchor` verification results. `contentHash` is SHA-256 of the fact content; a consumer with the content recomputes it. The WARM-specific terms are defined by the [v1 JSON-LD context](fixtures/warm-v1-context.jsonld). |
| Revocation list | `WarmCore::status_list_credential()` | The caller obtains a list snapshot and supplies it to `TrustStore`; WARM verifies its proof and issuer, requires explicit publisher authorization for the credential URL, decodes base64url Multibase and GZIP, and reads the index from the leftmost bit. An out-of-range index or stale required snapshot fails; a set bit means revoked. Missing status is explicitly reported as unchecked unless strict policy is enabled. |
| CAWG identity assertion | WARM CAWG Identity Claims Aggregation writer | Verify its COSE signature and signed `c2paAsset` references in authenticated containing-claim context. The WARM sidecar verifier compares those references with its signed manifest assertion entries, enforces a hard binding, and rejects duplicate/cyclic references. A real asset requires its C2PA claim validator. Apply a separate policy to the identity issuer. WARM currently targets [CAWG Identity Assertion 1.1](https://cawg.io/identity/1.1/). |

The optional `cawg.training-mining` policy in the WARM store manifest uses the four predefined entries from [CAWG Training and Data Mining Assertion 1.1](https://cawg.io/training-and-data-mining/1.1/). WARM accepts `allowed`, `notAllowed`, or `constrained`, with an optional `constraint_info` of at most 2,048 bytes. Custom entries and metadata are outside this profile's producer subset.

The sidecar producer now authenticates `AgentProvenanceParams.agent_credential` before construction: VC 2.0 context/type, current validity, issuer equal to the claim key's `did:key`, agent-subject binding, bounded structure/serialization and EdDSA proof. It rejects unsigned/altered/wrong-agent/wrong-issuer/future/expired credentials and status-bearing credentials requiring an unsupported trusted resolver. This is a local claim-key-issued admission policy. The credential is not added to the sidecar wire format; consumer interpretation and verification rules do not change. Old sidecars remain readable and gain no retroactive credential authority. External issuer authorization and full CAWG Identity 1.1 validation remain separate requirements.

The WARM-specific fields have [a fact subject schema](fixtures/warm-fact-subject-v1.schema.json) and [a v2 C2PA assertion schema](fixtures/warm-c2pa-assertion-v2.schema.json). An [offline sample](fixtures/w3c-sample.json) contains active and revoked fact credentials and their signed status list. Its `example.com` status URL is illustrative; it is not a published endpoint. The docs build installs the v1 context at `https://docs.holographic-memory.com/ns/hms/v1.jsonld`; that URL needs deployment before external JSON-LD processing can rely on it. These extensions do not redefine standard fields.





## Issuing and publishing

Enable the optional `provenance` Cargo feature and set `ProvenanceConfig.enabled`. Set `ProvenanceConfig.status_list_url` to a stable HTTPS URL before issuing externally verifiable credentials. The same URL appears in each new credential's `credentialStatus.statusListCredential` and in the exported status-list credential's `id`. Publish the serialized result of `WarmCore::status_list_credential()` at that URL after issuance or revocation. The library exports the signed JSON; it does not host the URL or refresh a remote copy. The URL must continue to serve the latest signed list while its credentials remain in circulation.

When importing a list, first trust its issuer key, then call `TrustStore::authorize_status_list_publisher(url, key)`, and only then call `trust_status_list_credential`. WARM verifies the signed list and rejects older snapshots and attempts to clear revocation bits. `trust_status_list(url, bits)` instead accepts a caller-asserted raw bitstring without proof or signed time; it is suitable for manager-owned live state, not independent published-list verification. Enable `require_status_lists(true)` when a missing list must fail verification, or `require_recent_status_lists(max_age_seconds)` to require a signed snapshot within a chosen age; a stale or future-dated signed snapshot fails explicitly, as does an out-of-range index. Missing status is reported as unchecked unless strict policy is enabled. Retrieval is the caller's responsibility within this local-first library: the caller must authenticate the configured HTTPS source, bound its response size/time, import a fresh signed snapshot, and choose the age policy. WARM does not fetch URLs. WARM assigns sequential status indexes today, which can reveal issuance order; deployments with stricter privacy requirements need a different allocation policy before issuing credentials.

Legacy WARM credentials use a `did:key` status-list URI, an older `#key-0` proof method, and the former context URL. WARM retains cryptographic verification of those stored records for compatibility. They cannot be treated as v2's portable status-list flow or assumed to satisfy v2 semantics and trust policy. The current `did:key` proof method ends in the key's multibase value, as required by the DID method.

## Evidence and limits

Integration validation passed 121 focused provenance tests, scoped Clippy, and fresh independent Digital Bazaar proof/status-list verification. The [integration receipt](evidence/integration-core-20261028.json) binds this evidence to source inputs. No live status endpoint or production C2PA configuration was exercised.

Run [the W3C interop check](https://github.com/writerslogic/warm/blob/main/scripts/check-w3c-interop.sh) to generate fresh credentials in Rust and verify their VC model and EdDSA proofs with Digital Bazaar's JavaScript implementation. Its Bitstring Status List reader accepts index 7, rejects revoked index 8, and the proof verifier rejects altered documents. Versions are pinned in `interop/w3c/package-lock.json`.

These checks cover specific artifact formats and cryptographic paths. They do not establish a published status endpoint, general VC semantic validation beyond the WARM fact profile, full CAWG validation, or certification. The separately retained native C2PA signing toolkit is outside this integrated subset.

## Migration from v1

The stricter credential semantics, status-snapshot trust rules, and authenticated CAWG references require `urn:hms:provenance:v2`. A v1 artifact is not automatically v2-valid. The v1 JSON-LD context and fact-subject namespace remain unchanged; the v1 and v2 assertion schemas describe WARM sidecar formats.

Consumers must evaluate `vc_proof`, `vc_semantics`, and `trust_anchor` separately; proof success alone does not establish valid provenance. Configure a stable HTTPS status URL before new issuance, trust the list publisher key, and authorize it for that exact URL before importing signed snapshots. Import verifies `validFrom` and any `validUntil` syntax, interval, and expiry. Every later cached lookup also enforces the signed expiry, including when optional age policy is disabled. Refresh snapshots under a caller-chosen age policy; reject rollback, cleared revocation bits, and required missing, stale, or out-of-range status. Retrieval remains caller owned. CAWG references must match the authenticated containing manifest, including its hard binding; key control alone establishes neither authority over an external agent credential nor full CAWG validation.

Legacy signatures remain cryptographically readable through the documented compatibility path. A v1 artifact is not automatically v2-valid and must not be relabeled without reissuing and signing under v2's rules. Fact credentials do not gain a new profile field: select v2 verification policy in the consuming application. Publication, cross-implementation review, and production governance approval remain outstanding.


## Governance

WritersLogic stewards this draft and its namespace. Changes to a signed field, verifier rule, or trust rule require a new profile version and a public migration note. Compatible clarifications can retain the current version if existing valid artifacts keep the same interpretation. Proposed changes should include cross-implementation fixtures and a recorded review before publication. A production governance framework also needs named issuers and relying parties, key recovery and revocation procedures, dispute handling, and independent participation. Those parties and decisions are not established by this repository.

This maps to the [Trust over IP Technology Architecture v1.0 PR1 public-review draft](https://trustoverip.org/wp-content/uploads/ToIP-Technical-Architecture-Specification-V1.0-PR1-2022-11-14.pdf) as a technical layer (identifiers, credentials, and evidence) plus proposed governance rules. It is not a [ToIP Governance Architecture v1.0](https://trustoverip.org/wp-content/uploads/ToIP-Governance-Architecture-Specification-V1.0-2022-12-21.pdf) governance framework or a ToIP conformance result. DIF's relationship here is the named CAWG Identity and Training and Data Mining 1.1 specifications, not blanket DIF compliance. Evidence of a de facto standard will require adoption and verification by organizations beyond WritersLogic.
