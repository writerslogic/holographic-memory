<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trust and provenance interoperability

The optional `provenance` feature issues signed fact records, status-list credentials and HMS sidecars. Enable it explicitly; the core memory engine does not require it. These are bounded interoperability checks, not certification.

| Specification or ecosystem | Integrated behavior | Evidence and limits |
|---|---|---|
| [W3C VC Data Model 2.0](https://www.w3.org/TR/vc-data-model/) and [Data Integrity EdDSA](https://www.w3.org/TR/vc-di-eddsa/) | RFC 8785 canonicalized `eddsa-jcs-2022` proofs, separately reported credential semantics and issuer trust. | Fresh Rust output passes independent Digital Bazaar proof verification and tamper rejection. Only the documented HMS fact profile and Ed25519 `did:key` assertion proofs are supported. |
| [Bitstring Status List 1.0](https://www.w3.org/TR/vc-bitstring-status-list/) | Signed GZIP/base64url lists; trusted URL-authorized publishers; rollback and bit-clearing rejection; bounded decoding; signed expiry checked at import and every cached lookup. | Rust expiry−1 ns, expiry and expiry+1 ns regressions cover both strict and optional freshness policies. Digital Bazaar accepts active index 7 and rejects revoked index 8. The caller hosts and refreshes HTTPS snapshots. Sequential allocation does not implement randomized index assignment. |
| [DID Core](https://www.w3.org/TR/did-core/) | Ed25519 `did:key`/`did:jwk` utilities and `did:web` documents/URLs. | Local utilities and proof-purpose checks; no general DID resolver. Hosts authenticate access claimants separately. |
| [CAWG Identity Assertion 1.1](https://cawg.io/identity/1.1/) and [Training and Data Mining Assertion 1.1](https://cawg.io/training-and-data-mining/1.1/) | COSE-signed authenticated containing-manifest references and a bounded four-entry training-policy subset. | Local tests reject altered, duplicate, cyclic and missing hard-binding references. Agent-credential admission checks claim-key issuer, subject, validity and proof. Full CAWG validation and external issuer authority remain separate. |
| [C2PA](https://spec.c2pa.org/specifications/) | HMS store manifests and sidecars use some C2PA-style labels. | These HMS formats are not C2PA Content Credentials. Native asset-signing/release tooling remains a separately retained candidate with failing macOS private-key integration tests; it is excluded from this integrated subset. |
| [DIF](https://identity.foundation/specs/) and [Trust over IP](https://trustoverip.org/our-work/technical-architecture/) | Identifiers, signed claims, trust anchors and a draft governance mapping. | Architectural relationships, not blanket compliance or independently governed deployment. |

The [draft v2 profile](provenance-profile-v1.md) specifies producer admission, consumer checks, compatibility, status publication and migration. Its established filename and v1 field namespace remain stable. Proof validity, semantic validity, status and issuer authorization must be assessed separately.

## Reproduction

Run `cargo test --locked --lib --no-default-features --features provenance core::provenance:: -- --test-threads=1`, scoped provenance Clippy, and `bash scripts/check-w3c-interop.sh`. The latter generates fresh credentials and uses pinned independent JavaScript implementations; it does not call a hosted model. The [integration receipt](evidence/integration-core-20261028.json) records inputs and actual results.

No production status endpoint, general VC/DID resolver, C2PA signing acceptance, CAWG conformance result, or certification is claimed. Local provenance checks do not measure AML retrieval or answer quality.
