# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["cbor2==5.9.0", "cryptography==47.0.0", "base58==2.1.1"]
# ///
"""Independent wire fixture for the bounded WARM RFC 9942 receipt profile.

Keys are deterministic public test material. No service or authority is contacted.
"""

import base64
import hashlib
import json

import base58
import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def encode(value):
    return cbor2.dumps(value, canonical=True)


def sign(key, protected, unprotected, payload, detached=False):
    headers = encode(protected)
    to_sign = encode(["Signature1", headers, b"", payload])
    signature = key.sign(to_sign)
    key.public_key().verify(signature, to_sign)
    return encode(cbor2.CBORTag(18, [headers, unprotected, None if detached else payload, signature]))


def b64(value):
    return base64.b64encode(value).decode("ascii")


def fixture():
    issuer = Ed25519PrivateKey.from_private_bytes(bytes([11]) * 32)
    service = Ed25519PrivateKey.from_private_bytes(bytes([12]) * 32)
    issuer_public = issuer.public_key().public_bytes_raw()
    service_public = service.public_key().public_bytes_raw()
    issuer_id = "did:key:z" + base58.b58encode(b"\xed\x01" + issuer_public).decode("ascii")
    service_id, subject = "https://transparency.fixture.invalid", "independent-fact"
    payload = encode({"fact_id": subject, "content_hash": hashlib.sha256(b"source evidence").digest(),
                      "timestamp_ms": 1000, "source_uri": "fixture:transcript/independent", "store_id": None})
    statement = sign(issuer, {1: -8, 3: "application/cbor", 4: issuer_public, 15: {1: issuer_id, 2: subject}}, {}, payload)
    sibling = hashlib.sha256(b"\x00independent second statement").digest()
    root = hashlib.sha256(b"\x01" + hashlib.sha256(b"\x00" + statement).digest() + sibling).digest()
    protected = {1: -8, 4: service_public, 15: {1: service_id, 2: subject}, 395: 1}
    proof = {396: {-1: [encode([2, 0, [sibling]])]}}
    receipt = sign(service, protected, proof, root, detached=True)
    damaged = receipt[:-1] + bytes([receipt[-1] ^ 1])
    wrong_identity = {**protected, 15: {1: "https://wrong.fixture.invalid", 2: subject}}
    wrong_proof = {396: {-1: [encode([2, 0, [bytes(32)]])]}}
    return {
        "profile": "WARM RFC9942 VDS1 detached Ed25519; independently encoded and signed in Python",
        "implementations": {"cbor2": "5.9.0", "cryptography": "47.0.0", "base58": "2.1.1"},
        "issuer_public": b64(issuer_public), "service_public": b64(service_public),
        "subject": subject, "service_issuer": service_id, "current_root": b64(root),
        "statement": b64(statement), "checkpoint_not_before_ms": 1000,
        "checkpoint_valid_until_ms": 3000, "now_ms": 2000,
        "vectors": [
            {"name": "valid", "receipt": b64(receipt), "valid": True},
            {"name": "wrong-signature", "receipt": b64(damaged), "valid": False},
            {"name": "wrong-identity", "receipt": b64(sign(service, wrong_identity, proof, root, True)), "valid": False},
            {"name": "wrong-proof", "receipt": b64(sign(service, protected, wrong_proof, root, True)), "valid": False},
            {"name": "attached-payload", "receipt": b64(sign(service, protected, proof, root)), "valid": False},
            {"name": "truncated", "receipt": b64(receipt[:12]), "valid": False},
        ],
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2, sort_keys=True))
