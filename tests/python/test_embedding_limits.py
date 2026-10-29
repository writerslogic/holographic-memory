# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

import holographic_memory as hm


@pytest.fixture
def pack():
    root = Path(__file__).resolve().parents[1] / "fixtures"
    metadata = json.loads((root / "tiny-embed-int8.json").read_text())
    return str(root / "tiny-embed-int8.hmsemb"), metadata["sha256"]


def test_reports_expose_utf8_and_token_truncation_without_changing_vectors(pack):
    path, sha = pack
    embedder = hm.Embedder(path, sha256=sha, max_length=8, threads=1, offline=True)
    texts = ["cat " * 5, "cat " * 6, "cat " * 7, "é" * 127 + "🦀"]
    for report_method, vectors_method in [
        (embedder.embed_query_with_report, embedder.embed_query),
        (embedder.embed_passages_with_report, embedder.embed_passages),
    ]:
        report = report_method(texts)
        assert report["embeddings"] == vectors_method(texts)
        assert [item["token_truncated"] for item in report["inputs"][:3]] == [False, False, True]
        assert report["inputs"][3]["byte_truncated"]
        assert report["inputs"][3]["retained_bytes"] == 254


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_limits_reject_oversize_calls_and_allow_following_valid_call(pack, limit):
    path, sha = pack
    embedder = hm.Embedder(path, sha256=sha, threads=1, offline=True, limits={"max_texts": limit})
    for count in [limit - 1, limit, limit + 1]:
        if count > limit:
            with pytest.raises(ValueError, match="max_texts"):
                embedder.embed_query(["cat"] * count)
        else:
            assert len(embedder.embed_query(["cat"] * count)) == count
    assert len(embedder.embed_passages(["dog"])) == 1


def test_limits_reject_unknown_or_zero_settings_before_loading(pack):
    path, sha = pack
    with pytest.raises(ValueError, match="unknown field"):
        hm.Embedder(path, sha256=sha, limits={"max_text": 1})
    with pytest.raises(ValueError, match="must be positive"):
        hm.Embedder(path, sha256=sha, limits={"maxTokens": 0})


def test_failed_index_batch_preserves_existing_text_and_does_not_add_prefix(pack):
    path, sha = pack
    embedder = hm.Embedder(path, sha256=sha, threads=1, offline=True, limits={"max_input_bytes": 8})
    index = hm.SemanticIndex(embedder)
    index.add(["original"], ["cat"])
    with pytest.raises(ValueError, match="max_input_bytes"):
        index.add(["original", "tentative"], ["dog", "cat " * 3])
    hits = index.search("cat", 10)
    assert len(hits) == 1
    assert hits[0][0] == "original"
    assert hits[0][2] == "cat"
