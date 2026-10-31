# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from writerslogic_warm_sdk.importers.common import scrub, uri_secret
from writerslogic_warm_sdk.importers.sources.llamaindex import LlamaIndexSource


def test_malformed_connection_uri_fails_without_disclosing_the_password():
    password = "synthetic-malformed-uri-password"
    with pytest.raises(ValueError, match="malformed connection URI") as error:
        uri_secret(f"postgres://user:{password}@[invalid/database")
    assert password not in str(error.value)


def test_valid_uri_registers_the_password_for_error_redaction():
    uri = "postgres://user:synthetic-uri-password@localhost/database"
    assert uri_secret(uri) == uri
    assert "synthetic-uri-password" not in scrub("failed: " + uri)


@pytest.mark.parametrize("content", ["invalid JSON", "null", "[]", "17", '{"text": "retained text"}'])
def test_vector_only_llamaindex_metadata_preserves_the_record(tmp_path, content):
    store = {
        "embedding_dict": {"document": [0.1, 0.2]},
        "metadata_dict": {"document": {"_node_content": content, "source": "fixture"}},
    }
    (tmp_path / "vectors.json").write_text(json.dumps(store), encoding="utf-8")
    source = LlamaIndexSource(str(tmp_path), "vectors.json", "documents.json", "namespace", "auto")
    records = list(source.records())
    assert len(records) == 1
    record = records[0]
    assert record.id == "document" and record.vector == [0.1, 0.2]
    assert record.metadata == {"source": "fixture"}
    assert record.text == ("retained text" if content.startswith("{") else None)
