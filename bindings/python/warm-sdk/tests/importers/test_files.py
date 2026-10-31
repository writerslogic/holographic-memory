# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import json

from writerslogic_warm_sdk.importers.sources.base import ColumnMap
from writerslogic_warm_sdk.importers.sources.files import JsonlSource

from .conftest import assert_faithful, vec


def test_jsonl_roundtrip(tmp_path, run):
    path = tmp_path / "docs.jsonl"
    with open(path, "w") as fh:
        for i in range(40):
            fh.write(json.dumps({"id": f"d{i}", "text": f"doc number {i}", "emb": vec(i), "tag": i % 3,
                                 "ts": 1_700_000_000 + i}) + "\n")
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb", "created_at=ts"]))
    report, _ = run(src, batch_size=7)
    assert_faithful(report, 40)
