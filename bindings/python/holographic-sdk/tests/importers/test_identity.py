# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest

from holographic_sdk.importers import HmsSink, ImportOptions, ImporterError, run_import
from holographic_sdk.importers.checkpoint import Checkpoint
from holographic_sdk.importers.mapping import Mapper, Target
from holographic_sdk.importers.model import Record, Reject
from holographic_sdk.importers.report import Stats
from holographic_sdk.importers.sources.base import ColumnMap, Source
from holographic_sdk.importers.sources.files import JsonlSource

from .conftest import API_KEY, CrashAfter, vec


def mapped(record, **options):
    opts = ImportOptions(**options)
    opts.validate()
    return Mapper("fixture", opts, Target(input_dim=2), Stats()).map(record)


@pytest.mark.parametrize("parts", [999, 1000, 1001])
def test_record_part_limit(parts):
    record = Record("doc", "word " * parts, [1.0, 0.0])
    if parts > 1000:
        with pytest.raises(Reject, match="too_many_parts"):
            mapped(record, max_text_words=1)
    else:
        assert len(mapped(record, max_text_words=1)) == parts


@pytest.mark.parametrize("length", [255, 256, 257])
def test_typed_ids_bound_long_source_keys_without_ambiguity(length):
    left = mapped(Record("x" * length, "one two", [1.0, 0.0]), id_mapping="v2")
    right = mapped(Record("x" * length + "#part-0", "one two", [1.0, 0.0]), id_mapping="v2")
    assert len(left[0].id.encode()) <= 256
    assert left[0].id != right[0].id


def test_typed_ids_separate_namespaces_literal_parts_and_split_parts():
    records = [Record("b::c", "one", [1, 0], namespace="a"),
               Record("c", "one", [1, 0], namespace="a::b"),
               Record("doc", "one two", [1, 0]),
               Record("doc#part-0", "one", [1, 0])]
    docs = [doc for record in records for doc in mapped(record, id_mapping="v2", max_text_words=1)]
    assert len({doc.id for doc in docs}) == len(docs)
    assert len({doc.metadata["hms_import_identity"] for doc in docs}) == len(docs)
    assert docs == [doc for record in records for doc in mapped(record, id_mapping="v2", max_text_words=1)]


def test_an_older_destination_fails_before_any_import_write():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"components": {"schemas": {"DocIn": {"properties": {"id": {"type": "string"}}}}}})

    sink = HmsSink("http://older.invalid", transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ImporterError, match="atomic import identity"):
            sink.send(mapped(Record("doc", "text", [1, 0])))
        assert [(request.method, request.url.path) for request in requests] == [("GET", "/openapi.json")]
    finally:
        sink.close()


def test_checkpoint_upgrade_requires_an_explicit_restart(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"version": 1, "fingerprint": "same", "position": 3}))
    with pytest.raises(ImporterError, match="unsupported version"):
        Checkpoint.open(str(path), "same", False)
    checkpoint = Checkpoint.open(str(path), "same", True)
    assert checkpoint.position == 0


@pytest.mark.parametrize("position", [-1, 1.5, True, "1", 1 << 63])
def test_checkpoint_positions_are_validated_without_truncation(tmp_path, position):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"version": 2, "fingerprint": "same", "position": position}))
    with pytest.raises(ImporterError, match="invalid position"):
        Checkpoint.open(str(path), "same", False)
    checkpoint = Checkpoint.open(str(path), "same", True)
    assert checkpoint.position == 0


def source_file(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb", "namespace=ns"]))


def execute(source, sink, **options):
    return run_import(source, sink, ImportOptions(verify_sample=0, no_checkpoint=True, **options))


def test_legacy_mapping_collision_fails_before_the_batch_can_overwrite(tmp_path, server):
    source = source_file(tmp_path / "source.jsonl", [
        {"id": "b::c", "ns": "a", "text": "first", "emb": vec(1)},
        {"id": "c", "ns": "a::b", "text": "second", "emb": vec(2)},
    ])
    sink = server.sink(server.tenant("collision"))
    try:
        with pytest.raises(ImporterError, match="collision"):
            execute(source, sink)
        assert list(sink.documents()) == []
        execute(source, sink, id_mapping="v2")
        assert len(list(sink.documents())) == 2
    finally:
        sink.close()


def test_different_sources_cannot_overwrite_the_same_legacy_id(tmp_path, server):
    first = source_file(tmp_path / "first.jsonl", [{"id": "doc", "text": "first", "emb": vec(1)}])
    second = source_file(tmp_path / "second.jsonl", [{"id": "doc", "text": "second", "emb": vec(2)}])
    sink = server.sink(server.tenant("owners"))
    try:
        execute(first, sink)
        with pytest.raises(ImporterError, match="identity conflict"):
            execute(second, sink)
        assert sink.query(vector=vec(1), top_k=1)[0]["text"] == "first"
    finally:
        sink.close()


def test_legacy_adoption_and_id_migration_are_explicit(tmp_path, server):
    source = source_file(tmp_path / "source.jsonl", [{"id": "doc", "text": "updated", "emb": vec(1)}])
    sink = server.sink(server.tenant("adoption"))
    sink.client(None).add_documents([{
        "id": "doc", "text": "legacy", "vector": vec(1),
        "metadata": {"hms_source": "jsonl", "hms_namespace": "", "hms_source_id": "doc"},
    }])
    try:
        with pytest.raises(ImporterError, match="identity conflict"):
            execute(source, sink)
        execute(source, sink, adopt_legacy=True)
        assert list(sink.documents())[0]["metadata"]["hms_import_identity"]
        with pytest.raises(ImporterError, match="layout changed"):
            execute(source, sink, id_mapping="v2")
        report = execute(source, sink, id_mapping="v2", split_policy="replace")
        documents = list(sink.documents())
        assert len(documents) == 1 and documents[0]["id"].startswith("h2:")
        assert report.server["stale_parts_deleted_this_run"] == 1
    finally:
        sink.close()


def test_changed_split_layout_rejects_or_replaces_obsolete_parts(tmp_path, server):
    path = tmp_path / "source.jsonl"
    source = source_file(path, [{"id": "doc", "text": "one two three four", "emb": vec(1)}])
    sink = server.sink(server.tenant("splits"))
    try:
        execute(source, sink, max_text_words=1)
        assert len(list(sink.documents())) == 4
        source = source_file(path, [{"id": "doc", "text": "replacement", "emb": vec(1)}])
        with pytest.raises(ImporterError, match="layout changed"):
            execute(source, sink, max_text_words=1)
        assert len(list(sink.documents())) == 4
        report = execute(source, sink, max_text_words=1, split_policy="replace")
        assert [doc["id"] for doc in sink.documents()] == ["doc"]
        assert report.server["stale_parts_deleted_this_run"] == 4
    finally:
        sink.close()


def test_interrupted_split_cleanup_retries_without_advancing_the_checkpoint(tmp_path, server):
    path = tmp_path / "source.jsonl"
    source = source_file(path, [{"id": "doc", "text": "one two three four", "emb": vec(1)}])
    tenant = server.tenant("cleanup")
    sink = server.sink(tenant)
    execute(source, sink, max_text_words=1)
    sink.close()
    source = source_file(path, [{"id": "doc", "text": "replacement", "emb": vec(1)}])
    sink = server.sink(tenant)
    original = sink._call
    deletes = 0

    def interrupted(fn, *args, **kwargs):
        nonlocal deletes
        if fn.__name__ == "_delete_owned":
            deletes += 1
            if deletes == 2:
                raise ImporterError("simulated cleanup crash")
        return original(fn, *args, **kwargs)

    sink._call = interrupted
    checkpoint = tmp_path / "checkpoint.json"
    options = ImportOptions(verify_sample=0, max_text_words=1, split_policy="replace", checkpoint_path=str(checkpoint))
    try:
        with pytest.raises(ImporterError, match="cleanup crash") as error:
            run_import(source, sink, options, tenant=tenant)
        assert error.value.report.server["stale_parts_deleted_this_run"] == 1
        assert not checkpoint.exists()
    finally:
        sink.close()
    sink = server.sink(tenant)
    try:
        report = run_import(source, sink, options, tenant=tenant)
        assert [doc["id"] for doc in sink.documents()] == ["doc"]
        assert report.server["stale_parts_deleted_this_run"] == 3
        assert json.loads(checkpoint.read_text())["complete"] is True
    finally:
        sink.close()


class MutableSource(Source):
    name = "mutable"

    def __init__(self, seekable):
        self.seekable = seekable
        self.items = [Record(str(i), f"text {i}", vec(i), {"revision": 1}, cursor=i) for i in range(6)]

    def describe(self):
        return {"collection": "test"}

    def records(self, cursor=None):
        yield from self.items[0 if cursor is None else cursor + 1:]


def test_completed_migration_retains_deleted_source_records_and_reports_consistency(tmp_path, server):
    source = MutableSource(True)
    tenant = server.tenant("source-deletion")
    options = ImportOptions(verify_sample=0, checkpoint_path=str(tmp_path / "checkpoint.json"))
    sink = server.sink(tenant)
    try:
        run_import(source, sink, options, tenant=tenant)
        source.items.pop(0)
        source.items[0].text = "changed after completed migration"
        report = run_import(source, sink, options, tenant=tenant)
        assert report.resumed_from is None
        records = {doc["id"]: doc for doc in sink.documents()}
        assert set(records) == {str(i) for i in range(6)}
        hit = sink.query(vector=source.items[0].vector, top_k=1, filter={"hms_source_id": "1"})[0]
        assert hit["text"] == source.items[0].text
        assert report.to_dict()["source_consistency"] == {
            "mode": "observed_stream",
            "snapshot_isolation": "source_managed",
            "resume_validation": "content_verified_prefix_v2",
            "deleted_source_records": "retained",
        }
    finally:
        sink.close()


@pytest.mark.parametrize("seekable", [False, True])
@pytest.mark.parametrize("field", ["text", "metadata", "vector"])
def test_resume_revalidates_content_even_when_the_source_has_cursors(tmp_path, server, seekable, field):
    source = MutableSource(seekable)
    tenant = server.tenant("prefix")
    options = ImportOptions(batch_size=2, verify_sample=0, checkpoint_path=str(tmp_path / "checkpoint.json"))
    crashing = CrashAfter(server.url, api_key=API_KEY, tenant=tenant, crash_after=1)
    try:
        with pytest.raises(ImporterError, match="simulated crash"):
            run_import(source, crashing, options, tenant=tenant)
    finally:
        crashing.close()
    setattr(source.items[0], field, {"text": "changed text", "metadata": {"revision": 2}, "vector": vec(100)}[field])
    sink = server.sink(tenant)
    try:
        report = run_import(source, sink, options, tenant=tenant)
        assert report.server["resume_prefix_mismatch"] is True
        assert report.resumed_from is None
        hit = sink.query(vector=source.items[0].vector, top_k=1, filter={"hms_source_id": "0"})[0]
        assert hit["text"] == source.items[0].text
        assert hit["metadata"]["revision"] == source.items[0].metadata["revision"]
        assert report.stats.counts["docs_acked"] == 6
    finally:
        sink.close()


def test_changed_prefix_restarts_its_checkpoint_position_before_another_crash(tmp_path, server):
    source = MutableSource(True)
    tenant = server.tenant("restart-position")
    checkpoint = tmp_path / "checkpoint.json"
    options = ImportOptions(batch_size=2, verify_sample=0, checkpoint_path=str(checkpoint))
    for attempt in range(2):
        crashing = CrashAfter(server.url, api_key=API_KEY, tenant=tenant, crash_after=1)
        try:
            with pytest.raises(ImporterError, match="simulated crash"):
                run_import(source, crashing, options, tenant=tenant)
        finally:
            crashing.close()
        assert json.loads(checkpoint.read_text())["position"] == 2
        source.items[0].text = "changed prefix"
    sink = server.sink(tenant)
    try:
        report = run_import(source, sink, options, tenant=tenant)
        assert report.resumed_from == 2
        assert report.stats.counts["docs_acked"] == 6
        assert json.loads(checkpoint.read_text())["position"] == 6
    finally:
        sink.close()
