# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lexical comparators run inside their pinned environment. Same phases and output shape as
examples/competitive_text.rs (build, serve, ops) over the {documents, queries} input JSON.

Queries are reduced to the lowercase alphanumeric word set (as the existing SQLite control in
benchmarks/public/document_competition.py does) and combined with OR; each system then applies
its own documented analyzer (stemmer, stop words) and default BM25 parameters.
"""

import argparse
import itertools
import json
import sqlite3
import sys
import time
from pathlib import Path

T_IMPORT = time.time_ns()

K = 100


class Unsupported(Exception):
    pass


def words(text):
    return sorted({"".join(chars) for kind, chars in itertools.groupby(text.lower(), str.isalnum) if kind})


def peak_rss():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


class SqliteFts5:
    def __init__(self, store):
        self.store = Path(store)

    def build(self, documents):
        self.store.mkdir(parents=True)
        db = sqlite3.connect(self.store / "fts.db")
        try:
            db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, content='', contentless_delete=1, tokenize='porter unicode61')")
        except sqlite3.OperationalError:
            # SQLite older than 3.43 cannot delete from a contentless table; fall back to a stored-content
            # table (larger on disk; the SQLite version is recorded in the results).
            db.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='porter unicode61')")
        db.execute("CREATE TABLE identities (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL)")
        with db:
            for i, d in enumerate(documents, 1):
                db.execute("INSERT INTO docs(rowid, body) VALUES (?, ?)", (i, d["text"]))
                db.execute("INSERT INTO identities(rowid, id) VALUES (?, ?)", (i, d["id"]))
        db.execute("INSERT INTO docs(docs) VALUES ('optimize')")
        db.commit()
        db.close()

    def open(self):
        self.db = sqlite3.connect(self.store / "fts.db")
        # macOS fsync does not flush the drive cache; HMS's sync_all (and tantivy's, Lucene's) do.
        # Matching that guarantee keeps the durable replace/delete rows comparable.
        self.db.execute("PRAGMA fullfsync = ON")
        self.db.execute("PRAGMA checkpoint_fullfsync = ON")

    def prepare(self, text):
        return " OR ".join('"' + w + '"' for w in words(text))

    def search(self, expression):
        if not expression:
            return []
        rows = self.db.execute(
            "SELECT identities.id, bm25(docs) FROM docs JOIN identities ON docs.rowid=identities.rowid "
            "WHERE docs MATCH ? ORDER BY 2, identities.id LIMIT ?", (expression, K)).fetchall()
        return [(i, -s) for i, s in rows]

    def replace(self, document):
        row = self.db.execute("SELECT rowid FROM identities WHERE id = ?", (document["id"],)).fetchone()[0]
        with self.db:
            self.db.execute("DELETE FROM docs WHERE rowid = ?", (row,))
            self.db.execute("INSERT INTO docs(rowid, body) VALUES (?, ?)", (row, document["text"] + " benchmarkrevision"))

    def delete(self, document):
        row = self.db.execute("SELECT rowid FROM identities WHERE id = ?", (document["id"],)).fetchone()[0]
        with self.db:
            self.db.execute("DELETE FROM docs WHERE rowid = ?", (row,))
            self.db.execute("DELETE FROM identities WHERE rowid = ?", (row,))


class Tantivy:
    def __init__(self, store):
        self.store = Path(store)

    def _schema(self):
        import tantivy
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("id", stored=True, tokenizer_name="raw")
        sb.add_text_field("body", stored=False, tokenizer_name="en_stem")
        return sb.build()

    def build(self, documents):
        import tantivy
        self.store.mkdir(parents=True)
        index = tantivy.Index(self._schema(), path=str(self.store))
        writer = index.writer(heap_size=200_000_000, num_threads=1)
        for d in documents:
            writer.add_document(tantivy.Document(id=d["id"], body=d["text"]))
        writer.commit()
        writer.wait_merging_threads()

    def open(self):
        import tantivy
        self.index = tantivy.Index(self._schema(), path=str(self.store))
        self.index.reload()
        self.searcher = self.index.searcher()

    def prepare(self, text):
        return " ".join(words(text))

    def search(self, text):
        query, _ = self.index.parse_query_lenient(text, ["body"])
        hits = self.searcher.search(query, K).hits
        return [(self.searcher.doc(addr)["id"][0], score) for score, addr in hits]

    def _writer(self):
        return self.index.writer(heap_size=50_000_000, num_threads=1)

    def replace(self, document):
        import tantivy
        writer = self._writer()
        writer.delete_documents("id", document["id"])
        writer.add_document(tantivy.Document(id=document["id"], body=document["text"] + " benchmarkrevision"))
        writer.commit()

    def delete(self, document):
        writer = self._writer()
        writer.delete_documents("id", document["id"])
        writer.commit()


class Whoosh:
    def __init__(self, store):
        self.store = Path(store)

    def _schema(self):
        from whoosh.analysis import StemmingAnalyzer
        from whoosh.fields import ID, TEXT, Schema
        return Schema(id=ID(stored=True, unique=True), body=TEXT(analyzer=StemmingAnalyzer()))

    def build(self, documents):
        from whoosh import index
        self.store.mkdir(parents=True)
        ix = index.create_in(str(self.store), self._schema())
        writer = ix.writer(procs=1, limitmb=512)
        for d in documents:
            writer.add_document(id=d["id"], body=d["text"])
        writer.commit()

    def open(self):
        from whoosh import index
        from whoosh.qparser import OrGroup, QueryParser
        self.ix = index.open_dir(str(self.store))
        self.searcher = self.ix.searcher()
        self.parser = QueryParser("body", self.ix.schema, group=OrGroup)

    def prepare(self, text):
        return " ".join(words(text))

    def search(self, text):
        if not text:
            return []
        hits = self.searcher.search(self.parser.parse(text), limit=K)
        return [(h["id"], h.score) for h in hits]

    def replace(self, document):
        writer = self.ix.writer()
        writer.update_document(id=document["id"], body=document["text"] + " benchmarkrevision")
        writer.commit()

    def delete(self, document):
        writer = self.ix.writer()
        writer.delete_by_term("id", document["id"])
        writer.commit()


class Bm25s:
    def __init__(self, store):
        self.store = Path(store)

    def _tokens(self, texts):
        import bm25s
        import Stemmer
        return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)

    def build(self, documents):
        import bm25s
        self.store.mkdir(parents=True)
        retriever = bm25s.BM25()
        retriever.index(self._tokens([d["text"] for d in documents]), show_progress=False)
        retriever.save(str(self.store))
        (self.store / "ids.json").write_text(json.dumps([d["id"] for d in documents]))

    def open(self):
        import bm25s
        import Stemmer
        self.retriever = bm25s.BM25.load(str(self.store), load_corpus=False)
        self.ids = json.loads((self.store / "ids.json").read_text())
        self.stemmer = Stemmer.Stemmer("english")

    def prepare(self, text):
        return " ".join(words(text))

    def search(self, text):
        import bm25s
        if not text:
            return []
        tokens = bm25s.tokenize([text], stopwords="en", stemmer=self.stemmer, show_progress=False)
        docs, scores = self.retriever.retrieve(tokens, k=min(K, len(self.ids)), show_progress=False, n_threads=1)
        return [(self.ids[int(i)], float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]

    def replace(self, document):
        raise Unsupported("immutable index; any change is a full rebuild")

    def delete(self, document):
        raise Unsupported("immutable index; any change is a full rebuild")


SYSTEMS = {"sqlite-fts5": SqliteFts5, "tantivy": Tantivy, "whoosh": Whoosh, "bm25s": Bm25s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "serve", "ops"])
    p.add_argument("--system", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--ops", type=int, default=100)
    p.add_argument("--t0-ns", type=int, default=T_IMPORT)
    a = p.parse_args()
    data = json.loads(Path(a.input).read_text())
    system = SYSTEMS[a.system](a.store)
    if a.mode == "build":
        start = time.perf_counter()
        system.build(data["documents"])
        report = {"documents": len(data["documents"]), "ingest_s": time.perf_counter() - start}
    elif a.mode == "serve":
        start = time.perf_counter()
        system.open()
        open_s = time.perf_counter() - start
        queries = data["queries"]
        prepared = [system.prepare(q["text"]) for q in queries]
        system.search(prepared[0])
        cold_start_s = (time.time_ns() - a.t0_ns) / 1e9
        for text in prepared[:16]:
            system.search(text)
        rows, passes = [], []
        for n in range(a.passes):
            ns = []
            for q, text in zip(queries, prepared):
                t = time.perf_counter_ns()
                found = system.search(text)
                ns.append(time.perf_counter_ns() - t)
                if n == 0:
                    rows.append({"id": q["id"], "results": [{"documentId": i, "lexicalScore": s} for i, s in found]})
            passes.append(ns)
        report = {"open_s": open_s, "cold_start_s": cold_start_s, "rows": rows, "passes_ns": passes}
    else:
        system.open()
        docs = data["documents"]
        report = {"count": a.ops}
        for name, subset in (("replace", docs[: a.ops]), ("delete", docs[a.ops: 2 * a.ops])):
            try:
                t = time.perf_counter()
                for d in subset:
                    getattr(system, name)(d)
                report[name] = {"supported": True, "per_op_ms_durable": (time.perf_counter() - t) / a.ops * 1e3}
            except Unsupported as e:
                report[name] = {"supported": False, "reason": str(e)}
    report["peak_rss_bytes_self"] = peak_rss()
    Path(a.out).write_text(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
