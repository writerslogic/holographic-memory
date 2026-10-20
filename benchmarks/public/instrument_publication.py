# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact, isolated timing edits; never modifies the working engine."""

from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"instrumentation anchor count changed: {old[:80]}")
    return text.replace(old, new)


def instrument(source: Path, arm: str, harness: Path):
    module = source / "src/core/engine/mod.rs"
    module.write_text(replace_once(module.read_text(), "mod compaction;", "mod compaction;\npub mod diagnostic;"))
    (source / "src/core/engine/diagnostic.rs").write_bytes((harness / "publication_trace.rs").read_bytes())
    example = source / "examples/publication_serving.rs"
    example.write_bytes((harness / "publication_serving.rs").read_bytes())
    mutation = source / "src/core/engine/mutation.rs"
    text = mutation.read_text()
    if arm == "legacy-reader-lock":
        old = """        let _transaction = self.mutation_gate.write();
        let shards = self.shards.write();
        self.arena_write(&payload)?;
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())"""
        new = """        let wait = super::diagnostic::now();
        let _transaction = self.mutation_gate.write();
        let acquired = super::diagnostic::now();
        let shards = self.shards.write();
        let append_start = super::diagnostic::now();
        self.arena_write(&payload)?;
        let append_end = super::diagnostic::now();
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        let publication_end = super::diagnostic::now();
        drop(shards);
        drop(_transaction);
        let released = super::diagnostic::now();
        super::diagnostic::record(super::diagnostic::Row {
            kind: "commit", count: mutations.len(), wait_ns: wait, acquired_ns: acquired,
            append_start_ns: append_start, append_end_ns: append_end,
            exclusion_start_ns: acquired, publication_end_ns: publication_end, released_ns: released,
        });
        Ok(())"""
    else:
        old = """        let transaction = self.mutation_gate.begin_write();
        append(&payload)?;
        let _transaction = transaction.publish();
        let shards = self.shards.write();
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())"""
        new = """        let wait = super::diagnostic::now();
        let transaction = self.mutation_gate.begin_write();
        let acquired = super::diagnostic::now();
        let append_start = super::diagnostic::now();
        append(&payload)?;
        let append_end = super::diagnostic::now();
        let _transaction = transaction.publish();
        let exclusion_start = super::diagnostic::now();
        let shards = self.shards.write();
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        let publication_end = super::diagnostic::now();
        drop(shards);
        drop(_transaction);
        let released = super::diagnostic::now();
        super::diagnostic::record(super::diagnostic::Row {
            kind: "commit", count: mutations.len(), wait_ns: wait, acquired_ns: acquired,
            append_start_ns: append_start, append_end_ns: append_end,
            exclusion_start_ns: exclusion_start, publication_end_ns: publication_end, released_ns: released,
        });
        Ok(())"""
    mutation.write_text(replace_once(text, old, new))
    compaction = source / "src/core/engine/compaction.rs"
    text = compaction.read_text()
    old = """        before_publication();
        let _transaction = self.mutation_gate.maintenance();
        let end = self.compaction_end(&generation)?;"""
    new = """        before_publication();
        let wait = super::diagnostic::now();
        let _transaction = self.mutation_gate.maintenance();
        let acquired = super::diagnostic::now();
        let end = self.compaction_end(&generation)?;"""
    text = replace_once(text, old, new)
    old = """        self.persist_indices(&self.shards.read())?;
        if let Some(audit) = &self.audit {"""
    new = """        self.persist_indices(&self.shards.read())?;
        let publication_end = super::diagnostic::now();
        if let Some(audit) = &self.audit {"""
    text = replace_once(text, old, new)
    old = """        Ok(())
    }

    fn compaction_snapshot"""
    new = """        drop(_transaction);
        let released = super::diagnostic::now();
        super::diagnostic::record(super::diagnostic::Row {
            kind: "compact-final", count: 0, wait_ns: wait, acquired_ns: acquired,
            append_start_ns: 0, append_end_ns: 0, exclusion_start_ns: 0,
            publication_end_ns: publication_end, released_ns: released,
        });
        Ok(())
    }

    fn compaction_snapshot"""
    compaction.write_text(replace_once(text, old, new))
