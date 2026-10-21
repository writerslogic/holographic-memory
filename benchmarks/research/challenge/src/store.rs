// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Storage: the root crate's memory API underneath, a family arena on top.
//!
//! # Mechanism
//!
//! Every assertion is remembered through `Memory::remember_many` as an ordinary fact whose
//! `subject` is the family key (`entity/attribute`), whose `valid_from`/`valid_to` carry the
//! validity window, whose `supersedes` carries the caller's explicit markers (a retraction
//! supersedes its target) and whose caller metadata holds the structured fields every
//! retriever in this crate reads (layout below). The store therefore keeps journaling, replay
//! and `ForgetScope::Subject` erasure of the memory API; nothing is persisted outside it.
//!
//! The challenge contract needs lookups the memory API does not expose: scope cover and
//! compatibility, latest-per-scope, retraction targets and concurrent ties. Its metadata
//! filter is exact top-level equality (`src/core/memory_api/types.rs:98-99`), its subject
//! index is a flat member list without scope (`src/core/memory_api/index.rs:41`), and its
//! documents are keyed by id with no family-contiguous placement
//! (`docs/memory-api/PERSISTED-FORMAT.md`: "A fact is an ordinary document"). So the
//! [`Arena`] is a derived, in-process index built from `Memory::get` records: one
//! contiguous [`Family`] block per `(entity, attribute)`. It is never persisted; it is
//! rebuilt from the memory API after every ingest.
//!
//! # Metadata layout (caller metadata of every fact)
//!
//! ```json
//! {"entity": "acct017", "attribute": "payment_terms", "scope": {"region": "eu"},
//!  "kind": "claim" | "retraction", "seq": 8, "value": "net 15",
//!  "from": 1700000000000 | null, "to": null, "target": "acct017/payment_terms#09" | null}
//! ```
//!
//! # Erase unit
//!
//! [`ChallengeStore::wipe_family`] drops the family's arena block (one `Vec` drop) and then
//! forgets every fact of the subject in one memory API call, so a family is removed from the
//! index and the store with no per-record bookkeeping.
//!
//! # Invariants
//!
//! - A fact id is `entity/attribute#seq` and is unique store-wide.
//! - The arena mirrors the memory API: every id it holds resolves through `Memory::get`
//!   until the family is wiped, after which neither holds it.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use holographic_memory::core::memory_api::{
    FactRecord, ForgetOptions, ForgetScope, ManualClock, Memory, RememberInput,
};
use holographic_memory::HmsCore;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::contract::{Assertion, AssertionKind, Family, FamilyKey, Millis, Scope};

/// Hypervector width of the engine underneath; the memory API does not depend on it.
pub const DEFAULT_DIMENSIONS: u32 = 1024;

/// The memory API caps one `remember_many` call (`src/core/memory_api/write.rs:16`).
const MAX_BATCH: usize = 4096;

/// One assertion with the family it belongs to, as the generator produces it.
#[derive(Clone, Debug)]
pub struct RecordInput {
    pub key: FamilyKey,
    pub assertion: Assertion,
}

/// The structured fields of a fact's caller metadata, as every retriever reads them.
#[derive(Clone, Debug, PartialEq)]
pub struct MetaView {
    pub key: FamilyKey,
    pub kind: AssertionKind,
    pub scope: Scope,
    pub seq: u32,
    pub value: String,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub target: Option<String>,
}

impl MetaView {
    pub fn valid_at(&self, at: Millis) -> bool {
        self.valid_from.is_none_or(|from| from <= at) && self.valid_to.is_none_or(|to| at < to)
    }
}

fn millis_value(value: Option<Millis>) -> Value {
    value.map_or(Value::Null, Value::from)
}

/// The memory API input for one assertion. `ingest_time` is the event time of the
/// observation; the clock supplies the ingest time at write.
pub fn remember_input(key: &FamilyKey, assertion: &Assertion, event_time: Millis) -> RememberInput {
    let mut metadata = Map::new();
    metadata.insert("entity".into(), Value::String(key.entity.clone()));
    metadata.insert("attribute".into(), Value::String(key.attribute.clone()));
    metadata.insert("scope".into(), assertion.scope.to_json());
    metadata.insert(
        "kind".into(),
        Value::String(
            match assertion.kind {
                AssertionKind::Claim => "claim",
                AssertionKind::Retraction => "retraction",
            }
            .into(),
        ),
    );
    metadata.insert("seq".into(), Value::from(assertion.seq));
    metadata.insert("value".into(), Value::String(assertion.value.clone()));
    metadata.insert("from".into(), millis_value(assertion.valid_from));
    metadata.insert("to".into(), millis_value(assertion.valid_to));
    metadata.insert(
        "target".into(),
        assertion
            .target
            .clone()
            .map_or(Value::Null, Value::String),
    );
    let mut supersedes = assertion.supersedes.clone();
    if let Some(target) = &assertion.target {
        if !supersedes.contains(target) {
            supersedes.push(target.clone());
        }
    }
    RememberInput {
        id: Some(assertion.id.clone()),
        text: assertion.text.clone(),
        event_time: Some(event_time),
        valid_from: assertion.valid_from,
        valid_to: assertion.valid_to,
        subject: Some(key.subject()),
        supersedes: (!supersedes.is_empty()).then_some(supersedes),
        entities: Some(vec![key.entity.clone()]),
        metadata: Some(Value::Object(metadata)),
        embedding: None,
        source_uri: None,
    }
}

fn string_field(metadata: &Map<String, Value>, name: &str) -> Result<String> {
    metadata
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("metadata field {name:?} is missing or not a string"))
}

fn millis_field(metadata: &Map<String, Value>, name: &str) -> Result<Option<Millis>> {
    match metadata.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| anyhow!("metadata field {name:?} is not an integer")),
    }
}

/// Reads the structured fields back from caller metadata (the `_mem` key, if present, is
/// ignored).
pub fn parse_meta(metadata: &Value) -> Result<MetaView> {
    let map = metadata
        .as_object()
        .ok_or_else(|| anyhow!("metadata is not an object"))?;
    let kind = match string_field(map, "kind")?.as_str() {
        "claim" => AssertionKind::Claim,
        "retraction" => AssertionKind::Retraction,
        other => return Err(anyhow!("unknown assertion kind {other:?}")),
    };
    let scope = map
        .get("scope")
        .and_then(Scope::from_json)
        .ok_or_else(|| anyhow!("metadata field \"scope\" is not a scope"))?;
    let seq = map
        .get("seq")
        .and_then(Value::as_u64)
        .and_then(|seq| u32::try_from(seq).ok())
        .ok_or_else(|| anyhow!("metadata field \"seq\" is not a u32"))?;
    let target = match map.get("target") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("metadata field \"target\" is not a string"))?,
        ),
    };
    Ok(MetaView {
        key: FamilyKey::new(string_field(map, "entity")?, string_field(map, "attribute")?),
        kind,
        scope,
        seq,
        value: string_field(map, "value")?,
        valid_from: millis_field(map, "from")?,
        valid_to: millis_field(map, "to")?,
        target,
    })
}

/// An arena assertion from a memory API record. The validity window is the memory API's
/// (`FactRecord::valid_from`/`valid_to`), the supersession list is the record's own.
pub fn assertion_from_record(record: &FactRecord) -> Result<(FamilyKey, Assertion)> {
    let meta = parse_meta(&record.metadata)
        .with_context(|| format!("fact {:?} carries no challenge metadata", record.id))?;
    let text = record
        .text
        .clone()
        .ok_or_else(|| anyhow!("fact {:?} has no stored text", record.id))?;
    Ok((
        meta.key,
        Assertion {
            id: record.id.clone(),
            seq: meta.seq,
            kind: meta.kind,
            scope: meta.scope,
            value: meta.value,
            text,
            valid_from: record.valid_from,
            valid_to: record.valid_to,
            supersedes: record.supersedes.clone(),
            target: meta.target,
        },
    ))
}

/// Family blocks, one per key. A wiped family keeps its slot (tombstoned) so block indices
/// stay stable; its block is dropped.
#[derive(Default)]
pub struct Arena {
    blocks: Vec<Option<Family>>,
    index: HashMap<FamilyKey, usize>,
}

impl Arena {
    /// Builds the arena from assertions grouped by family. Each group becomes one block.
    pub fn from_groups(groups: BTreeMap<FamilyKey, Vec<Assertion>>) -> Self {
        let mut arena = Self::default();
        for (key, assertions) in groups {
            let slot = arena.blocks.len();
            arena.blocks.push(Some(Family::new(key.clone(), assertions)));
            arena.index.insert(key, slot);
        }
        arena
    }

    pub fn family(&self, key: &FamilyKey) -> Option<&Family> {
        self.index
            .get(key)
            .and_then(|&slot| self.blocks[slot].as_ref())
    }

    /// Drops one family's block. Returns how many assertions went with it, or none when the
    /// family is unknown or already wiped.
    pub fn wipe(&mut self, key: &FamilyKey) -> Option<usize> {
        let slot = *self.index.get(key)?;
        self.blocks[slot].take().map(|family| family.len())
    }

    pub fn families(&self) -> usize {
        self.blocks.iter().filter(|block| block.is_some()).count()
    }

    pub fn assertions(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(|family| family.len())
            .sum()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct IngestStats {
    pub batches: usize,
    pub facts_created: usize,
    pub facts_idempotent: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ArenaStats {
    pub families: usize,
    pub assertions: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct WipeReport {
    pub key: FamilyKey,
    pub arena_assertions_dropped: usize,
    pub facts_forgotten: usize,
    pub elapsed_us: f64,
}

/// The memory API over an engine in a private directory, plus the arena.
pub struct ChallengeStore {
    _tempdir: Option<tempfile::TempDir>,
    path: PathBuf,
    memory: Memory,
    clock: Arc<ManualClock>,
    arena: Arena,
}

impl ChallengeStore {
    /// Opens a fresh store at `dir`, or in a temporary directory removed on drop.
    pub fn open(dir: Option<PathBuf>, dimensions: u32) -> Result<Self> {
        let (tempdir, path) = match dir {
            Some(path) => {
                std::fs::create_dir_all(&path)
                    .with_context(|| format!("cannot create store directory {}", path.display()))?;
                (None, path)
            }
            None => {
                let tempdir = tempfile::tempdir().context("cannot create a temporary store directory")?;
                let path = tempdir.path().to_path_buf();
                (Some(tempdir), path)
            }
        };
        let core = Arc::new(
            HmsCore::new(dimensions, Some(path.display().to_string()), None)
                .with_context(|| format!("cannot open the engine at {}", path.display()))?,
        );
        let clock = Arc::new(ManualClock::new(0));
        let memory = Memory::new(core).with_clock(clock.clone());
        Ok(Self {
            _tempdir: tempdir,
            path,
            memory,
            clock,
            arena: Arena::default(),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    pub fn core(&self) -> &Arc<HmsCore> {
        self.memory.core()
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Remembers every batch in order, the clock set to the batch's ingest time. Inputs in
    /// one batch may supersede facts of earlier batches or earlier inputs of the same batch.
    pub fn ingest(&mut self, batches: &[(Millis, Vec<RecordInput>)]) -> Result<IngestStats> {
        let started = Instant::now();
        let mut stats = IngestStats::default();
        for (ingest_time, records) in batches {
            self.clock.set(*ingest_time);
            for chunk in records.chunks(MAX_BATCH) {
                let inputs: Vec<RememberInput> = chunk
                    .iter()
                    .map(|record| remember_input(&record.key, &record.assertion, *ingest_time))
                    .collect();
                let outcomes = self
                    .memory
                    .remember_many(inputs)
                    .with_context(|| format!("remember_many failed at ingest time {ingest_time}"))?;
                for outcome in &outcomes {
                    if outcome.created {
                        stats.facts_created += 1;
                    } else {
                        stats.facts_idempotent += 1;
                    }
                }
            }
            stats.batches += 1;
        }
        stats.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(stats)
    }

    /// Rebuilds the arena from the memory API: every id listed is fetched with `Memory::get`
    /// and grouped by the family its metadata names.
    pub fn build_arena(&mut self, ids_by_family: &[(FamilyKey, Vec<String>)]) -> Result<ArenaStats> {
        let started = Instant::now();
        let mut groups: BTreeMap<FamilyKey, Vec<Assertion>> = BTreeMap::new();
        for (expected, ids) in ids_by_family {
            for id in ids {
                let record = self
                    .memory
                    .get(id)
                    .with_context(|| format!("get {id:?} failed"))?
                    .ok_or_else(|| anyhow!("fact {id:?} was ingested but is not stored"))?;
                let (key, assertion) = assertion_from_record(&record)?;
                if &key != expected {
                    return Err(anyhow!(
                        "fact {id:?} belongs to family {key} but was listed under {expected}"
                    ));
                }
                groups.entry(key).or_default().push(assertion);
            }
        }
        self.arena = Arena::from_groups(groups);
        Ok(ArenaStats {
            families: self.arena.families(),
            assertions: self.arena.assertions(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// The erase unit: drops the family's block and forgets its subject in the memory API.
    pub fn wipe_family(&mut self, key: &FamilyKey) -> Result<WipeReport> {
        let started = Instant::now();
        let arena_assertions_dropped = self.arena.wipe(key).unwrap_or(0);
        let report = self
            .memory
            .forget(
                &ForgetScope::Subject {
                    subject: key.subject(),
                },
                &ForgetOptions::default(),
            )
            .with_context(|| format!("forget subject {key} failed"))?;
        Ok(WipeReport {
            key: key.clone(),
            arena_assertions_dropped,
            facts_forgotten: report.forgotten.len(),
            elapsed_us: started.elapsed().as_secs_f64() * 1e6,
        })
    }

    /// Whether any listed id still resolves in the memory API.
    pub fn any_stored(&self, ids: &[String]) -> Result<bool> {
        for id in ids {
            if self.memory.get(id)?.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
