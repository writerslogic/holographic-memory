// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use writerslogic_warm::core::documents::{EncodeMode, IngestOptions};
use writerslogic_warm::core::memory_api::{ForgetOptions, ForgetScope};
use writerslogic_warm::{DocumentInput, DocumentRecord, WarmCore, Memory, SearchOptions};
use serde::{Deserialize, Serialize};

use crate::contract::{
    validate_claims, validate_id, validate_sources, Claim, Selection, Source, MAX_CLAIMS,
    MAX_SOURCES, MAX_TOTAL_BYTES,
};

pub const MAX_METADATA_BYTES: usize = 64 * 1024;
pub const MAX_CONTROL_BYTES: usize = 64 * 1024;
pub const MAX_REVOKED_VERSIONS: usize = 512;
pub const MAX_PENDING_DELETIONS: usize = 256;
pub const STORE_DIMENSIONS: u32 = 256;
const PAGE_SIZE: usize = 16;
const SCHEMA_VERSION: u32 = 1;
const CONTROL_VERSION: &str = "evidence-control:v1";
const CONTROL_TEXT: &str = "Evidence deletion replay fence.";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    schema_version: u32,
    namespace: String,
    source_id: String,
    source_version: String,
    observed_at: i64,
    claims: Vec<Claim>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionPin {
    source_id: String,
    source_version: String,
}

impl VersionPin {
    fn of(source: &Source) -> Self {
        Self {
            source_id: source.id.clone(),
            source_version: source.version.clone(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Control {
    schema_version: u32,
    namespace: String,
    revoked: Vec<VersionPin>,
    pending: Vec<VersionPin>,
}

pub fn validate_bundle(namespace: &str, sources: &[Source], claims: &[Claim]) -> Result<()> {
    validate_id(namespace)?;
    validate_snapshot(sources, claims)?;
    let mut total_metadata = 0usize;
    for source in sources {
        let metadata = source_metadata(namespace, source, claims);
        let size = bounded_size(&metadata, MAX_METADATA_BYTES)?;
        add_size(&mut total_metadata, size, "stored metadata exceeds 8 MiB")?;
    }
    Ok(())
}

/// Namespace selection is a query convention; it is not an authorization boundary.
pub struct EvidenceStore {
    core: Arc<WarmCore>,
    memory: Memory,
    namespace: String,
    prefix: String,
}

impl EvidenceStore {
    pub fn open(path: impl AsRef<Path>, namespace: &str) -> Result<Self> {
        validate_id(namespace)?;
        let path = path
            .as_ref()
            .to_str()
            .context("evidence store path is not UTF-8")?;
        let core = Arc::new(WarmCore::new(STORE_DIMENSIONS, Some(path.into()), None)?);
        let store = Self {
            memory: Memory::new(Arc::clone(&core)),
            core,
            namespace: namespace.into(),
            prefix: format!("evidence:v1:{}:", hex(namespace)),
        };
        store.recover_pending()?;
        store.snapshot()?;
        Ok(store)
    }

    /// Replaces each supplied source's complete claim set. The first evidence span owns a claim.
    pub fn put(&mut self, sources: &[Source], claims: &[Claim]) -> Result<()> {
        validate_sources(sources)?;
        ensure!(claims.len() <= MAX_CLAIMS, "too many claims");
        let replacing: BTreeSet<_> = sources.iter().map(|source| source.id.as_str()).collect();
        for claim in claims {
            ensure!(
                replacing.contains(owner(claim)?),
                "claim owner must be a supplied source"
            );
        }
        self.recover_pending()?;
        let control = self.read_control()?;
        let revoked: BTreeSet<_> = control.revoked.iter().collect();
        ensure!(
            sources
                .iter()
                .all(|source| !revoked.contains(&VersionPin::of(source))),
            "source version was revoked by an accepted deletion"
        );
        let (stored_sources, stored_claims) = self.snapshot()?;
        let mut resulting_sources: BTreeMap<_, _> = stored_sources
            .into_iter()
            .map(|source| (source.id.clone(), source))
            .collect();
        for source in sources {
            if let Some(previous) = resulting_sources.get(&source.id) {
                ensure!(
                    previous.version != source.version || previous == source,
                    "source content changed without a new version"
                );
            }
            resulting_sources.insert(source.id.clone(), source.clone());
        }
        let resulting_sources: Vec<_> = resulting_sources.into_values().collect();
        let mut resulting_claims = Vec::new();
        for claim in stored_claims {
            if !replacing.contains(owner(&claim)?) {
                resulting_claims.push(claim);
            }
        }
        resulting_claims.extend_from_slice(claims);
        validate_snapshot(&resulting_sources, &resulting_claims)?;

        let mut total_metadata = 0usize;
        let mut documents = Vec::with_capacity(sources.len());
        for source in &resulting_sources {
            let metadata = source_metadata(&self.namespace, source, &resulting_claims);
            let size = bounded_size(&metadata, MAX_METADATA_BYTES)?;
            add_size(&mut total_metadata, size, "stored metadata exceeds 8 MiB")?;
            if replacing.contains(source.id.as_str()) {
                documents.push(DocumentInput {
                    id: self.document_id(&source.id),
                    text: source.text.clone(),
                    source_uri: None,
                    version: Some(document_version(&source.version)),
                    metadata: Some(serde_json::to_value(metadata)?),
                    chunk_words: None,
                    overlap_words: None,
                    store_text: Some(true),
                    embeddings: None,
                    title_weight: None,
                });
            }
        }
        if !documents.is_empty() {
            self.core.memorize_documents_with(
                documents,
                &IngestOptions {
                    encode: EncodeMode::Lexical,
                    positions: true,
                    whole_document: true,
                    ..Default::default()
                },
            )?;
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<(Vec<Source>, Vec<Claim>)> {
        let control = self.read_control()?;
        ensure!(
            control.pending.is_empty(),
            "pending deletion requires recovery"
        );
        let snapshot = self.source_snapshot()?;
        validate_live_sources(&snapshot.0, &control)?;
        Ok(snapshot)
    }

    fn source_snapshot(&self) -> Result<(Vec<Source>, Vec<Claim>)> {
        let mut sources = Vec::new();
        let mut claims = Vec::new();
        let mut from = self.prefix.clone();
        let mut text_bytes = 0usize;
        let mut metadata_bytes = 0usize;
        loop {
            let records = self.core.list_documents(Some(&from), PAGE_SIZE);
            let mut last = None;
            let mut finished = false;
            for record in &records {
                if !record.id.starts_with(&self.prefix) {
                    finished = true;
                    break;
                }
                ensure!(sources.len() < MAX_SOURCES, "too many stored sources");
                let size = bounded_size(&record.metadata, MAX_METADATA_BYTES)?;
                add_size(&mut metadata_bytes, size, "stored metadata exceeds 8 MiB")?;
                let (source, source_claims) = self.decode(record)?;
                add_size(
                    &mut text_bytes,
                    source.text.len(),
                    "stored source text exceeds 8 MiB",
                )?;
                ensure!(
                    claims
                        .len()
                        .checked_add(source_claims.len())
                        .is_some_and(|n| n <= MAX_CLAIMS),
                    "too many stored claims"
                );
                sources.push(source);
                claims.extend(source_claims);
                last = Some(record.id.as_str());
            }
            if finished || records.len() < PAGE_SIZE {
                break;
            }
            from = format!("{}\0", last.context("document pagination did not advance")?);
        }
        sources.sort_by(|a, b| a.id.cmp(&b.id));
        claims.sort_by(|a, b| a.id.cmp(&b.id));
        validate_snapshot(&sources, &claims)?;
        Ok((sources, claims))
    }

    pub fn lexical(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        ensure!(
            !query.trim().is_empty() && query.len() <= 8192,
            "invalid lexical query"
        );
        ensure!(limit <= MAX_SOURCES, "source limit exceeds 1024");
        let (sources, _) = self.snapshot()?;
        if sources.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let by_document: BTreeMap<_, _> = sources
            .iter()
            .map(|source| (self.document_id(&source.id), source))
            .collect();
        let mut ids = Vec::new();
        let mut remaining: BTreeSet<_> = by_document.keys().cloned().collect();
        let mut seen = BTreeSet::new();
        while ids.len() < limit && !remaining.is_empty() {
            let count = (limit - ids.len()).min(1000);
            let results = self.core.search_documents(
                query,
                &SearchOptions {
                    k: Some(count as u32),
                    candidate_limit: Some(MAX_SOURCES as u32),
                    document_ids: Some(remaining.iter().cloned().collect()),
                    lexical_weight: Some(1.0),
                    semantic_weight: Some(0.0),
                    ..Default::default()
                },
            )?;
            let exhausted = results.len() < count;
            for result in results {
                let source = by_document
                    .get(&result.document_id)
                    .context("lexical result is outside the namespace")?;
                ensure!(
                    result.version == document_version(&source.version),
                    "lexical result version changed"
                );
                ensure!(seen.insert(&source.id), "duplicate lexical source result");
                ensure!(
                    remaining.remove(&result.document_id),
                    "lexical result was already returned"
                );
                ids.push(source.id.clone());
            }
            if exhausted {
                break;
            }
        }
        Ok(ids)
    }

    pub fn matching(&self, selection: &Selection, limit: usize) -> Result<Vec<String>> {
        validate_id(&selection.subject)?;
        validate_id(&selection.predicate)?;
        if let Some(window) = selection.event_window {
            window.validate()?;
        }
        ensure!(limit <= MAX_SOURCES, "source limit exceeds 1024");
        let (_, claims) = self.snapshot()?;
        let mut ids = BTreeSet::new();
        for claim in &claims {
            if claim.subject == selection.subject
                && claim.predicate == selection.predicate
                && claim.modality == selection.modality
            {
                ids.insert(owner(claim)?.to_owned());
            }
        }
        Ok(ids.into_iter().take(limit).collect())
    }

    /// Hydrates whole source units, including evidence and supersession dependencies.
    pub fn load(&self, ids: &[String]) -> Result<(Vec<Source>, Vec<Claim>)> {
        let mut selected = requested_ids(ids)?;
        let (sources, claims) = self.snapshot()?;
        let by_source: BTreeMap<_, _> = sources
            .iter()
            .map(|source| (source.id.as_str(), source))
            .collect();
        ensure!(
            selected
                .iter()
                .all(|id| by_source.contains_key(id.as_str())),
            "requested source is absent"
        );
        let by_claim: BTreeMap<_, _> = claims
            .iter()
            .map(|claim| (claim.id.as_str(), claim))
            .collect();
        let mut dependencies: BTreeMap<&str, BTreeSet<&str>> = sources
            .iter()
            .map(|source| (source.id.as_str(), BTreeSet::new()))
            .collect();
        for claim in &claims {
            let owner_id = owner(claim)?;
            for span in &claim.evidence {
                dependencies
                    .get_mut(owner_id)
                    .expect("validated claim owner")
                    .insert(span.source_id.as_str());
            }
            for superseded in &claim.supersedes {
                let predecessor = owner(by_claim[superseded.as_str()])?;
                dependencies
                    .get_mut(owner_id)
                    .expect("validated claim owner")
                    .insert(predecessor);
                dependencies
                    .get_mut(predecessor)
                    .expect("validated supersession owner")
                    .insert(owner_id);
            }
        }
        let mut pending: Vec<_> = selected.iter().cloned().collect();
        while let Some(id) = pending.pop() {
            for &dependency in &dependencies[id.as_str()] {
                if selected.insert(dependency.to_owned()) {
                    ensure!(
                        selected.len() <= MAX_SOURCES,
                        "source dependency budget exceeded"
                    );
                    pending.push(dependency.to_owned());
                }
            }
        }
        let sources: Vec<_> = sources
            .into_iter()
            .filter(|source| selected.contains(&source.id))
            .collect();
        let claims: Vec<_> = claims
            .into_iter()
            .filter(|claim| selected.contains(&claim.evidence[0].source_id))
            .collect();
        validate_snapshot(&sources, &claims)?;
        Ok((sources, claims))
    }

    /// Uses WARM deletion and scrubbing compaction; it does not promise erasure from backups.
    pub fn forget_sources(&mut self, ids: &[String]) -> Result<usize> {
        let requested = requested_ids(ids)?;
        self.recover_pending()?;
        if requested.is_empty() {
            return Ok(0);
        }
        let (sources, claims) = self.snapshot()?;
        let remaining_sources: Vec<_> = sources
            .iter()
            .filter(|source| !requested.contains(&source.id))
            .cloned()
            .collect();
        let remaining_claims: Vec<_> = claims
            .into_iter()
            .filter(|claim| !requested.contains(&claim.evidence[0].source_id))
            .collect();
        validate_snapshot(&remaining_sources, &remaining_claims)?;
        let deleting: Vec<_> = sources
            .iter()
            .filter(|source| requested.contains(&source.id))
            .map(VersionPin::of)
            .collect();
        if deleting.is_empty() {
            return Ok(0);
        }
        let mut control = self.read_control()?;
        let mut revoked: BTreeSet<_> = control.revoked.into_iter().collect();
        revoked.extend(deleting.iter().cloned());
        control.revoked = revoked.into_iter().collect();
        control.pending = deleting;
        validate_control(&control, &self.namespace)?;
        self.write_control(&control)?;
        self.complete_pending(control)
    }

    fn control_id(&self) -> String {
        format!("evidence-control:v1:{}", hex(&self.namespace))
    }

    fn read_control(&self) -> Result<Control> {
        let Some(record) = self.core.get_document(&self.control_id()) else {
            return Ok(Control {
                schema_version: SCHEMA_VERSION,
                namespace: self.namespace.clone(),
                revoked: Vec::new(),
                pending: Vec::new(),
            });
        };
        ensure!(
            record.version == CONTROL_VERSION,
            "invalid deletion control version"
        );
        ensure!(
            record.chunk_count == 1
                && record.text.as_deref() == Some(CONTROL_TEXT)
                && record.source_uri.is_none(),
            "invalid deletion control document"
        );
        bounded_size(&record.metadata, MAX_CONTROL_BYTES)?;
        let control: Control =
            serde_json::from_value(record.metadata).context("invalid deletion control metadata")?;
        validate_control(&control, &self.namespace)?;
        Ok(control)
    }

    fn write_control(&self, control: &Control) -> Result<()> {
        validate_control(control, &self.namespace)?;
        self.core.memorize_document_with(
            DocumentInput {
                id: self.control_id(),
                text: CONTROL_TEXT.into(),
                source_uri: None,
                version: Some(CONTROL_VERSION.into()),
                metadata: Some(serde_json::to_value(control)?),
                chunk_words: None,
                overlap_words: None,
                store_text: Some(true),
                embeddings: None,
                title_weight: None,
            },
            &IngestOptions {
                encode: EncodeMode::Lexical,
                positions: true,
                whole_document: true,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn recover_pending(&self) -> Result<()> {
        let control = self.read_control()?;
        if !control.pending.is_empty() {
            self.complete_pending(control)?;
        }
        Ok(())
    }

    fn complete_pending(&self, control: Control) -> Result<usize> {
        validate_control(&control, &self.namespace)?;
        let (sources, claims) = self.source_snapshot()?;
        let pending: BTreeMap<_, _> = control
            .pending
            .iter()
            .map(|pin| (pin.source_id.as_str(), pin.source_version.as_str()))
            .collect();
        for source in &sources {
            if let Some(version) = pending.get(source.id.as_str()) {
                ensure!(
                    source.version == *version,
                    "pending deletion source version differs"
                );
            }
        }
        let survivors: Vec<_> = sources
            .into_iter()
            .filter(|source| !pending.contains_key(source.id.as_str()))
            .collect();
        let surviving_claims: Vec<_> = claims
            .into_iter()
            .filter(|claim| !pending.contains_key(claim.evidence[0].source_id.as_str()))
            .collect();
        validate_snapshot(&survivors, &surviving_claims)?;
        validate_live_sources(&survivors, &control)?;
        let documents = control
            .pending
            .iter()
            .map(|pin| self.document_id(&pin.source_id))
            .collect();
        let mut finalized = control.clone();
        finalized.pending.clear();
        validate_control(&finalized, &self.namespace)?;
        let report = self.memory.forget(
            &ForgetScope::Ids { ids: documents },
            &ForgetOptions {
                purge: true,
                ..Default::default()
            },
        )?;
        if !report.purged {
            self.core.compact_scrubbing()?;
        }
        self.write_control(&finalized)?;
        Ok(report.forgotten.len())
    }

    fn document_id(&self, source_id: &str) -> String {
        format!("{}{}", self.prefix, hex(source_id))
    }

    fn decode(&self, record: &DocumentRecord) -> Result<(Source, Vec<Claim>)> {
        let metadata: Metadata = serde_json::from_value(record.metadata.clone())
            .context("invalid evidence store metadata")?;
        ensure!(
            metadata.schema_version == SCHEMA_VERSION,
            "unsupported evidence metadata schema"
        );
        ensure!(
            metadata.namespace == self.namespace,
            "evidence metadata namespace differs"
        );
        ensure!(
            record.id == self.document_id(&metadata.source_id),
            "evidence metadata source identity differs"
        );
        ensure!(
            record.version == document_version(&metadata.source_version),
            "evidence metadata version differs"
        );
        ensure!(
            record.chunk_count == 1,
            "evidence source requires one whole document chunk"
        );
        let source = Source {
            id: metadata.source_id,
            version: metadata.source_version,
            text: record
                .text
                .clone()
                .context("evidence source text is absent")?,
            observed_at: metadata.observed_at,
        };
        validate_sources(std::slice::from_ref(&source))?;
        for claim in &metadata.claims {
            ensure!(
                owner(claim)? == source.id,
                "stored claim owner differs from source metadata"
            );
        }
        Ok((source, metadata.claims))
    }
}

fn owner(claim: &Claim) -> Result<&str> {
    Ok(claim
        .evidence
        .first()
        .context("claim evidence is absent")?
        .source_id
        .as_str())
}

fn source_metadata(namespace: &str, source: &Source, claims: &[Claim]) -> Metadata {
    Metadata {
        schema_version: SCHEMA_VERSION,
        namespace: namespace.into(),
        source_id: source.id.clone(),
        source_version: source.version.clone(),
        observed_at: source.observed_at,
        claims: claims
            .iter()
            .filter(|claim| claim.evidence[0].source_id == source.id)
            .cloned()
            .collect(),
    }
}

fn validate_control(control: &Control, namespace: &str) -> Result<()> {
    ensure!(
        control.schema_version == SCHEMA_VERSION,
        "unsupported deletion control schema"
    );
    ensure!(
        control.namespace == namespace,
        "deletion control namespace differs"
    );
    ensure!(
        control.revoked.len() <= MAX_REVOKED_VERSIONS,
        "too many revoked source versions"
    );
    ensure!(
        control.pending.len() <= MAX_PENDING_DELETIONS,
        "too many pending source deletions"
    );
    bounded_size(control, MAX_CONTROL_BYTES)?;
    let mut revoked = BTreeSet::new();
    for pin in &control.revoked {
        validate_id(&pin.source_id)?;
        validate_id(&pin.source_version)?;
        ensure!(revoked.insert(pin), "duplicate revoked source version");
    }
    let mut pending_ids = BTreeSet::new();
    for pin in &control.pending {
        validate_id(&pin.source_id)?;
        validate_id(&pin.source_version)?;
        ensure!(
            pending_ids.insert(&pin.source_id),
            "duplicate pending source identity"
        );
        ensure!(
            revoked.contains(pin),
            "pending source version is not revoked"
        );
    }
    Ok(())
}

fn validate_live_sources(sources: &[Source], control: &Control) -> Result<()> {
    let revoked: BTreeSet<_> = control.revoked.iter().collect();
    ensure!(
        sources
            .iter()
            .all(|source| !revoked.contains(&VersionPin::of(source))),
        "stored source version was revoked by an accepted deletion"
    );
    Ok(())
}

fn validate_snapshot(sources: &[Source], claims: &[Claim]) -> Result<()> {
    validate_claims(claims, sources)?;
    crate::execute::supersession_links(claims)?;
    Ok(())
}

fn requested_ids(ids: &[String]) -> Result<BTreeSet<String>> {
    ensure!(ids.len() <= MAX_SOURCES, "too many requested sources");
    for id in ids {
        validate_id(id)?;
    }
    Ok(ids.iter().cloned().collect())
}

fn hex(value: &str) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.bytes() {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 15) as usize] as char);
    }
    encoded
}

fn document_version(source_version: &str) -> String {
    format!("evidence:v1:{}", hex(source_version))
}

fn add_size(total: &mut usize, size: usize, message: &'static str) -> Result<()> {
    *total = total
        .checked_add(size)
        .filter(|n| *n <= MAX_TOTAL_BYTES)
        .context(message)?;
    Ok(())
}

fn bounded_size(value: &impl Serialize, limit: usize) -> Result<usize> {
    struct Counter {
        size: usize,
        limit: usize,
    }
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.size = self
                .size
                .checked_add(bytes.len())
                .filter(|n| *n <= self.limit)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "evidence metadata exceeds size bound",
                    )
                })?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter { size: 0, limit };
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_control_is_rejected_before_any_control_record_is_written() {
        let mut control = Control {
            schema_version: SCHEMA_VERSION,
            namespace: "bounds".into(),
            revoked: (0..128)
                .map(|i| VersionPin {
                    source_id: format!("r{i:04}{}", "x".repeat(251)),
                    source_version: "v".repeat(256),
                })
                .collect(),
            pending: Vec::new(),
        };
        let mut excess = bounded_size(&control, usize::MAX).unwrap() - (MAX_CONTROL_BYTES + 1);
        for pin in &mut control.revoked {
            let remove = excess.min(255);
            pin.source_version.truncate(256 - remove);
            excess -= remove;
        }
        assert_eq!(excess, 0);
        assert_eq!(
            bounded_size(&control, usize::MAX).unwrap(),
            MAX_CONTROL_BYTES + 1
        );
        let directory = tempfile::tempdir().unwrap();
        let store = EvidenceStore::open(directory.path(), "bounds").unwrap();
        assert!(store.write_control(&control).is_err());
        assert!(store.core.get_document(&store.control_id()).is_none());
    }
}
