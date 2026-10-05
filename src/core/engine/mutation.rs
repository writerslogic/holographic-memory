// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{HmsCore, ShardSet};
use crate::core::{
    entangled::EntangledHVec,
    rules::CompositionRule,
    types::{Relation, RelationType},
};

pub(super) const TRANSACTION_MAGIC: &[u8] = b"HMS-TXN-2\0";
pub(super) const MEANING_PREFIX: &str = "hms:meaning:";

#[derive(Clone, Serialize, Deserialize)]
pub(super) enum Mutation {
    Vector {
        id: String,
        vector: EntangledHVec,
    },
    Atom {
        id: String,
        vector: EntangledHVec,
    },
    Composite {
        id: String,
        vector: EntangledHVec,
    },
    StoredTriple(crate::core::triple_store::TripleRecord),
    Document(crate::core::documents::StoredDocument),
    Derived {
        id: String,
        children: Vec<String>,
    },
    DeleteDocument {
        id: String,
    },
    Delete {
        id: String,
    },
    Triplet {
        id: String,
        subject: String,
        relation: String,
        object: String,
    },
    Relation(Relation),
    RemoveRelation {
        source: String,
        relation: String,
        target: String,
    },
    RelationType(RelationType),
    Rule(CompositionRule),
}

pub(crate) fn validate_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty() && id.len() <= u16::MAX as usize && !id.contains('\0'),
        "IDs must contain 1..=65535 UTF-8 bytes and no null characters"
    );
    Ok(())
}

pub(super) fn validate_public_id(id: &str) -> Result<()> {
    validate_id(id)?;
    ensure!(
        !id.starts_with(MEANING_PREFIX) && !id.starts_with(crate::core::documents::CHUNK_PREFIX),
        "hms:chunk: and hms:meaning: IDs are reserved"
    );
    Ok(())
}

impl HmsCore {
    pub(super) fn commit(&self, mutations: &[Mutation]) -> Result<()> {
        ensure!(
            !mutations.is_empty() && mutations.len() <= 4096,
            "transaction requires 1..=4096 operations"
        );
        for mutation in mutations {
            self.validate_mutation(mutation)?;
        }
        let encoded = serde_json::to_vec(mutations)?;
        let mut payload = Vec::with_capacity(TRANSACTION_MAGIC.len() + encoded.len());
        payload.extend_from_slice(TRANSACTION_MAGIC);
        payload.extend_from_slice(&encoded);
        let _transaction = self.mutation_gate.write();
        let shards = self.shards.write();
        self.arena_write(&payload)?;
        self.apply_mutations(mutations, &shards)?;
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    pub(super) fn validate_mutation(&self, mutation: &Mutation) -> Result<()> {
        match mutation {
            Mutation::Vector { id, vector }
            | Mutation::Atom { id, vector }
            | Mutation::Composite { id, vector } => {
                validate_id(id)?;
                ensure!(
                    vector.dim == self.dimensions
                        && vector
                            .indices()
                            .last()
                            .is_none_or(|&i| (i as usize) < self.dimensions)
                        && vector.indices().windows(2).all(|p| p[0] < p[1]),
                    "invalid vector dimensions or indices"
                );
            }
            Mutation::Document(document) => {
                validate_id(&document.id)?;
                ensure!(
                    !document.chunks.is_empty() && document.chunks.len() <= 4096,
                    "invalid document chunk count"
                );
                for chunk in &document.chunks {
                    validate_id(&chunk.id)?;
                    ensure!(
                        chunk.vector.dim == self.dimensions
                            && chunk
                                .vector
                                .indices()
                                .last()
                                .is_none_or(|&i| (i as usize) < self.dimensions)
                            && chunk.vector.indices().windows(2).all(|p| p[0] < p[1])
                            && chunk.start_byte < chunk.end_byte
                            && chunk.end_byte as usize
                                <= crate::core::documents::MAX_DOCUMENT_BYTES
                            && (chunk.end_byte - chunk.start_byte) as usize
                                <= crate::core::documents::MAX_CHUNK_BYTES
                            && chunk
                                .text
                                .as_ref()
                                .is_none_or(|text| text.len()
                                    == (chunk.end_byte - chunk.start_byte) as usize),
                        "document vector dimension mismatch"
                    );
                    if let Some(embedding) = &chunk.embedding {
                        let dim = self
                            .config
                            .embedding_space
                            .as_ref()
                            .map_or(0, |s| s.dimensions);
                        crate::core::documents::normalize_embedding(embedding, dim)?;
                    }
                }
            }
            Mutation::Derived { id, children } => {
                validate_public_id(id)?;
                let prefix = format!("{MEANING_PREFIX}{}:{id}:", id.len());
                ensure!(
                    children.len() <= 4094
                        && children.iter().all(|child| child.starts_with(&prefix)),
                    "invalid derived meaning ownership"
                );
                for child in children {
                    validate_id(child)?;
                }
            }
            Mutation::DeleteDocument { id } => validate_id(id)?,
            Mutation::StoredTriple(t) => {
                ensure!(
                    self.meaning_enabled(),
                    "stored triple requires meaning memory"
                );
                for id in [&t.subject_id, &t.relation_id, &t.object_id, &t.composite_id] {
                    validate_id(id)?;
                }
            }
            Mutation::Delete { id } => validate_id(id)?,
            Mutation::Triplet {
                id,
                subject,
                relation,
                object,
            } => {
                for id in [id, subject, relation, object] {
                    validate_id(id)?;
                }
                ensure!(
                    self.meaning_enabled(),
                    "triplet ingestion requires meaningEnabled"
                );
            }
            Mutation::Relation(r) => {
                for id in [&r.source_id, &r.relation_type, &r.target_id] {
                    validate_id(id)?;
                }
                ensure!(
                    r.valid_from.is_finite()
                        && r.valid_to.is_finite()
                        && r.valid_from >= 0.0
                        && r.valid_to >= 0.0
                        && (r.valid_to == 0.0 || r.valid_to >= r.valid_from),
                    "invalid relation validity interval"
                );
            }
            Mutation::RemoveRelation {
                source,
                relation,
                target,
            } => {
                for id in [source, relation, target] {
                    validate_id(id)?;
                }
            }
            Mutation::RelationType(r) => validate_id(&r.name)?,
            Mutation::Rule(r) => {
                ensure!(
                    self.meaning_enabled(),
                    "composition rules require meaningEnabled"
                );
                validate_id(&r.name)?;
                validate_id(&r.output_relation)?;
                ensure!(
                    !r.input_relations.is_empty(),
                    "rule requires input relations"
                );
                for id in &r.input_relations {
                    validate_id(id)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn apply_mutations(&self, mutations: &[Mutation], shards: &ShardSet) -> Result<()> {
        for mutation in mutations {
            self.validate_mutation(mutation)?;
            match mutation {
                Mutation::Vector { id, vector } => {
                    self.remove_derived_from_memory(id, shards)?;
                    if let Some(composites) = &self.composite_memory {
                        composites.delete(id);
                    }
                    if let Some(triples) = &self.triple_store {
                        triples.remove_composite(id);
                    }
                    shards.insert(id.clone(), vector.clone(), self.dimensions)?;
                    if let Some(atoms) = &self.atom_memory {
                        atoms.insert_with_vec(id, vector);
                    }
                }
                Mutation::Atom { id, vector } => {
                    if let Some(atoms) = &self.atom_memory {
                        atoms.insert_with_vec(id, vector);
                    }
                }
                Mutation::Composite { id, vector } => {
                    if let Some(composites) = &self.composite_memory {
                        composites.insert(id.clone(), vector.clone());
                    }
                    shards.insert(id.clone(), vector.clone(), self.dimensions)?;
                }
                Mutation::StoredTriple(record) => {
                    if let Some(triples) = &self.triple_store {
                        triples.load_triple(record.clone());
                    }
                }
                Mutation::Document(document) => {
                    self.remove_document_from_memory(&document.id, shards)?;
                    for chunk in &document.chunks {
                        shards.insert(chunk.id.clone(), chunk.vector.clone(), self.dimensions)?;
                    }
                    self.documents
                        .write()
                        .insert(document.id.clone(), document.clone());
                }
                Mutation::Derived { id, children } => {
                    self.derived.write().insert(id.clone(), children.clone());
                }
                Mutation::DeleteDocument { id } => {
                    self.remove_document_from_memory(id, shards)?;
                }
                Mutation::Delete { id } => {
                    self.remove_derived_from_memory(id, shards)?;
                    self.remove_memory_id(id, shards)?;
                }
                Mutation::Triplet {
                    id,
                    subject,
                    relation,
                    object,
                } => {
                    self.remove_derived_from_memory(id, shards)?;
                    let atoms = self
                        .atom_memory
                        .as_ref()
                        .expect("validated meaning configuration");
                    let composites = self
                        .composite_memory
                        .as_ref()
                        .expect("validated meaning configuration");
                    let triples = self
                        .triple_store
                        .as_ref()
                        .expect("validated meaning configuration");
                    let roles = self
                        .role_registry
                        .as_ref()
                        .expect("validated meaning configuration");
                    let (_, s) = atoms.get_or_insert(subject);
                    let (_, r) = atoms.get_or_insert(relation);
                    let (_, o) = atoms.get_or_insert(object);
                    let vector = roles.compose_triple(&s, &r, &o);
                    composites.insert(id.clone(), vector.clone());
                    triples.remove_composite(id);
                    triples.add(subject, relation, object, id);
                    shards.insert(id.clone(), vector, self.dimensions)?;
                }
                Mutation::Relation(r) => {
                    self.graph
                        .remove(&r.source_id, &r.relation_type, &r.target_id);
                    self.graph.add(r);
                }
                Mutation::RemoveRelation {
                    source,
                    relation,
                    target,
                } => {
                    self.graph.remove(source, relation, target);
                }
                Mutation::RelationType(r) => self.graph.declare_type(r.clone()),
                Mutation::Rule(r) => {
                    self.rule_store
                        .as_ref()
                        .expect("validated meaning configuration")
                        .add_rule(r.clone());
                }
            }
        }
        Ok(())
    }

    fn remove_derived_from_memory(&self, id: &str, shards: &ShardSet) -> Result<()> {
        if let Some(children) = self.derived.write().remove(id) {
            for child in children {
                self.remove_memory_id(&child, shards)?;
            }
        }
        Ok(())
    }

    fn remove_memory_id(&self, id: &str, shards: &ShardSet) -> Result<()> {
        shards.remove(id, self.dimensions)?;
        if let Some(atoms) = &self.atom_memory {
            atoms.delete(id);
        }
        if let Some(composites) = &self.composite_memory {
            composites.delete(id);
        }
        if let Some(triples) = &self.triple_store {
            triples.remove_composite(id);
        }
        Ok(())
    }
}
