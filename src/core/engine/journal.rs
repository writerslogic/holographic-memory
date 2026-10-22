// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Journal frame payloads (docs/file-format/MUTATIONS.md). A store writes the format its
//! `store.json` schema names and reads both:
//!
//! - `HMS-TXN-2\0` then the mutations as one JSON array (schema 2);
//! - `HMS-TXN-3\0` then the binary form (schema 3): the mutation count, the frame's sorted
//!   term dictionary, then per mutation a tag byte and its record. Documents and document
//!   deletes are binary; every other mutation is its JSON element, length-prefixed.

use anyhow::{bail, Result};
use std::borrow::Cow;
use std::sync::Arc;

use super::mutation::Mutation;
use crate::core::documents::binary::{
    self, put_bytes, put_varint, DecodedTerms, FrameTerms, Reader,
};
use crate::core::documents::{StoredDocument, TextSegment};

pub(crate) const JSON_MAGIC: &[u8] = crate::core::format::TRANSACTION_MAGIC;
pub(crate) const BINARY_MAGIC: &[u8] = crate::core::format::BINARY_TRANSACTION_MAGIC;

const TAG_JSON: u8 = 0;
const TAG_DOCUMENT: u8 = 1;
const TAG_DELETE_DOCUMENT: u8 = 2;

/// The frame payload format a store writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JournalFormat {
    /// `HMS-TXN-2`: stores with `store.json` schema 2.
    Json,
    /// `HMS-TXN-3`: stores with `store.json` schema 3.
    Binary,
}

impl JournalFormat {
    pub(crate) fn for_schema(version: u32) -> Self {
        if version >= 3 {
            Self::Binary
        } else {
            Self::Json
        }
    }
}

fn document_of(mutation: &Mutation) -> Option<&StoredDocument> {
    match mutation {
        Mutation::Document(document) if binary::representable(document) => Some(document),
        _ => None,
    }
}

/// `mutations` as an `HMS-TXN-3` payload.
pub(super) fn binary_payload(mutations: &[Mutation]) -> Result<Vec<u8>> {
    let terms = FrameTerms::new(mutations.iter().filter_map(document_of));
    let mut out = Vec::with_capacity(BINARY_MAGIC.len() + 64);
    out.extend_from_slice(BINARY_MAGIC);
    put_varint(&mut out, mutations.len() as u64);
    terms.write(&mut out);
    let mut cursor = terms.cursor();
    for mutation in mutations {
        if let Some(document) = document_of(mutation) {
            out.push(TAG_DOCUMENT);
            binary::write_document(&mut out, document, None, &mut cursor)?;
        } else if let Mutation::DeleteDocument { id } = mutation {
            out.push(TAG_DELETE_DOCUMENT);
            put_bytes(&mut out, id.as_bytes());
        } else {
            out.push(TAG_JSON);
            put_bytes(&mut out, &serde_json::to_vec(mutation)?);
        }
    }
    Ok(out)
}

/// One stored document as an `HMS-TXN-3` payload, its chunk text resolved from `segment`
/// (compaction frames carry text inline). `None` if the binary form cannot carry it.
pub(super) fn binary_document_payload(
    document: &StoredDocument,
    segment: Option<&TextSegment>,
) -> Result<Option<Vec<u8>>> {
    if !binary::representable(document) {
        return Ok(None);
    }
    let terms = FrameTerms::new(std::iter::once(document));
    let mut out = Vec::with_capacity(BINARY_MAGIC.len() + 64);
    out.extend_from_slice(BINARY_MAGIC);
    put_varint(&mut out, 1);
    terms.write(&mut out);
    out.push(TAG_DOCUMENT);
    binary::write_document(&mut out, document, segment, &mut terms.cursor())?;
    Ok(Some(out))
}

/// The mutations of an `HMS-TXN-3` payload body (after the magic), decoded one at a time.
pub(crate) struct BinaryDecoder<'a> {
    reader: Reader<'a>,
    terms: DecodedTerms,
    remaining: usize,
}

impl<'a> BinaryDecoder<'a> {
    pub(crate) fn new(body: &'a [u8]) -> Result<Self> {
        let mut reader = Reader::new(body);
        // Every mutation is at least its tag and one more byte.
        let remaining = reader.count(2)?;
        let terms = DecodedTerms::read(&mut reader)?;
        Ok(Self {
            reader,
            terms,
            remaining,
        })
    }

    pub(super) fn next_mutation(&mut self) -> Result<Option<Mutation>> {
        if self.remaining == 0 {
            anyhow::ensure!(
                self.reader.is_empty(),
                "binary journal payload has bytes after its last mutation"
            );
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(match self.reader.byte()? {
            TAG_JSON => serde_json::from_slice(self.reader.bytes()?)?,
            TAG_DOCUMENT => Mutation::Document(Arc::new(binary::read_document(
                &mut self.reader,
                &self.terms,
            )?)),
            TAG_DELETE_DOCUMENT => Mutation::DeleteDocument {
                id: self.reader.str()?.to_owned(),
            },
            other => bail!("binary journal mutation tag {other} is invalid"),
        }))
    }
}

/// The mutations of a transaction payload of either format.
#[cfg(test)]
pub(super) fn decode(payload: &[u8]) -> Result<Vec<Mutation>> {
    if let Some(json) = payload.strip_prefix(JSON_MAGIC) {
        return Ok(serde_json::from_slice(json)?);
    }
    let body = payload
        .strip_prefix(BINARY_MAGIC)
        .ok_or_else(|| anyhow::anyhow!("payload has no journal magic"))?;
    let mut decoder = BinaryDecoder::new(body)?;
    let mut mutations = Vec::new();
    while let Some(mutation) = decoder.next_mutation()? {
        mutations.push(mutation);
    }
    Ok(mutations)
}

/// The JSON mutation array of a transaction payload of either format, as an `HMS-TXN-2`
/// payload of the same mutations carries it; `None` without a journal magic this build reads.
/// For tools that inspect frames as JSON.
pub(crate) fn payload_json(payload: &[u8]) -> Option<Result<Cow<'_, [u8]>>> {
    if let Some(json) = payload.strip_prefix(JSON_MAGIC) {
        return Some(Ok(Cow::Borrowed(json)));
    }
    let body = payload.strip_prefix(BINARY_MAGIC)?;
    Some((|| {
        let mut decoder = BinaryDecoder::new(body)?;
        let mut mutations = Vec::new();
        while let Some(mutation) = decoder.next_mutation()? {
            mutations.push(mutation);
        }
        Ok(Cow::Owned(serde_json::to_vec(&mutations)?))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::documents::{self, EncodeMode, StoredChunk};
    use crate::core::encoding::TextEncoder;
    use crate::core::entangled::EntangledHVec;
    use crate::DocumentInput;
    use serde_json::json;

    fn input(value: serde_json::Value) -> DocumentInput {
        serde_json::from_value(value).unwrap()
    }

    /// Mutations covering every field the binary form carries: prepared documents (text kept
    /// and dropped, title weighting, embeddings, each encode mode, several chunks, non-ASCII
    /// terms) and hand-built ones (an explicit chunk id, unstemmed and long terms that keep the
    /// tree table, a vector), next to mutations carried as JSON.
    fn mutations() -> Vec<Mutation> {
        let prepare = |value, mode| {
            Mutation::Document(Arc::new(
                documents::prepare_with_options(
                    input(value),
                    64,
                    Some(2),
                    None,
                    TextEncoder::default(),
                    &documents::IngestOptions {
                        encode: mode,
                        ..Default::default()
                    },
                )
                .unwrap(),
            ))
        };
        let mut odd = documents::prepare_with_options(
            input(json!({"id": "odd", "text": "zeta alpha", "sourceUri": "s:é", "metadata": {"k": [1.5, null, "v"]}})),
            64,
            None,
            None,
            TextEncoder::default(),
            &documents::IngestOptions {
                encode: EncodeMode::Lexical,
                ..Default::default()
            },
        )
        .unwrap();
        let chunk: &mut StoredChunk = &mut odd.chunks[0];
        chunk.id = "custom:chunk".into();
        chunk.stemmed = false;
        chunk.vector = EntangledHVec {
            dim: 64,
            indices: vec![0, 7, 63],
        };
        chunk.vector_state = documents::VectorState::Encoded;
        chunk.terms =
            std::collections::BTreeMap::from([("x".repeat(40), u32::MAX), ("y".repeat(41), 0)])
                .into();
        vec![
            prepare(
                json!({"id": "a", "text": "Title Running\nrunners ran café Οδός", "titleWeight": 3,
                       "embeddings": [[1.0, -0.25]], "version": "7"}),
                EncodeMode::Eager,
            ),
            prepare(
                json!({"id": "b", "text": "one two three four five six", "chunkWords": 2,
                       "overlapWords": 1, "storeText": false}),
                EncodeMode::Lexical,
            ),
            prepare(
                json!({"id": "c", "text": "deferred words here"}),
                EncodeMode::Deferred,
            ),
            Mutation::Document(Arc::new(odd)),
            Mutation::DeleteDocument {
                id: "gone\"é".into(),
            },
            Mutation::Delete {
                id: "vector".into(),
            },
            Mutation::Vector {
                id: "v".into(),
                vector: EntangledHVec::new_deterministic(64, 3),
            },
        ]
    }

    /// A binary payload reads back as exactly the mutations it was built from: the same JSON
    /// an `HMS-TXN-2` payload of them holds, and the same binary bytes when encoded again.
    #[test]
    fn binary_payload_round_trips_every_field() -> Result<()> {
        let mutations = mutations();
        let payload = binary_payload(&mutations)?;
        assert!(payload.starts_with(BINARY_MAGIC));
        let json = payload_json(&payload).unwrap()?;
        assert_eq!(&*json, serde_json::to_vec(&mutations)?.as_slice());
        let decoded = decode(&payload)?;
        assert_eq!(binary_payload(&decoded)?, payload);
        Ok(())
    }

    /// Damaged binary payloads fail to decode; none panics or allocates from a hostile length.
    #[test]
    fn truncated_or_corrupted_payloads_are_errors() -> Result<()> {
        let payload = binary_payload(&mutations())?;
        for end in BINARY_MAGIC.len()..payload.len() {
            assert!(
                payload_json(&payload[..end]).unwrap().is_err(),
                "prefix {end}"
            );
        }
        for at in BINARY_MAGIC.len()..payload.len() {
            for value in [0x00, 0x7f, 0x80, 0xff] {
                let mut damaged = payload.clone();
                damaged[at] = value;
                let _ = payload_json(&damaged).unwrap();
            }
        }
        let mut huge = BINARY_MAGIC.to_vec();
        put_varint(&mut huge, u64::MAX);
        assert!(payload_json(&huge).unwrap().is_err());
        Ok(())
    }
}
