// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Binary form of stored documents in `HMS-TXN-3` journal frames (docs/file-format/MUTATIONS.md).
//!
//! A frame carries one sorted, front-coded term dictionary; each chunk stores its terms as
//! ascending dictionary ids (delta-coded), then their frequencies, so a term's bytes are written
//! once per frame instead of once per chunk. Every length and count read is checked against
//! the bytes that remain before anything is allocated from it.

use anyhow::{bail, ensure, Context, Result};
use std::borrow::Cow;
use std::sync::Arc;

use super::packed_terms::{SharedDictionary, StoredTerms};
use super::text::TextSegment;
use super::{Embedding, StoredChunk, StoredDocument, VectorState};
use crate::core::entangled::EntangledHVec;

const CHUNK_EXPLICIT_ID: u8 = 1;
const CHUNK_TEXT: u8 = 1 << 1;
const CHUNK_STEMMED: u8 = 1 << 2;
const CHUNK_EMBEDDING: u8 = 1 << 3;
const CHUNK_STATE_SHIFT: u8 = 4;
const CHUNK_POSITIONS: u8 = 1 << 6;
const CHUNK_FLAGS: u8 = 0b0111_1111;

pub(crate) fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub(crate) fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A cursor over untrusted bytes; every read fails instead of panicking or over-reading.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.at == self.data.len()
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.at
    }

    /// Offset of the next unread byte.
    pub(crate) fn position(&self) -> usize {
        self.at
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        ensure!(
            len <= self.remaining(),
            "binary journal record is truncated"
        );
        let bytes = &self.data[self.at..self.at + len];
        self.at += len;
        Ok(bytes)
    }

    pub(crate) fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            ensure!(
                shift < 63 || byte <= 1,
                "binary journal varint overflows 64 bits"
            );
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("binary journal varint overflows 64 bits")
    }

    fn number<T: TryFrom<u64>>(&mut self, what: &str) -> Result<T> {
        let value = self.varint()?;
        T::try_from(value)
            .ok()
            .with_context(|| format!("{what} {value} is out of range"))
    }

    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.number::<usize>("byte length")?;
        self.take(len)
    }

    pub(crate) fn str(&mut self) -> Result<&'a str> {
        std::str::from_utf8(self.bytes()?).context("binary journal string is not UTF-8")
    }

    /// A count of elements that take at least `min` bytes each: bounded by the bytes left, so
    /// a hostile count cannot size an allocation.
    pub(crate) fn count(&mut self, min: usize) -> Result<usize> {
        let count = self.number::<usize>("count")?;
        ensure!(
            count <= self.remaining() / min.max(1),
            "binary journal count {count} exceeds the record"
        );
        Ok(count)
    }
}

/// Writer side of a frame's term dictionary: the distinct terms of the frame's documents in
/// byte order, and for every (chunk, term) of those documents, in order, its dictionary id.
pub(crate) struct FrameTerms<'a> {
    sorted: Vec<&'a str>,
    ids: Vec<u32>,
}

/// Next ids to write, in the order the documents were given to [`FrameTerms::new`].
pub(crate) struct TermCursor<'a> {
    ids: std::slice::Iter<'a, u32>,
}

impl<'a> FrameTerms<'a> {
    /// The dictionary of `documents`; they must then be written in this order with
    /// [`write_document`]. One hash lookup per (chunk, term), then a sort of the distinct
    /// terms only.
    pub(crate) fn new(documents: impl Iterator<Item = &'a StoredDocument> + Clone) -> Self {
        let pairs: usize = documents
            .clone()
            .flat_map(|d| d.chunks.iter())
            .map(|c| c.terms.keys().len())
            .sum();
        let mut provisional: std::collections::HashMap<&'a str, u32, foldhash::fast::RandomState> =
            std::collections::HashMap::with_capacity_and_hasher(pairs / 8, Default::default());
        let mut order: Vec<&'a str> = Vec::new();
        let mut ids = Vec::with_capacity(pairs);
        for document in documents {
            for chunk in &document.chunks {
                for term in chunk.terms.keys() {
                    let next = order.len() as u32;
                    let id = *provisional.entry(term).or_insert_with(|| {
                        order.push(term);
                        next
                    });
                    ids.push(id);
                }
            }
        }
        let mut by_term: Vec<u32> = (0..order.len() as u32).collect();
        by_term.sort_unstable_by(|&a, &b| order[a as usize].cmp(order[b as usize]));
        let mut rank = vec![0u32; order.len()];
        for (position, &id) in by_term.iter().enumerate() {
            rank[id as usize] = position as u32;
        }
        for id in &mut ids {
            *id = rank[*id as usize];
        }
        let sorted = by_term.iter().map(|&id| order[id as usize]).collect();
        Self { sorted, ids }
    }

    pub(super) fn into_shared(self) -> (Arc<SharedDictionary>, Vec<u32>) {
        (Arc::new(SharedDictionary::new(&self.sorted)), self.ids)
    }

    /// Front-coded: per term, the bytes it shares with the previous one, then the rest.
    pub(crate) fn write(&self, out: &mut Vec<u8>) {
        put_varint(out, self.sorted.len() as u64);
        let mut previous: &[u8] = &[];
        for term in &self.sorted {
            let term = term.as_bytes();
            let shared = previous
                .iter()
                .zip(term)
                .take_while(|(a, b)| a == b)
                .count();
            put_varint(out, shared as u64);
            put_bytes(out, &term[shared..]);
            previous = term;
        }
    }

    pub(crate) fn cursor(&self) -> TermCursor<'_> {
        TermCursor {
            ids: self.ids.iter(),
        }
    }
}

/// Reader side of a frame's term dictionary.
pub(crate) struct DecodedTerms {
    dictionary: Arc<SharedDictionary>,
}

impl DecodedTerms {
    pub(crate) fn read(reader: &mut Reader<'_>) -> Result<Self> {
        // Each entry is at least two varints.
        let count = reader.count(2)?;
        let mut bytes: Vec<u8> = Vec::new();
        let mut offsets = Vec::with_capacity(count + 1);
        offsets.push(0);
        for _ in 0..count {
            let previous_start = offsets[offsets.len().saturating_sub(2)];
            let previous_len = bytes.len() - previous_start;
            let shared = reader.number::<usize>("shared prefix")?;
            ensure!(
                shared <= previous_len && (offsets.len() > 1 || shared == 0),
                "binary journal term prefix exceeds the previous term"
            );
            let suffix = reader.bytes()?;
            // Shared prefixes repeat earlier bytes, so the decoded size is bounded separately:
            // a frame's terms never exceed the largest frame payload.
            ensure!(
                bytes.len() + shared + suffix.len() <= crate::core::storage::MAX_RAW_FRAME_SIZE,
                "binary journal term dictionary exceeds the frame size limit"
            );
            let start = bytes.len();
            bytes.extend_from_within(previous_start..previous_start + shared);
            bytes.extend_from_slice(suffix);
            let term = &bytes[start..];
            ensure!(
                std::str::from_utf8(term).is_ok(),
                "binary journal term is not UTF-8"
            );
            ensure!(
                offsets.len() == 1 || &bytes[previous_start..start] < term,
                "binary journal terms are not strictly ascending"
            );
            offsets.push(bytes.len());
        }
        let text = String::from_utf8(bytes).context("binary dictionary is not UTF-8")?;
        let starts = offsets[..count]
            .iter()
            .map(|&v| u32::try_from(v))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("dictionary exceeds u32")?;
        Ok(Self {
            dictionary: Arc::new(SharedDictionary::from_parts(text, starts)),
        })
    }

    fn len(&self) -> usize {
        self.dictionary.len()
    }
}

/// Whether [`write_document`] can carry `document` exactly: chunk vector indices strictly
/// ascending, which every committed document satisfies (`validate_mutation`).
pub(crate) fn representable(document: &StoredDocument) -> bool {
    document
        .chunks
        .iter()
        .all(|chunk| chunk.vector.indices().windows(2).all(|p| p[0] < p[1]))
}

fn state_bits(state: VectorState) -> u8 {
    match state {
        VectorState::Encoded => 0,
        VectorState::Pending => 1,
        VectorState::Lexical => 2,
    }
}

/// Writes `document` (it must be [`representable`]); its chunk text comes from the chunk or,
/// once moved there, from `segment`. Consumes the document's ids from `terms`.
pub(crate) fn write_document(
    out: &mut Vec<u8>,
    document: &StoredDocument,
    segment: Option<&TextSegment>,
    terms: &mut TermCursor<'_>,
) -> Result<()> {
    debug_assert!(representable(document));
    put_bytes(out, document.id.as_bytes());
    match &document.source_uri {
        None => out.push(0),
        Some(uri) => {
            out.push(1);
            put_bytes(out, uri.as_bytes());
        }
    }
    put_bytes(out, document.version.as_bytes());
    put_bytes(out, &serde_json::to_vec(&document.metadata)?);
    put_varint(out, document.chunks.len() as u64);
    for (index, chunk) in document.chunks.iter().enumerate() {
        let text: Option<Cow<'_, str>> = match (&chunk.text, chunk.text_at) {
            (Some(text), _) => Some(Cow::Borrowed(text)),
            (None, Some((at, len))) => Some(Cow::Owned(
                segment
                    .context("chunk text moved to a segment that is not attached")?
                    .read(at, len)?,
            )),
            (None, None) => None,
        };
        let explicit_id = !super::is_chunk_id(&chunk.id, &document.id, index);
        let flags = (u8::from(explicit_id) * CHUNK_EXPLICIT_ID)
            | (u8::from(text.is_some()) * CHUNK_TEXT)
            | (u8::from(chunk.stemmed) * CHUNK_STEMMED)
            | (u8::from(chunk.embedding.is_some()) * CHUNK_EMBEDDING)
            | (state_bits(chunk.vector_state) << CHUNK_STATE_SHIFT)
            | (u8::from(chunk.terms.positions.is_some()) * CHUNK_POSITIONS);
        out.push(flags);
        if explicit_id {
            put_bytes(out, chunk.id.as_bytes());
        }
        if let Some(text) = &text {
            put_bytes(out, text.as_bytes());
        }
        put_varint(out, u64::from(chunk.start_byte));
        put_varint(out, u64::from(chunk.end_byte));
        put_varint(out, chunk.word_count as u64);
        put_varint(out, chunk.vector.dim as u64);
        let indices = chunk.vector.indices();
        put_varint(out, indices.len() as u64);
        let mut previous = 0u32;
        for (i, &index) in indices.iter().enumerate() {
            put_varint(
                out,
                u64::from(if i == 0 { index } else { index - previous }),
            );
            previous = index;
        }
        if let Some(embedding) = &chunk.embedding {
            put_varint(out, embedding.len() as u64);
            for value in embedding.iter() {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        let count = chunk.terms.keys().len();
        put_varint(out, count as u64);
        let mut previous = 0u32;
        for i in 0..count {
            let id = *terms
                .ids
                .next()
                .context("frame dictionary ran out of ids")?;
            put_varint(out, u64::from(if i == 0 { id } else { id - previous }));
            previous = id;
        }
        for (_, &frequency) in &chunk.terms {
            put_varint(out, u64::from(frequency));
        }
        if let Some(positions) = &chunk.terms.positions {
            out.push(positions.version);
            put_varint(out, positions.offsets.len() as u64);
            for offset in positions.offsets.iter() {
                put_varint(out, u64::from(offset));
            }
            put_bytes(out, &positions.data);
        }
    }
    Ok(())
}

/// Reads a document written by [`write_document`], exactly as the JSON form of the same
/// document deserializes.
pub(crate) fn read_document(
    reader: &mut Reader<'_>,
    terms: &DecodedTerms,
) -> Result<StoredDocument> {
    let id = reader.str()?.to_owned();
    let source_uri = match reader.byte()? {
        0 => None,
        1 => Some(reader.str()?.to_owned()),
        other => bail!("binary journal source flag {other} is invalid"),
    };
    let version = reader.str()?.to_owned();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(reader.bytes()?).context("binary journal metadata")?;
    super::compact_metadata(&mut metadata);
    // A chunk is at least its flags and seven varints.
    let count = reader.count(8)?;
    let mut chunks = Vec::with_capacity(count);
    for index in 0..count {
        let flags = reader.byte()?;
        ensure!(
            flags & !CHUNK_FLAGS == 0,
            "binary journal chunk flags {flags:#x} are invalid"
        );
        let vector_state = match (flags >> CHUNK_STATE_SHIFT) & 3 {
            0 => VectorState::Encoded,
            1 => VectorState::Pending,
            2 => VectorState::Lexical,
            other => bail!("binary journal vector state {other} is invalid"),
        };
        let chunk_id = if flags & CHUNK_EXPLICIT_ID != 0 {
            reader.str()?.to_owned()
        } else {
            super::chunk_id(&id, index)
        };
        let text = if flags & CHUNK_TEXT != 0 {
            Some(reader.str()?.to_owned())
        } else {
            None
        };
        let start_byte = reader.number::<u32>("chunk start")?;
        let end_byte = reader.number::<u32>("chunk end")?;
        let word_count = reader.number::<usize>("word count")?;
        let dim = reader.number::<usize>("vector dimensions")?;
        let index_count = reader.count(1)?;
        let mut indices = Vec::with_capacity(index_count);
        let mut previous = 0u32;
        for i in 0..index_count {
            let gap = reader.number::<u32>("vector index")?;
            previous = if i == 0 {
                gap
            } else {
                previous
                    .checked_add(gap)
                    .context("binary journal vector index overflows")?
            };
            indices.push(previous);
        }
        let embedding = if flags & CHUNK_EMBEDDING != 0 {
            let len = reader.count(4)?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let bytes = reader.take(4)?;
                values.push(f32::from_le_bytes(bytes.try_into().expect("four bytes")));
            }
            Some(Embedding::from_f32(values))
        } else {
            None
        };
        // A term is at least an id varint and a frequency varint.
        let term_count = reader.count(2)?;
        let mut ids = Vec::with_capacity(term_count);
        let mut previous = 0usize;
        for i in 0..term_count {
            let gap = reader.number::<usize>("term id")?;
            let id = if i == 0 {
                gap
            } else {
                ensure!(
                    gap > 0,
                    "binary journal term ids are not strictly ascending"
                );
                previous
                    .checked_add(gap)
                    .context("binary journal term id overflows")?
            };
            ensure!(
                id < terms.len(),
                "binary journal term id {id} is not in the dictionary"
            );
            ids.push(id);
            previous = id;
        }
        let mut frequencies = Vec::with_capacity(term_count);
        for _ in 0..term_count {
            frequencies.push(reader.number::<u32>("term frequency")?);
        }
        let ids: Vec<u32> = ids
            .into_iter()
            .map(|id| u32::try_from(id).expect("bounded dictionary id"))
            .collect();
        let mut table =
            StoredTerms::from_dictionary(Arc::clone(&terms.dictionary), &ids, frequencies);
        if flags & CHUNK_POSITIONS != 0 {
            let version = reader.byte()?;
            ensure!(
                matches!(
                    version,
                    super::positions::VERSION | super::positions::DOCUMENT_VERSION
                ),
                "positional postings version {version} is newer than supported"
            );
            let count = reader.count(1)?;
            ensure!(count == term_count + 1, "positional term count mismatch");
            let mut offsets = Vec::with_capacity(count);
            for _ in 0..count {
                offsets.push(reader.number::<u32>("position offset")?);
            }
            let data = reader.bytes()?;
            ensure!(
                data.len() <= super::MAX_DOCUMENT_BYTES,
                "positional payload exceeds document limit"
            );
            let positions = Arc::new(super::positions::Positions {
                version,
                offsets: offsets.into(),
                data: data.into(),
            });
            positions.validate(&table)?;
            table.positions = Some(positions);
        }
        chunks.push(StoredChunk {
            id: chunk_id,
            text,
            text_at: None,
            start_byte,
            end_byte,
            vector: EntangledHVec { dim, indices },
            embedding,
            terms: table,
            word_count,
            stemmed: flags & CHUNK_STEMMED != 0,
            vector_state,
        });
    }
    Ok(StoredDocument {
        id,
        source_uri,
        version,
        metadata,
        chunks,
    })
}

#[cfg(test)]
mod positional_tests {
    use super::*;
    use crate::core::documents::{prepare_with_options, DocumentInput, EncodeMode, IngestOptions};
    use crate::core::encoding::TextEncoder;

    const LEGACY: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/lexical-postings-v1/legacy.bin"
    ));
    const POSITIONAL: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/lexical-postings-v1/positions-v1.bin"
    ));
    const WHOLE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/lexical-postings-v1/whole-article-v2.bin"
    ));
    const JSON: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/lexical-postings-v1/positions-v1.json"
    ));

    fn decode(bytes: &[u8]) -> Result<StoredDocument> {
        let mut reader = Reader::new(bytes);
        let terms = DecodedTerms::read(&mut reader)?;
        let document = read_document(&mut reader, &terms)?;
        ensure!(reader.is_empty(), "trailing fixture bytes");
        Ok(document)
    }

    #[test]
    fn positional_and_legacy_golden_bytes_round_trip() {
        for (positions, whole_document, bytes) in [
            (false, false, LEGACY),
            (true, false, POSITIONAL),
            (true, true, WHOLE),
        ] {
            let input: DocumentInput = serde_json::from_value(serde_json::json!({
                "id":"gold", "text":"red red fox", "storeText":false
            }))
            .unwrap();
            let document = prepare_with_options(
                input,
                64,
                None,
                None,
                TextEncoder::default(),
                &IngestOptions {
                    encode: EncodeMode::Lexical,
                    unstemmed: true,
                    positions,
                    whole_document,
                },
            )
            .unwrap();
            let dictionary = FrameTerms::new(std::iter::once(&document));
            let mut encoded = Vec::new();
            dictionary.write(&mut encoded);
            write_document(&mut encoded, &document, None, &mut dictionary.cursor()).unwrap();
            assert_eq!(encoded, bytes);
            let restored = decode(bytes).unwrap();
            assert_eq!(
                serde_json::to_value(&document).unwrap(),
                serde_json::to_value(&restored).unwrap()
            );
            assert_eq!(restored.chunks[0].terms.positions.is_some(), positions);
            for end in 0..bytes.len() {
                assert!(decode(&bytes[..end]).is_err(), "prefix {end}");
            }
        }
        let terms: StoredTerms = serde_json::from_slice(JSON).unwrap();
        assert_eq!(serde_json::to_vec(&terms).unwrap(), JSON);
        let mut future = POSITIONAL.to_vec();
        let version = future.len() - 9;
        future[version] = 3;
        assert!(decode(&future)
            .err()
            .unwrap()
            .to_string()
            .contains("newer than supported"));
    }

    #[test]
    fn malformed_positional_json_is_rejected() {
        let fixture: serde_json::Value = serde_json::from_slice(JSON).unwrap();
        for (key, value) in [
            ("version", serde_json::json!(3)),
            ("offsets", serde_json::json!([0, 0, 3])),
            ("offsets", serde_json::json!([0, 1, 4])),
            ("offsets", serde_json::json!([0, 3, 1])),
            ("data", serde_json::json!([2, 0, 0])),
            ("data", serde_json::json!([2, 0, 128])),
            ("data", serde_json::json!([2, 0, 3])),
        ] {
            let mut malformed = fixture.clone();
            malformed["$hms_positions"][key] = value;
            assert!(
                serde_json::from_value::<StoredTerms>(malformed).is_err(),
                "{key}"
            );
        }
    }
}
