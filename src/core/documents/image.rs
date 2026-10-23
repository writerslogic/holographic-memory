// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The document checkpoint (`documents.ckpt`, see `docs/file-format/CHECKPOINT.md`): the whole
//! document store of a journal generation as the bytes a query reads. A generation holding one
//! opens by checking and mapping it instead of replaying document frames, ranks unfiltered
//! lexical queries straight from the mapping with the live store's code ([`LexicalSource`]),
//! and rebuilds the live store from it ([`ServingImage::hydrate`]) when anything else is asked.
//!
//! The posting lists hold the live lists' postings as varints and every count, word count and slot
//! (holes included) is the live store's, so ranking from a checkpoint is ranking from the store
//! that wrote it. Layout, little endian: an 8-byte magic, `HEADER_WORDS` `u64` header words (a
//! CRC32 of everything after the CRC word, the CRC32 of the generation directory's name, the
//! vector dimensions, the store's counts, then an `(offset, length)` pair per [`Section`]),
//! then the sections:
//!
//! - per posting form, a dictionary of the sorted terms in blocks of [`BLOCK`] front-coded
//!   entries (`prefix`, `suffix length`, suffix bytes, posting count, posting bytes; varints),
//!   and a block table of `u32` pairs: the block's dictionary offset and the posting offset of
//!   its first term;
//! - the posting bytes of every term, form 0 then form 1, in dictionary order;
//! - per slot, `u32` word count, document ordinal (`u32::MAX` for a free slot) and chunk index;
//! - document ids in id order (ordinal order): `docs + 1` `u32` offsets, then the UTF-8 bytes;
//! - per document, `docs + 1` `u32` offsets and a record: version, source URI, metadata JSON,
//!   and per chunk its flags (own id, stemmed terms), id, byte span and text (none or inline).
//!
//! A checkpoint holds lexical-only chunks (no vector, embedding or pending encoding), so the
//! chunk fields it leaves out are the same for every chunk it holds.

use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};

use super::binary::{put_bytes, put_varint, Reader};
use super::index::{DocumentStore, StoreIdentity};
use super::packed_terms::StoredTerms;
use super::postings::{PostingList, PostingLists, Postings, RestoredList};
use super::search::LexicalSource;
use super::text::TextSegment;
use super::{StoredChunk, StoredDocument, VectorState};
use crate::core::entangled::EntangledHVec;

pub(crate) const MAGIC: &[u8; 8] = b"HMSCKP\x00\x01";
const HEADER_WORDS: usize = 40;
pub(crate) const HEADER_LEN: usize = 8 + 8 * HEADER_WORDS;
/// Dictionary entries per block: one binary search over blocks, then at most this many
/// entries decoded per term looked up.
const BLOCK: usize = 32;

const H_CRC: usize = 0;
const H_GENERATION: usize = 1;
const H_DIMENSIONS: usize = 2;
const H_SLOTS: usize = 8;
const H_CHUNKS: usize = 9;
const H_WORDS: usize = 10;
const H_DOCS: usize = 11;
const H_TERMS: usize = 12; // and 13
const H_SECTIONS: usize = 14;

#[derive(Clone, Copy)]
enum Section {
    Dict0,
    Blocks0,
    Dict1,
    Blocks1,
    Postings,
    Slots,
    IdOffsets,
    Ids,
    DocOffsets,
    Docs,
}
const SECTIONS: usize = 10;
const _: () = assert!(H_SECTIONS + 2 * SECTIONS <= HEADER_WORDS);

const SLOT_BYTES: usize = 12;
const FREE: u32 = u32::MAX;

const TEXT_NONE: u8 = 0;
const TEXT_INLINE: u8 = 2;
/// Chunk flags: the chunk id is stored (it is not the default id for its index); the chunk's
/// terms are stemmed (its postings are in form 1).
const OWN_ID: u8 = 1;
const STEMMED: u8 = 2;

fn word(bytes: &[u8], index: usize) -> u64 {
    let at = 8 + 8 * index;
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

/// The generation-name CRC32 and the vector dimensions a checkpoint header records (`header`
/// holds at least the magic and the first three header words).
pub(crate) fn binding(header: &[u8]) -> (u32, u64) {
    (
        word(header, H_GENERATION) as u32,
        word(header, H_DIMENSIONS),
    )
}

/// The CRC a checkpoint header records: over everything after the CRC word.
pub(crate) fn recorded_crc(header: &[u8]) -> u32 {
    word(header, H_CRC) as u32
}

/// Offset the checkpoint CRC starts at.
pub(crate) const CRC_START: usize = 16;

/// Whether every chunk of `store` is lexical-only without word positions, the only chunks a
/// checkpoint holds.
pub(crate) fn representable(store: &DocumentStore) -> bool {
    store.nonpositional_chunks == store.chunk_count
        && store.documents.values().all(|document| {
            document.chunks.iter().all(|chunk| {
                chunk.vector_state == VectorState::Lexical
                    && chunk.embedding.is_none()
                    && chunk.vector.indices.is_empty()
                    && chunk.terms.positions.is_none()
            })
        })
}

/// Encodes `store` as the checkpoint of the generation whose name has CRC32 `generation`;
/// `None` when a chunk is not lexical-only or an offset would not fit the `u32` fields.
pub(crate) fn encode(
    store: &DocumentStore,
    generation: u32,
    dimensions: usize,
) -> Result<Option<Vec<u8>>> {
    if !representable(store) {
        return Ok(None);
    }
    let mut words = [0u64; HEADER_WORDS];
    let mut sections: Vec<Vec<u8>> = vec![Vec::new(); SECTIONS];
    let mut postings = Vec::new();
    for form in 0..2 {
        let (terms, offsets, lists) = store.postings[form].sorted_terms();
        let (mut dict, mut blocks) = (Vec::new(), Vec::new());
        let mut count = 0u64;
        for (i, list) in lists.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let term = &terms[offsets[i] as usize..offsets[i + 1] as usize];
            let prefix = if (count as usize).is_multiple_of(BLOCK) {
                let (Ok(at), Ok(posting)) =
                    (u32::try_from(dict.len()), u32::try_from(postings.len()))
                else {
                    return Ok(None);
                };
                blocks.extend_from_slice(&at.to_le_bytes());
                blocks.extend_from_slice(&posting.to_le_bytes());
                0
            } else {
                let previous = previous_term(&terms, &offsets, &lists, i);
                previous
                    .iter()
                    .zip(term)
                    .take_while(|(a, b)| a == b)
                    .count()
            };
            put_varint(&mut dict, prefix as u64);
            put_bytes(&mut dict, &term[prefix..]);
            let start = postings.len();
            super::postings::pack_checkpoint(list, &mut postings);
            put_varint(&mut dict, list.len() as u64);
            put_varint(&mut dict, (postings.len() - start) as u64);
            count += 1;
        }
        words[H_TERMS + form] = count;
        let (d, b) = if form == 0 {
            (Section::Dict0, Section::Blocks0)
        } else {
            (Section::Dict1, Section::Blocks1)
        };
        sections[d as usize] = dict;
        sections[b as usize] = blocks;
    }
    if u32::try_from(postings.len()).is_err() {
        return Ok(None);
    }
    sections[Section::Postings as usize] = postings;

    let segment = store.text_segment();
    let mut slots = Vec::with_capacity(store.slot_count * SLOT_BYTES);
    let mut owners = vec![(FREE, 0u32); store.slot_count];
    let (mut ids, mut id_offsets) = (Vec::new(), Vec::new());
    let (mut docs, mut doc_offsets) = (Vec::new(), Vec::new());
    for (ordinal, (id, document)) in store.documents.iter().enumerate() {
        let ordinal = u32::try_from(ordinal).context("checkpoint document count")?;
        let document_slots = store.slots.get(id).context("document without slots")?;
        ensure!(
            document_slots.len() == document.chunks.len(),
            "document slots and chunks disagree"
        );
        for (chunk, &slot) in document_slots.iter().enumerate() {
            owners[slot] = (ordinal, u32::try_from(chunk).context("chunk index")?);
        }
        let (Ok(id_at), Ok(doc_at)) = (u32::try_from(ids.len()), u32::try_from(docs.len())) else {
            return Ok(None);
        };
        id_offsets.extend_from_slice(&id_at.to_le_bytes());
        ids.extend_from_slice(id.as_bytes());
        doc_offsets.extend_from_slice(&doc_at.to_le_bytes());
        encode_document(&mut docs, document, segment.as_deref())?;
    }
    let (Ok(id_end), Ok(doc_end)) = (u32::try_from(ids.len()), u32::try_from(docs.len())) else {
        return Ok(None);
    };
    id_offsets.extend_from_slice(&id_end.to_le_bytes());
    doc_offsets.extend_from_slice(&doc_end.to_le_bytes());
    for (slot, &(ordinal, chunk)) in owners.iter().enumerate() {
        slots.extend_from_slice(&DocumentStore::slot_word_count(store, slot).to_le_bytes());
        slots.extend_from_slice(&ordinal.to_le_bytes());
        slots.extend_from_slice(&chunk.to_le_bytes());
    }
    sections[Section::Slots as usize] = slots;
    sections[Section::IdOffsets as usize] = id_offsets;
    sections[Section::Ids as usize] = ids;
    sections[Section::DocOffsets as usize] = doc_offsets;
    sections[Section::Docs as usize] = docs;

    words[H_GENERATION] = u64::from(generation);
    words[H_DIMENSIONS] = dimensions as u64;
    words[H_SLOTS] = store.slot_count as u64;
    words[H_CHUNKS] = store.chunk_count as u64;
    words[H_WORDS] = store.word_count as u64;
    words[H_DOCS] = store.documents.len() as u64;
    let mut at = HEADER_LEN as u64;
    for (i, section) in sections.iter().enumerate() {
        words[H_SECTIONS + 2 * i] = at;
        words[H_SECTIONS + 2 * i + 1] = section.len() as u64;
        at += section.len() as u64;
    }
    let mut out = Vec::with_capacity(at as usize);
    out.extend_from_slice(MAGIC);
    for value in words {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for section in &sections {
        out.extend_from_slice(section);
    }
    let crc = crc32fast::hash(&out[CRC_START..]);
    out[8..16].copy_from_slice(&u64::from(crc).to_le_bytes());
    Ok(Some(out))
}

/// The term before `lists[i]` among those with postings (front coding skips empty lists).
fn previous_term<'a>(
    terms: &'a [u8],
    offsets: &[u32],
    lists: &[&PostingList],
    i: usize,
) -> &'a [u8] {
    let j = (0..i)
        .rev()
        .find(|&j| !lists[j].is_empty())
        .expect("a block's later entry follows a written one");
    &terms[offsets[j] as usize..offsets[j + 1] as usize]
}

fn encode_document(
    out: &mut Vec<u8>,
    document: &StoredDocument,
    segment: Option<&TextSegment>,
) -> Result<()> {
    put_bytes(out, document.version.as_bytes());
    match &document.source_uri {
        Some(uri) => {
            out.push(1);
            put_bytes(out, uri.as_bytes());
        }
        None => out.push(0),
    }
    put_bytes(out, &serde_json::to_vec(&document.metadata)?);
    put_varint(out, document.chunks.len() as u64);
    for (index, chunk) in document.chunks.iter().enumerate() {
        let own_id = chunk.id != super::chunk_id(&document.id, index);
        out.push(if own_id { OWN_ID } else { 0 } | if chunk.stemmed { STEMMED } else { 0 });
        if own_id {
            put_bytes(out, chunk.id.as_bytes());
        }
        put_varint(out, u64::from(chunk.start_byte));
        put_varint(out, u64::from(chunk.end_byte));
        match super::index::text_of(chunk, segment) {
            Some(text) => {
                out.push(TEXT_INLINE);
                put_bytes(out, text.as_bytes());
            }
            None => out.push(TEXT_NONE),
        }
    }
    Ok(())
}

/// Bytes a checkpoint is read from: its file mapped, or (encrypted stores) decrypted in memory.
pub(crate) enum Backing {
    Mapped(memmap2::Mmap),
    #[cfg_attr(not(feature = "security"), allow(dead_code))]
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Owned(bytes) => bytes,
        }
    }
}

/// One posting form's dictionary.
#[derive(Clone, Copy, Default)]
struct Dictionary {
    dict: (usize, usize),
    blocks: (usize, usize),
    terms: usize,
}

/// A validated checkpoint. The CRC was checked by whoever produced `bytes`; construction
/// checks every section, dictionary entry, slot and offset it will later read without checks.
pub(crate) struct ServingImage {
    bytes: Backing,
    identity: StoreIdentity,
    slots: usize,
    chunks: usize,
    words: usize,
    docs: usize,
    dimensions: usize,
    dictionaries: [Dictionary; 2],
    sections: [(usize, usize); SECTIONS],
}

/// One decoded chunk of a checkpoint record.
struct RecordChunk<'a> {
    id: Option<&'a str>,
    stemmed: bool,
    start_byte: u32,
    end_byte: u32,
    text: Option<&'a str>,
}

/// One decoded checkpoint record.
struct Record<'a> {
    version: &'a str,
    source_uri: Option<&'a str>,
    metadata: serde_json::Value,
    chunks: Vec<RecordChunk<'a>>,
}

impl ServingImage {
    pub(crate) fn new(bytes: Backing) -> Result<Self> {
        ensure!(
            bytes.len() >= HEADER_LEN && &bytes[..8] == MAGIC,
            "not a document checkpoint of this version"
        );
        let count = |index: usize| usize::try_from(word(&bytes, index)).ok();
        let mut sections = [(0usize, 0usize); SECTIONS];
        let mut end = HEADER_LEN;
        for (i, section) in sections.iter_mut().enumerate() {
            let (at, len) = (count(H_SECTIONS + 2 * i), count(H_SECTIONS + 2 * i + 1));
            let (Some(at), Some(len)) = (at, len) else {
                bail!("checkpoint section out of range");
            };
            ensure!(
                at == end && at.checked_add(len).is_some_and(|e| e <= bytes.len()),
                "checkpoint section out of range"
            );
            *section = (at, len);
            end = at + len;
        }
        ensure!(end == bytes.len(), "checkpoint has trailing bytes");
        let (Some(slots), Some(chunks), Some(words), Some(docs), Some(dimensions)) = (
            count(H_SLOTS),
            count(H_CHUNKS),
            count(H_WORDS),
            count(H_DOCS),
            count(H_DIMENSIONS),
        ) else {
            bail!("checkpoint counts out of range");
        };
        let mut image = Self {
            bytes,
            identity: StoreIdentity::default(),
            slots,
            chunks,
            words,
            docs,
            dimensions,
            dictionaries: [Dictionary::default(); 2],
            sections,
        };
        image.check_documents()?;
        image.check_slots()?;
        for form in 0..2 {
            image.dictionaries[form] = image.check_dictionary(form)?;
        }
        Ok(image)
    }

    fn section(&self, section: Section) -> &[u8] {
        let (at, len) = self.sections[section as usize];
        &self.bytes[at..at + len]
    }

    fn check_documents(&self) -> Result<()> {
        for (offsets, data) in [
            (Section::IdOffsets, Section::Ids),
            (Section::DocOffsets, Section::Docs),
        ] {
            let (offsets, data) = (self.section(offsets), self.section(data));
            ensure!(
                offsets.len() == (self.docs + 1) * 4,
                "checkpoint document table size"
            );
            let mut previous = 0u32;
            for i in 0..=self.docs {
                let at = u32_at(offsets, 4 * i);
                ensure!(
                    at >= previous && at as usize <= data.len() && (i > 0 || at == 0),
                    "checkpoint document offsets"
                );
                previous = at;
            }
            ensure!(
                previous as usize == data.len(),
                "checkpoint document offsets"
            );
        }
        let ids = std::str::from_utf8(self.section(Section::Ids)).context("checkpoint ids")?;
        let mut previous: Option<&str> = None;
        for i in 0..self.docs {
            let (start, end) = self.span(Section::IdOffsets, i);
            ensure!(
                ids.is_char_boundary(start) && ids.is_char_boundary(end),
                "checkpoint id boundary"
            );
            let id = &ids[start..end];
            ensure!(
                previous.is_none_or(|p| p < id),
                "checkpoint ids out of order"
            );
            previous = Some(id);
        }
        Ok(())
    }

    fn check_slots(&self) -> Result<()> {
        let table = self.section(Section::Slots);
        ensure!(
            table.len() == self.slots * SLOT_BYTES,
            "checkpoint slot table size"
        );
        let mut used = 0usize;
        for slot in 0..self.slots {
            let ordinal = u32_at(table, slot * SLOT_BYTES + 4);
            ensure!(
                ordinal == FREE || (ordinal as usize) < self.docs,
                "checkpoint slot owner"
            );
            used += usize::from(ordinal != FREE);
        }
        ensure!(
            used == self.chunks && self.chunks <= self.slots,
            "checkpoint chunk count"
        );
        Ok(())
    }

    /// Checks form `form`'s dictionary: blocks in order, entries well formed with terms in
    /// strictly ascending byte order, posting ranges inside the posting section.
    fn check_dictionary(&self, form: usize) -> Result<Dictionary> {
        let (dict_section, block_section) = if form == 0 {
            (Section::Dict0, Section::Blocks0)
        } else {
            (Section::Dict1, Section::Blocks1)
        };
        let terms = usize::try_from(word(&self.bytes, H_TERMS + form))?;
        let (dict, blocks) = (self.section(dict_section), self.section(block_section));
        let block_count = terms.div_ceil(BLOCK);
        ensure!(
            blocks.len() == block_count * 8,
            "checkpoint block table size"
        );
        let postings = self.section(Section::Postings).len();
        let mut previous = Vec::new();
        let mut reader = Reader::new(dict);
        let mut posting_at = None::<usize>;
        for i in 0..terms {
            let position = reader.position();
            if i.is_multiple_of(BLOCK) {
                let block = i / BLOCK;
                ensure!(
                    u32_at(blocks, block * 8) as usize == position,
                    "checkpoint block offset"
                );
                let start = u32_at(blocks, block * 8 + 4) as usize;
                ensure!(
                    posting_at.is_none_or(|at| at == start),
                    "checkpoint block postings"
                );
                posting_at = Some(start);
            }
            let prefix = usize::try_from(reader.varint()?)?;
            let suffix = reader.bytes()?;
            ensure!(
                prefix <= previous.len() && (!i.is_multiple_of(BLOCK) || prefix == 0),
                "checkpoint term prefix"
            );
            let mut term = previous[..prefix].to_vec();
            term.extend_from_slice(suffix);
            ensure!(
                std::str::from_utf8(&term).is_ok() && (i == 0 || previous < term),
                "checkpoint terms out of order"
            );
            let count = reader.varint()?;
            let len = usize::try_from(reader.varint()?)?;
            ensure!(
                count > 0 && count <= len as u64 && count <= u64::from(u32::MAX),
                "checkpoint posting count"
            );
            let at = posting_at.expect("set at the first entry");
            ensure!(
                at.checked_add(len).is_some_and(|end| end <= postings),
                "checkpoint posting range"
            );
            posting_at = Some(at + len);
            previous = term;
        }
        ensure!(
            reader.is_empty(),
            "checkpoint dictionary has trailing bytes"
        );
        Ok(Dictionary {
            dict: self.sections[dict_section as usize],
            blocks: self.sections[block_section as usize],
            terms,
        })
    }

    fn span(&self, offsets: Section, i: usize) -> (usize, usize) {
        let table = self.section(offsets);
        (
            u32_at(table, 4 * i) as usize,
            u32_at(table, 4 * i + 4) as usize,
        )
    }

    pub(super) fn id(&self, ordinal: usize) -> &str {
        let (start, end) = self.span(Section::IdOffsets, ordinal);
        let ids = self.section(Section::Ids);
        std::str::from_utf8(&ids[start..end]).expect("ids checked at open")
    }

    pub(super) fn slot_total(&self) -> usize {
        self.slots
    }

    pub(super) fn document_total(&self) -> usize {
        self.docs
    }

    /// The ordinal of document `id` (ids are stored in ascending order).
    pub(super) fn ordinal(&self, id: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.docs);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.id(mid).cmp(id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The ordinal of the document owning `slot`, `None` for a free slot.
    pub(super) fn slot_owner(&self, slot: usize) -> Option<usize> {
        let (_, ordinal, _) = self.slot(slot);
        (ordinal != FREE).then_some(ordinal as usize)
    }

    fn slot(&self, slot: usize) -> (u32, u32, u32) {
        let table = self.section(Section::Slots);
        let at = slot * SLOT_BYTES;
        (
            u32_at(table, at),
            u32_at(table, at + 4),
            u32_at(table, at + 8),
        )
    }

    /// Decodes document `ordinal`'s record.
    fn record(&self, ordinal: usize) -> Result<Record<'_>> {
        let (start, end) = self.span(Section::DocOffsets, ordinal);
        let mut reader = Reader::new(&self.section(Section::Docs)[start..end]);
        let version = reader.str()?;
        let source_uri = match reader.byte()? {
            0 => None,
            1 => Some(reader.str()?),
            other => bail!("checkpoint source flag {other}"),
        };
        let mut metadata: serde_json::Value =
            serde_json::from_slice(reader.bytes()?).context("checkpoint metadata")?;
        super::compact_metadata(&mut metadata);
        let count = reader.count(4)?;
        let mut chunks = Vec::with_capacity(count);
        for _ in 0..count {
            let flags = reader.byte()?;
            ensure!(
                flags & !(OWN_ID | STEMMED) == 0,
                "checkpoint chunk flags {flags}"
            );
            let id = if flags & OWN_ID != 0 {
                Some(reader.str()?)
            } else {
                None
            };
            let start_byte = u32::try_from(reader.varint()?)?;
            let end_byte = u32::try_from(reader.varint()?)?;
            let text = match reader.byte()? {
                TEXT_NONE => None,
                TEXT_INLINE => Some(reader.str()?),
                other => bail!("checkpoint text kind {other}"),
            };
            chunks.push(RecordChunk {
                id,
                stemmed: flags & STEMMED != 0,
                start_byte,
                end_byte,
                text,
            });
        }
        ensure!(reader.is_empty(), "checkpoint record has trailing bytes");
        Ok(Record {
            version,
            source_uri,
            metadata,
            chunks,
        })
    }

    /// A stored chunk as the live store holds a lexical-only one; text inline.
    fn stored_chunk(
        &self,
        document: &str,
        index: usize,
        chunk: &RecordChunk<'_>,
        text: Option<String>,
        terms: StoredTerms,
        word_count: usize,
    ) -> StoredChunk {
        StoredChunk {
            id: chunk
                .id
                .map_or_else(|| super::chunk_id(document, index), str::to_owned),
            text,
            text_at: None,
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
            vector: EntangledHVec::from_indices(Vec::new(), self.dimensions),
            embedding: None,
            terms,
            word_count,
            stemmed: chunk.stemmed,
            vector_state: VectorState::Lexical,
        }
    }

    /// The document and chunk index in `slot`, decoded for a result: the fields a result
    /// reports, with only that chunk's text.
    pub(crate) fn chunk(&self, slot: usize) -> Result<(Arc<StoredDocument>, usize)> {
        let (word_count, ordinal, index) = self.slot(slot);
        ensure!(ordinal != FREE, "checkpoint slot is free");
        let (ordinal, index) = (ordinal as usize, index as usize);
        let id = self.id(ordinal);
        let record = self.record(ordinal)?;
        ensure!(index < record.chunks.len(), "checkpoint chunk index");
        let chunks = record
            .chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let text = (i == index)
                    .then(|| chunk.text.map(str::to_owned))
                    .flatten();
                let words = if i == index { word_count as usize } else { 0 };
                self.stored_chunk(id, i, chunk, text, StoredTerms::default(), words)
            })
            .collect();
        Ok((
            Arc::new(StoredDocument {
                id: id.to_owned(),
                source_uri: record.source_uri.map(str::to_owned),
                version: record.version.to_owned(),
                metadata: record.metadata,
                chunks,
            }),
            index,
        ))
    }

    /// The live store this checkpoint holds, chunk text moved to `text` when given: the same
    /// documents, slots (free ones included), posting bytes and counts, with each chunk's terms
    /// read back from the postings.
    pub(crate) fn hydrate(&self, text: Option<Arc<TextSegment>>) -> Result<DocumentStore> {
        // Owner of every occupied slot, and each document's slots in chunk order.
        let mut document_slots: Vec<Vec<usize>> = vec![Vec::new(); self.docs];
        let mut form_of = vec![u8::MAX; self.slots];
        for slot in 0..self.slots {
            let (_, ordinal, index) = self.slot(slot);
            if ordinal == FREE {
                continue;
            }
            let slots = &mut document_slots[ordinal as usize];
            ensure!(
                index as usize == slots.len(),
                "checkpoint chunk slots out of order"
            );
            slots.push(slot);
        }
        let records = (0..self.docs)
            .map(|ordinal| self.record(ordinal))
            .collect::<Result<Vec<_>>>()?;
        for (record, slots) in records.iter().zip(&document_slots) {
            ensure!(
                record.chunks.len() == slots.len(),
                "checkpoint record and slots disagree"
            );
            for (chunk, &slot) in record.chunks.iter().zip(slots) {
                form_of[slot] = u8::from(chunk.stemmed);
            }
        }
        // Each slot's `(term, frequency)` pairs, in term order per form (a chunk's terms are
        // all in its own form), read from one pass over the postings.
        let mut forward: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.slots];
        let mut lists: [Vec<RestoredList>; 2] = [Vec::new(), Vec::new()];
        for (form, out) in lists.iter_mut().enumerate() {
            let dictionary = self.dictionaries[form];
            let dict = &self.bytes[dictionary.dict.0..dictionary.dict.0 + dictionary.dict.1];
            let postings = self.section(Section::Postings);
            let mut reader = Reader::new(dict);
            let mut term = Vec::new();
            let mut at = if dictionary.terms == 0 {
                0
            } else {
                let blocks = &self.bytes[dictionary.blocks.0..dictionary.blocks.0 + 8];
                u32_at(blocks, 4) as usize
            };
            out.reserve_exact(dictionary.terms);
            for index in 0..dictionary.terms {
                let prefix = usize::try_from(reader.varint()?)?;
                term.truncate(prefix);
                term.extend_from_slice(reader.bytes()?);
                let count = u32::try_from(reader.varint()?)?;
                let len = usize::try_from(reader.varint()?)?;
                let bytes = &postings[at..at + len];
                at += len;
                let mut last = 0usize;
                let mut seen = 0u32;
                let mut failure = None;
                let mut live = Vec::with_capacity(count as usize);
                Postings { bytes, count }.try_for_each(|slot, frequency| {
                    let fits = slot < self.slots
                        && form_of[slot] == form as u8
                        && (seen == 0 || slot > last);
                    if !fits {
                        failure = Some(slot);
                        return false;
                    }
                    forward[slot].push((index as u32, frequency));
                    live.push((slot, frequency));
                    last = slot;
                    seen += 1;
                    true
                })?;
                ensure!(
                    failure.is_none() && seen == count,
                    "checkpoint posting list for a slot it cannot hold"
                );
                let term = String::from_utf8(term.clone()).context("checkpoint term")?;
                out.push((term, live));
            }
        }
        let terms: [Vec<&str>; 2] = [
            lists[0].iter().map(|(t, ..)| t.as_str()).collect(),
            lists[1].iter().map(|(t, ..)| t.as_str()).collect(),
        ];
        let mut documents = Vec::with_capacity(self.docs);
        for (ordinal, (record, slots)) in records.iter().zip(&document_slots).enumerate() {
            let id = self.id(ordinal);
            let mut chunks = Vec::with_capacity(slots.len());
            for (index, (chunk, &slot)) in record.chunks.iter().zip(slots).enumerate() {
                let pairs = std::mem::take(&mut forward[slot]);
                let names = &terms[usize::from(chunk.stemmed)];
                let bytes = pairs.iter().map(|&(t, _)| names[t as usize].len()).sum();
                let stored = StoredTerms::from_sorted_refs(
                    pairs.len(),
                    bytes,
                    pairs.iter().map(|&(t, f)| (names[t as usize], f)),
                );
                let mut stored_chunk = self.stored_chunk(
                    id,
                    index,
                    chunk,
                    chunk.text.map(str::to_owned),
                    stored,
                    self.slot(slot).0 as usize,
                );
                if let Some(segment) = &text {
                    if let Some(inline) = stored_chunk.text.take() {
                        stored_chunk.text_at = Some(segment.append(&inline)?);
                    }
                }
                chunks.push(stored_chunk);
            }
            documents.push((
                Arc::new(StoredDocument {
                    id: id.to_owned(),
                    source_uri: record.source_uri.map(str::to_owned),
                    version: record.version.to_owned(),
                    metadata: record.metadata.clone(),
                    chunks,
                }),
                slots.clone(),
            ));
        }
        drop(terms);
        let words: Vec<u32> = (0..self.slots).map(|slot| self.slot(slot).0).collect();
        let [zero, one] = lists;
        let postings = [
            PostingLists::from_sorted(zero, &words),
            PostingLists::from_sorted(one, &words),
        ];
        let store = DocumentStore::restore(documents, self.slots, postings, text);
        ensure!(
            (store.chunk_count, store.word_count) == (self.chunks, self.words),
            "checkpoint counts disagree with its slots"
        );
        Ok(store)
    }
}

impl LexicalSource for ServingImage {
    type List<'a> = Postings<'a>;

    fn slot_count(&self) -> usize {
        self.slots
    }

    fn chunk_count(&self) -> usize {
        self.chunks
    }

    fn word_count(&self) -> usize {
        self.words
    }

    fn state(&self) -> (StoreIdentity, u64) {
        (self.identity, 0)
    }

    fn postings(&self, form: usize, term: &str) -> Option<Postings<'_>> {
        let dictionary = &self.dictionaries[form];
        if dictionary.terms == 0 {
            return None;
        }
        let dict = &self.bytes[dictionary.dict.0..dictionary.dict.0 + dictionary.dict.1];
        let blocks = &self.bytes[dictionary.blocks.0..dictionary.blocks.0 + dictionary.blocks.1];
        let first_term = |block: usize| -> &[u8] {
            let mut reader = Reader::new(&dict[u32_at(blocks, block * 8) as usize..]);
            reader.varint().expect("dictionary checked at open");
            reader.bytes().expect("dictionary checked at open")
        };
        let target = term.as_bytes();
        // The last block whose first term is at or below the target.
        let (mut lo, mut hi) = (0usize, blocks.len() / 8);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if first_term(mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let block = lo;
        let entries = (dictionary.terms - block * BLOCK).min(BLOCK);
        let mut reader = Reader::new(&dict[u32_at(blocks, block * 8) as usize..]);
        let postings = self.section(Section::Postings);
        let mut at = u32_at(blocks, block * 8 + 4) as usize;
        let mut current: Vec<u8> = Vec::with_capacity(target.len() + 8);
        for _ in 0..entries {
            let prefix = reader.varint().expect("dictionary checked at open") as usize;
            let suffix = reader.bytes().expect("dictionary checked at open");
            current.truncate(prefix);
            current.extend_from_slice(suffix);
            let count = reader.varint().expect("dictionary checked at open") as u32;
            let len = reader.varint().expect("dictionary checked at open") as usize;
            match current.as_slice().cmp(target) {
                std::cmp::Ordering::Equal => {
                    return Some(Postings {
                        bytes: &postings[at..at + len],
                        count,
                    })
                }
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => at += len,
            }
        }
        None
    }

    fn slot_word_count(&self, slot: usize) -> u32 {
        self.slot(slot).0
    }

    fn slot_order(&self, slot: usize) -> (&str, usize) {
        let (_, ordinal, index) = self.slot(slot);
        if ordinal == FREE {
            return ("", index as usize);
        }
        (self.id(ordinal as usize), index as usize)
    }
}
