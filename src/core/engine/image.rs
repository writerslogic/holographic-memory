// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Journal checkpoints (`documents.ckpt`, `docs/file-format/CHECKPOINT.md`). A compaction of a
//! documents-only store writes the document store as one file inside the new journal
//! generation instead of document frames, so the generation is the checkpoint plus the frames
//! appended after it. An engine opening a generation with a checkpoint and no frames maps it
//! and answers unfiltered lexical queries from it. Document writes in every encode mode,
//! deferred encodings and deletes go to a store over the checkpoint that holds the documents
//! written since and marks the checkpoint documents they replace or delete dead
//! (`documents::overlay`); frames after the checkpoint are replayed the same way at open, which
//! rebuilds that dead set. A checkpoint holds no vector state, so chunk-vector queries and
//! deferred encoding read the store over it as they are. The first access of any other kind,
//! and any other write, rebuilds the live store from the checkpoint and the store over it
//! ([`HmsCore::documents`]).

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::core::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use anyhow::{bail, ensure, Context, Result};

use super::HmsCore;
use crate::core::documents::{
    self, Backing, CheckpointBase, DocumentStore, ServingImage, TextSegment,
};

pub(super) const FILE: &str = "documents.ckpt";
/// An encrypted checkpoint: this magic, the binding words in the clear (generation-name CRC32
/// and vector dimensions, at the plain header's word positions), then the checkpoint sealed
/// with the store key over those 40 bytes as associated data.
const SEALED_MAGIC: &[u8; 8] = b"HMSCKPE\x01";
const SEALED_HEADER: usize = 40;

/// How [`HmsCore::open_checkpoint`] left the document store.
pub(super) enum Opened {
    /// No checkpoint: the caller attaches a text segment and replays the journal.
    Absent,
    /// Served from the mapped checkpoint; the generation holds no frames.
    Served,
    /// Served from the mapped checkpoint; the caller replays the frames after it.
    Overlaid,
}

#[derive(Default)]
pub(super) struct ImageState {
    /// The checkpoint the store is served from while it is not rebuilt (the store holds it
    /// too; this is the lock-free test).
    serving: RwLock<Option<Arc<ServingImage>>>,
    hydrating: Mutex<()>,
    /// Whether a mutation other than a document write, encoding or delete was applied: a
    /// checkpoint holds document state only.
    pub(super) other_state: AtomicBool,
    /// Set once the engine opened completely; only then may closing write a checkpoint.
    opened: AtomicBool,
}

fn word(header: &[u8], index: usize) -> u64 {
    let at = 8 + 8 * index;
    u64::from_le_bytes(header[at..at + 8].try_into().expect("eight bytes"))
}

impl HmsCore {
    /// The document store, rebuilt from the checkpoint first if it is being served from one.
    /// The open checked the checkpoint whole, so the rebuild fails only on an I/O error;
    /// accessors that cannot return an error stop there (writes return it).
    pub(super) fn documents(&self) -> RwLockReadGuard<'_, DocumentStore> {
        self.hydrate_documents()
            .expect("rebuilding the document store from its checkpoint");
        self.documents.read()
    }

    /// [`Self::documents`] for writing.
    pub(super) fn documents_mut(&self) -> RwLockWriteGuard<'_, DocumentStore> {
        self.hydrate_documents()
            .expect("rebuilding the document store from its checkpoint");
        self.documents.write()
    }

    /// Runs `capture` on the store while it is served from a checkpoint, between transactions
    /// and excluding a rebuild meanwhile; `None` otherwise or when `capture` declines.
    pub(super) fn capture_from_checkpoint<T>(
        &self,
        capture: impl FnOnce(&DocumentStore) -> Option<T>,
    ) -> Option<T> {
        self.image.serving.read().as_ref()?;
        let _transaction = self.mutation_gate.read();
        let store = self.documents.read();
        if !store.has_base() {
            return None;
        }
        capture(&store)
    }

    /// Whether every mutation of a batch can be applied to a store served from a checkpoint
    /// without rebuilding it: document writes in any encode mode, deferred encodings and
    /// document deletes. A checkpoint holds no vector state (`documents::checkpoint_representable`
    /// and no vector shards), so the chunk vectors, embeddings and pending encodings of the
    /// documents written since are the store's whole vector side, and the store over the
    /// checkpoint keeps them as any store does.
    pub(super) fn checkpoint_accepts(&self, mutations: &[super::Mutation]) -> bool {
        self.image.serving.read().is_some()
            && mutations.iter().all(|mutation| {
                matches!(
                    mutation,
                    super::Mutation::Document(_)
                        | super::Mutation::DeleteDocument { .. }
                        | super::Mutation::EncodeDocument { .. }
                )
            })
    }

    /// Whether this store may hold checkpoints: schema 4 and later (an older release refuses
    /// the store instead of opening it without its checkpointed documents).
    fn checkpoint_schema(&self) -> bool {
        self.schema >= crate::core::format::CHECKPOINT_SCHEMA_VERSION
    }

    fn fresh_text_segment(&self) -> Result<Arc<TextSegment>> {
        Ok(Arc::new(TextSegment::create(
            &self.storage_path.join("documents.text"),
            #[cfg(feature = "security")]
            self.encryption.clone(),
        )?))
    }

    /// Builds the live store from the checkpoint being served and the store over it (the
    /// checkpoint's live documents, then the ones written since) and stops serving it. The
    /// result holds the same documents, so a writer that installs before or after it reaches
    /// the same state.
    pub(super) fn hydrate_documents(&self) -> Result<()> {
        if self.image.serving.read().is_none() {
            return Ok(());
        }
        let _hydrating = self.image.hydrating.lock();
        if self.image.serving.read().is_none() {
            return Ok(());
        }
        let mut documents = self.documents.write();
        let (image, dead, written) = documents
            .overlay()
            .context("store served from a checkpoint without it")?;
        // The dead documents' postings are left out of the rebuilt lists rather than removed
        // from them one document at a time (each removal re-encodes every list it touches).
        let mut store = image.hydrate_excluding(Some(self.fresh_text_segment()?), &dead)?;
        for id in &dead {
            store.remove_unindexed_scrubbed(id)?;
        }
        store.insert_batch(written);
        *documents = store;
        *self.image.serving.write() = None;
        Ok(())
    }

    /// Opens the checkpoint of the live generation, if it has one (see [`Opened`]). A
    /// checkpoint is the store's data, so one that fails its checks fails the open.
    pub(super) fn open_checkpoint(&self) -> Result<Opened> {
        let path = self.arena.data_dir().join(FILE);
        if path.symlink_metadata().is_err() {
            return Ok(Opened::Absent);
        }
        ensure!(
            self.checkpoint_schema(),
            "journal generation holds a checkpoint but the store schema predates checkpoints"
        );
        let image = Arc::new(
            self.load_checkpoint(&path)
                .with_context(|| format!("document checkpoint {}", path.display()))?,
        );
        *self.documents.write() = DocumentStore::over(CheckpointBase::new(Arc::clone(&image)));
        *self.image.serving.write() = Some(image);
        Ok(if self.arena.stats().used_bytes == 0 {
            Opened::Served
        } else {
            Opened::Overlaid
        })
    }

    fn generation_crc(name: &str) -> u32 {
        crc32fast::hash(name.as_bytes())
    }

    fn load_checkpoint(&self, path: &Path) -> Result<ServingImage> {
        let mut file = std::fs::File::open(path)?;
        let mut outer = [0u8; SEALED_HEADER];
        file.read_exact(&mut outer)?;
        let generation = Self::generation_crc(&self.arena.checkpoint().0);
        ensure!(
            (word(&outer, 1) as u32, word(&outer, 2)) == (generation, self.dimensions as u64),
            "checkpoint belongs to another generation or dimension count"
        );
        #[cfg(feature = "security")]
        let sealed_key = self.encryption.clone();
        #[cfg(not(feature = "security"))]
        let sealed_key: Option<()> = None;
        let bytes = if &outer[..8] == SEALED_MAGIC {
            #[cfg(feature = "security")]
            {
                let Some(cipher) = &sealed_key else {
                    bail!("sealed checkpoint in a store opened without encryption");
                };
                let mut sealed = Vec::new();
                file.read_to_end(&mut sealed)?;
                Backing::Owned(cipher.decrypt_with_aad(&sealed, &outer)?)
            }
            #[cfg(not(feature = "security"))]
            bail!("sealed checkpoint needs the security feature");
        } else {
            ensure!(
                &outer[..8] == documents::CHECKPOINT_MAGIC,
                "not a document checkpoint of this version"
            );
            ensure!(
                sealed_key.is_none(),
                "plaintext checkpoint in an encrypted store"
            );
            let mut hasher = crc32fast::Hasher::new();
            hasher.update(&outer[documents::CHECKPOINT_CRC_START..]);
            let mut buffer = vec![0u8; 1 << 16];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
            }
            file.seek(SeekFrom::Start(0))?;
            let mut header = vec![0u8; documents::CHECKPOINT_HEADER_LEN];
            file.read_exact(&mut header)?;
            ensure!(
                hasher.finalize() == documents::checkpoint_crc(&header),
                "checkpoint checksum mismatch"
            );
            // SAFETY: the store lock keeps other engines out of the store directory, and no
            // engine writes a published checkpoint: compactions write new ones in new
            // generations and remove an old generation only after the switch.
            Backing::Mapped(unsafe { memmap2::Mmap::map(&file)? })
        };
        ensure!(
            bytes.len() >= documents::CHECKPOINT_HEADER_LEN
                && documents::checkpoint_crc(&bytes)
                    == crc32fast::hash(&bytes[documents::CHECKPOINT_CRC_START..])
                && documents::checkpoint_binding(&bytes) == (generation, self.dimensions as u64),
            "checkpoint checksum mismatch"
        );
        ServingImage::new(bytes)
    }

    /// The sealed checkpoint of `store` for the generation directory `generation`, when this
    /// store takes checkpoints and holds only lexical-only documents; called under the
    /// maintenance lease of a compaction snapshot, which checked the vector shards are empty
    /// (encoded chunks also fill them, and a checkpoint does not hold them).
    pub(super) fn encode_checkpoint(
        &self,
        store: &DocumentStore,
        generation: &str,
    ) -> Result<Option<Vec<u8>>> {
        if !self.checkpoint_schema()
            || !self.config.document_checkpoint.enabled
            || self.image.other_state.load(Ordering::Acquire)
        {
            return Ok(None);
        }
        let Some(image) =
            documents::encode_checkpoint(store, Self::generation_crc(generation), self.dimensions)?
        else {
            return Ok(None);
        };
        self.seal_checkpoint(image).map(Some)
    }

    /// Whether a compaction now would write a checkpoint: checkpoints enabled, only document
    /// state, no vector shards, and every chunk lexical-only.
    pub(super) fn takes_checkpoint(&self) -> bool {
        if !self.checkpoint_schema()
            || !self.config.document_checkpoint.enabled
            || self.image.other_state.load(Ordering::Acquire)
        {
            return false;
        }
        // One lock at a time: commits take the shards before the store.
        self.shards.read().count() == 0
            && documents::checkpoint_representable(&self.documents.read())
    }

    /// Writes a checkpoint when it is due: the engine fully opened, a compaction would write
    /// one ([`Self::takes_checkpoint`]), and at least `document_checkpoint.min_journal_bytes` of
    /// frames in the live generation (the frames a checkpoint would fold in).
    pub(super) fn checkpoint_if_due(&self) -> Result<()> {
        if !self.image.opened.load(Ordering::Acquire) {
            return Ok(());
        }
        let used = self.arena.stats().used_bytes;
        if used == 0 || used < self.config.document_checkpoint.min_journal_bytes {
            return Ok(());
        }
        if !self.takes_checkpoint() {
            return Ok(());
        }
        self.compact()
    }

    #[cfg(feature = "security")]
    fn seal_checkpoint(&self, mut image: Vec<u8>) -> Result<Vec<u8>> {
        use zeroize::Zeroize;
        let Some(cipher) = &self.encryption else {
            return Ok(image);
        };
        let mut outer = image[..SEALED_HEADER].to_vec();
        outer[..8].copy_from_slice(SEALED_MAGIC);
        outer[8..16].fill(0);
        let sealed = cipher.encrypt_with_aad(&image, &outer);
        image.zeroize();
        outer.extend_from_slice(&sealed?);
        Ok(outer)
    }

    #[cfg(not(feature = "security"))]
    fn seal_checkpoint(&self, image: Vec<u8>) -> Result<Vec<u8>> {
        Ok(image)
    }

    /// Whether unfiltered lexical queries are being answered from a mapped checkpoint (the
    /// store opened from one and nothing has needed the rebuilt store since).
    #[doc(hidden)]
    pub fn serves_from_checkpoint(&self) -> bool {
        self.image.serving.read().is_some()
    }

    pub(super) fn mark_opened(&self) {
        self.image.opened.store(true, Ordering::Release);
    }
}
