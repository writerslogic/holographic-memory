// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Counter-backed bundle with exact deletion, and the thresholded-only bundle
//! that models today's mechanism.
//!
//! # Mechanism
//!
//! The engine's sparse hypervector (`EntangledHVec`, `src/core/entangled.rs`)
//! is a sorted list of `k ~ dim/256` active indices. The engine bundles such
//! vectors by thresholding a per-index sum: `bundle` keeps indices whose count
//! reaches the majority threshold, `bundle_bloom` keeps every index that occurs
//! at least once (threshold 1, the union). Once the sum has been thresholded to
//! a bit, the per-item contribution is gone and no later operation can remove
//! one item without disturbing the others.
//!
//! [`CounterBundle`] keeps the sum itself: one integer count per dimension,
//! never thresholded in storage. Insert adds 1 at each of the item's indices,
//! delete subtracts 1. Because integer addition is a group operation, the
//! counts after `insert(x); delete(x)` are bit-for-bit the counts before, and
//! the state after any sequence of operations equals the fold of the surviving
//! multiset in any order. The binary view used for recall, `{ i : count[i] >=
//! tau }`, is derived from the counts on demand; at `tau = 1` it is exactly
//! `bundle_bloom` of the surviving items.
//!
//! # Data layout
//!
//! * `counts: Vec<C>` with `C` = `i16` or `i32` (the [`Counter`] trait), one
//!   entry per dimension; `dim * 2` or `dim * 4` bytes, independent of the
//!   number of stored items.
//! * `ledger: HashMap<u64, u32>`: multiplicity of each inserted item keyed by
//!   the first 8 bytes of its SHA-256 item digest. It lets `delete` refuse an
//!   item that was never inserted, which is what keeps the "state == fold of
//!   the inserted multiset" invariant true even against a buggy caller.
//!   12 payload bytes per distinct live item.
//! * Scalars: `view_size` (number of indices at or above `tau`), `live_bits`
//!   (sum of all counts), `items` (live item count), and monotone operation
//!   counters for audit.
//!
//! # Invariants
//!
//! * Every operation is atomic: validation (dimension, index range, overflow,
//!   underflow, capacity policy) completes before the first count is touched,
//!   so a failed call leaves the state unchanged.
//! * `counts[i] >= 0` for all `i`; a count can only reach zero by deleting the
//!   items that raised it. An underflow is reported as corruption, never
//!   masked by saturation.
//! * `view_size == |{ i : counts[i] >= tau }|` and `live_bits == sum(counts)`
//!   are maintained incrementally and are the basis of the capacity policy.
//! * The state digest is SHA-256 over a domain tag, `dim`, `tau`, the counter
//!   width, the live item count and the little-endian counts. Two bundles with
//!   equal digests have equal counts (up to SHA-256 collision resistance).
//!
//! [`ThresholdedBundle`] is the baseline: the binary union only. Its `delete`
//! has the two options a bit-only representation allows: leave the bits (the
//! deleted item stays fully recognisable) or clear the item's bits (which also
//! clears every surviving item's bit at those positions).

use std::collections::HashMap;
use std::fmt;

use holographic_memory::EntangledHVec;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Integer counter type for one dimension of a [`CounterBundle`].
pub trait Counter: Copy + Default + Eq + fmt::Debug + Send + Sync + 'static {
    /// Width in bits; part of the state digest and of the memory accounting.
    const BITS: u32;
    /// Name reported in JSON (`"i16"`, `"i32"`).
    const NAME: &'static str;
    /// Largest representable count.
    const MAX: i64;
    /// `self + 1`, or `None` on overflow.
    fn checked_inc(self) -> Option<Self>;
    /// `self - 1`, or `None` when the count is already zero.
    fn checked_dec(self) -> Option<Self>;
    /// Widened value.
    fn get(self) -> i64;
    /// Little-endian encoding appended to `out` (used by the state digest).
    fn write_le(self, out: &mut Vec<u8>);
}

impl Counter for i16 {
    const BITS: u32 = 16;
    const NAME: &'static str = "i16";
    const MAX: i64 = i16::MAX as i64;
    fn checked_inc(self) -> Option<Self> {
        self.checked_add(1)
    }
    fn checked_dec(self) -> Option<Self> {
        if self <= 0 {
            None
        } else {
            Some(self - 1)
        }
    }
    fn get(self) -> i64 {
        i64::from(self)
    }
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Counter for i32 {
    const BITS: u32 = 32;
    const NAME: &'static str = "i32";
    const MAX: i64 = i32::MAX as i64;
    fn checked_inc(self) -> Option<Self> {
        self.checked_add(1)
    }
    fn checked_dec(self) -> Option<Self> {
        if self <= 0 {
            None
        } else {
            Some(self - 1)
        }
    }
    fn get(self) -> i64 {
        i64::from(self)
    }
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

/// Failure modes of bundle mutation. Every variant leaves the bundle unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum BundleError {
    /// `dim == 0` or `tau == 0`.
    InvalidParameter { what: &'static str },
    /// An item index is `>= dim`.
    IndexOutOfRange { index: u32, dim: usize },
    /// Item indices are not strictly increasing.
    UnsortedIndices,
    /// A count would exceed `Counter::MAX`.
    CounterOverflow { index: u32, max: i64 },
    /// A count would go below zero: the ledger and the counts disagree.
    CounterUnderflow { index: u32 },
    /// The item (by digest) has no live insertion to subtract.
    NotPresent { item: String },
    /// `CapacityPolicy::max_items` reached.
    ItemLimit { max_items: usize },
    /// The insert would push the view density above `CapacityPolicy::max_view_density`.
    DensityLimit { density: f64, max_view_density: f64 },
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { what } => write!(f, "invalid parameter: {what}"),
            Self::IndexOutOfRange { index, dim } => {
                write!(f, "index {index} out of range for dim {dim}")
            }
            Self::UnsortedIndices => write!(f, "item indices are not strictly increasing"),
            Self::CounterOverflow { index, max } => {
                write!(f, "count at index {index} would exceed {max}")
            }
            Self::CounterUnderflow { index } => {
                write!(f, "count at index {index} would go negative (state corrupt)")
            }
            Self::NotPresent { item } => write!(f, "item {item} has no live insertion"),
            Self::ItemLimit { max_items } => write!(f, "item limit {max_items} reached"),
            Self::DensityLimit {
                density,
                max_view_density,
            } => write!(
                f,
                "view density {density:.4} would exceed the limit {max_view_density:.4}"
            ),
        }
    }
}

impl std::error::Error for BundleError {}

/// Bounded-capacity policy checked before every insert.
///
/// `max_items` bounds the live item count. `max_view_density` bounds
/// `view_size / dim` after the insert; at `tau = 1` a density of 0.5 is the
/// point where a random item already has half of its bits present in the view,
/// so recall on the thresholded view is weak beyond it. Counter overflow is
/// always refused regardless of policy.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CapacityPolicy {
    pub max_items: usize,
    pub max_view_density: f64,
}

impl CapacityPolicy {
    /// No item or density bound; only counter overflow is refused.
    pub const UNBOUNDED: Self = Self {
        max_items: usize::MAX,
        max_view_density: 1.0,
    };
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

/// A SHA-256 digest, serialised as 64 lowercase hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest32(pub [u8; 32]);

impl Digest32 {
    /// Lowercase hex.
    pub fn hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Parse 64 hex characters.
    pub fn parse_hex(s: &str) -> Option<Self> {
        if s.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            let hi = (chunk[0] as char).to_digit(16)?;
            let lo = (chunk[1] as char).to_digit(16)?;
            out[i] = ((hi << 4) | lo) as u8;
        }
        Some(Self(out))
    }

    /// First 8 bytes as a little-endian `u64`; the ledger key.
    pub fn key64(&self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.0[..8]);
        u64::from_le_bytes(b)
    }
}

impl fmt::Debug for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl Serialize for Digest32 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.hex())
    }
}

impl<'de> Deserialize<'de> for Digest32 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse_hex(&s).ok_or_else(|| serde::de::Error::custom("expected 64 hex characters"))
    }
}

fn finish(h: Sha256) -> Digest32 {
    let arr: [u8; 32] = h.finalize().into();
    Digest32(arr)
}

const ITEM_TAG: &[u8] = b"hms-forget/item-v1";
const STATE_TAG: &[u8] = b"hms-forget/state-v1";

/// Digest of an item: domain tag, `dim`, index count and the sorted indices.
pub fn item_digest(dim: usize, indices: &[u32]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(ITEM_TAG);
    h.update((dim as u64).to_le_bytes());
    h.update((indices.len() as u64).to_le_bytes());
    for &i in indices {
        h.update(i.to_le_bytes());
    }
    finish(h)
}

/// Digest of the engine hypervector (its sorted active indices).
pub fn hvec_digest(dim: usize, item: &EntangledHVec) -> Digest32 {
    item_digest(dim, item.indices())
}

fn validate(dim: usize, item: &EntangledHVec) -> Result<&[u32], BundleError> {
    let idx = item.indices();
    let mut prev: Option<u32> = None;
    for &i in idx {
        if i as usize >= dim {
            return Err(BundleError::IndexOutOfRange { index: i, dim });
        }
        if let Some(p) = prev {
            if i <= p {
                return Err(BundleError::UnsortedIndices);
            }
        }
        prev = Some(i);
    }
    Ok(idx)
}

/// Byte-level memory accounting of one bundle state.
#[derive(Clone, Debug, Serialize)]
pub struct MemoryAccounting {
    pub dim: usize,
    pub items: usize,
    pub active_per_item: usize,
    pub counter_bits: u32,
    /// `dim * counter_bits / 8`; independent of the item count.
    pub counter_bytes: usize,
    pub ledger_entries: usize,
    /// `12 * ledger_entries` (u64 key + u32 multiplicity), hash-map overhead excluded.
    pub ledger_payload_bytes: usize,
    pub view_size: usize,
    pub view_density: f64,
    /// Dense bitset of the view, `ceil(dim / 8)`.
    pub view_bitset_bytes: usize,
    /// Sparse index list of the view, `4 * view_size`.
    pub view_sparse_bytes: usize,
    pub bytes_per_item_counter: f64,
    pub bytes_per_item_total: f64,
    /// What the per-document shard store holds for the same items: `4 * k` per item
    /// (sorted `u32` indices; ids and map overhead excluded).
    pub per_document_store_bytes: usize,
    pub per_document_bytes_per_item: usize,
    /// Item count above which the counter bundle is smaller than per-document storage.
    pub breakeven_items: usize,
}

/// Counter-backed bundle; see the module documentation.
#[derive(Clone, Debug)]
pub struct CounterBundle<C: Counter> {
    dim: usize,
    tau: u32,
    counts: Vec<C>,
    view_size: usize,
    live_bits: u64,
    items: usize,
    inserted_total: u64,
    deleted_total: u64,
    ledger: HashMap<u64, u32>,
    policy: CapacityPolicy,
}

impl<C: Counter> CounterBundle<C> {
    /// Empty bundle. `tau` is the view threshold (1 = union, matching `bundle_bloom`).
    pub fn new(dim: usize, tau: u32, policy: CapacityPolicy) -> Result<Self, BundleError> {
        if dim == 0 {
            return Err(BundleError::InvalidParameter { what: "dim must be > 0" });
        }
        if tau == 0 {
            return Err(BundleError::InvalidParameter { what: "tau must be >= 1" });
        }
        Ok(Self {
            dim,
            tau,
            counts: vec![C::default(); dim],
            view_size: 0,
            live_bits: 0,
            items: 0,
            inserted_total: 0,
            deleted_total: 0,
            ledger: HashMap::new(),
            policy,
        })
    }

    /// Fold of `items` into a fresh bundle; the reference for the exactness check.
    pub fn rebuild<'a, I>(
        dim: usize,
        tau: u32,
        policy: CapacityPolicy,
        items: I,
    ) -> Result<Self, BundleError>
    where
        I: IntoIterator<Item = &'a EntangledHVec>,
    {
        let mut b = Self::new(dim, tau, policy)?;
        for item in items {
            b.insert(item)?;
        }
        Ok(b)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn tau(&self) -> u32 {
        self.tau
    }
    pub fn items(&self) -> usize {
        self.items
    }
    pub fn inserted_total(&self) -> u64 {
        self.inserted_total
    }
    pub fn deleted_total(&self) -> u64 {
        self.deleted_total
    }
    pub fn view_size(&self) -> usize {
        self.view_size
    }
    pub fn view_density(&self) -> f64 {
        self.view_size as f64 / self.dim as f64
    }
    /// Mean count per dimension, `sum(counts) / dim`; the Poisson rate of the null.
    pub fn lambda(&self) -> f64 {
        self.live_bits as f64 / self.dim as f64
    }
    pub fn live_bits(&self) -> u64 {
        self.live_bits
    }
    pub fn counts(&self) -> &[C] {
        &self.counts
    }
    pub fn policy(&self) -> CapacityPolicy {
        self.policy
    }
    /// Live multiplicity of an item digest.
    pub fn multiplicity(&self, item: &Digest32) -> u32 {
        self.ledger.get(&item.key64()).copied().unwrap_or(0)
    }
    pub fn ledger_entries(&self) -> usize {
        self.ledger.len()
    }

    /// Add one copy of `item`. Returns its digest.
    pub fn insert(&mut self, item: &EntangledHVec) -> Result<Digest32, BundleError> {
        let idx = validate(self.dim, item)?;
        if self.items >= self.policy.max_items {
            return Err(BundleError::ItemLimit {
                max_items: self.policy.max_items,
            });
        }
        let tau = i64::from(self.tau);
        let mut crossing = 0usize;
        for &i in idx {
            let c = self.counts[i as usize];
            if c.checked_inc().is_none() {
                return Err(BundleError::CounterOverflow {
                    index: i,
                    max: C::MAX,
                });
            }
            if c.get() + 1 == tau {
                crossing += 1;
            }
        }
        let new_view = self.view_size + crossing;
        let density = new_view as f64 / self.dim as f64;
        if density > self.policy.max_view_density {
            return Err(BundleError::DensityLimit {
                density,
                max_view_density: self.policy.max_view_density,
            });
        }
        for &i in idx {
            let slot = &mut self.counts[i as usize];
            *slot = slot
                .checked_inc()
                .expect("overflow was excluded by the validation pass");
        }
        self.view_size = new_view;
        self.live_bits += idx.len() as u64;
        self.items += 1;
        self.inserted_total += 1;
        let d = item_digest(self.dim, idx);
        *self.ledger.entry(d.key64()).or_insert(0) += 1;
        Ok(d)
    }

    /// Subtract one copy of `item`. Exact inverse of [`Self::insert`].
    pub fn delete(&mut self, item: &EntangledHVec) -> Result<Digest32, BundleError> {
        let idx = validate(self.dim, item)?;
        let d = item_digest(self.dim, idx);
        let key = d.key64();
        if self.ledger.get(&key).copied().unwrap_or(0) == 0 {
            return Err(BundleError::NotPresent { item: d.hex() });
        }
        let tau = i64::from(self.tau);
        let mut leaving = 0usize;
        for &i in idx {
            let c = self.counts[i as usize];
            if c.checked_dec().is_none() {
                return Err(BundleError::CounterUnderflow { index: i });
            }
            if c.get() == tau {
                leaving += 1;
            }
        }
        for &i in idx {
            let slot = &mut self.counts[i as usize];
            *slot = slot
                .checked_dec()
                .expect("underflow was excluded by the validation pass");
        }
        self.view_size -= leaving;
        self.live_bits -= idx.len() as u64;
        self.items -= 1;
        self.deleted_total += 1;
        let remaining = match self.ledger.get_mut(&key) {
            Some(m) => {
                *m -= 1;
                *m
            }
            None => 0,
        };
        if remaining == 0 {
            self.ledger.remove(&key);
        }
        Ok(d)
    }

    /// Number of the item's indices whose count reaches `tau` (overlap with the view).
    pub fn view_overlap(&self, item: &EntangledHVec) -> usize {
        let tau = i64::from(self.tau);
        item.indices()
            .iter()
            .filter(|&&i| {
                self.counts
                    .get(i as usize)
                    .map(|c| c.get() >= tau)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Unthresholded readout: `sum(counts[i])` over the item's indices.
    pub fn count_dot(&self, item: &EntangledHVec) -> i64 {
        item.indices()
            .iter()
            .map(|&i| self.counts.get(i as usize).map(|c| c.get()).unwrap_or(0))
            .sum()
    }

    /// `(sum(counts), sum(counts^2))` over all dimensions.
    pub fn count_moments(&self) -> (f64, f64) {
        let mut s = 0.0f64;
        let mut s2 = 0.0f64;
        for c in &self.counts {
            let v = c.get() as f64;
            s += v;
            s2 += v * v;
        }
        (s, s2)
    }

    /// Sorted indices with `count >= tau`.
    pub fn view_indices(&self) -> Vec<u32> {
        let tau = i64::from(self.tau);
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.get() >= tau)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// The view as an engine hypervector.
    pub fn view_hvec(&self) -> EntangledHVec {
        EntangledHVec::from_indices(self.view_indices(), self.dim)
    }

    /// Digest of the view indices.
    pub fn view_digest(&self) -> Digest32 {
        item_digest(self.dim, &self.view_indices())
    }

    fn state_header(&self, items: usize) -> Sha256 {
        let mut h = Sha256::new();
        h.update(STATE_TAG);
        h.update((self.dim as u64).to_le_bytes());
        h.update(self.tau.to_le_bytes());
        h.update(C::BITS.to_le_bytes());
        h.update((items as u64).to_le_bytes());
        h
    }

    /// SHA-256 commitment to the full counter state.
    pub fn state_digest(&self) -> Digest32 {
        let mut h = self.state_header(self.items);
        let mut buf = Vec::with_capacity(self.dim * (C::BITS as usize / 8));
        for c in &self.counts {
            c.write_le(&mut buf);
        }
        h.update(&buf);
        finish(h)
    }

    /// Digest of the state that `insert(item)` would produce, computed without
    /// mutating or cloning the counts. Used by certificate verification to
    /// reproduce `state_before` from `state_after` and the deleted item.
    pub fn digest_with_item_added(&self, item: &EntangledHVec) -> Result<Digest32, BundleError> {
        let idx = validate(self.dim, item)?;
        let mut h = self.state_header(self.items + 1);
        let mut buf = Vec::with_capacity(self.dim * (C::BITS as usize / 8));
        let mut p = 0usize;
        for (i, c) in self.counts.iter().enumerate() {
            let v = if p < idx.len() && idx[p] as usize == i {
                p += 1;
                c.checked_inc().ok_or(BundleError::CounterOverflow {
                    index: i as u32,
                    max: C::MAX,
                })?
            } else {
                *c
            };
            v.write_le(&mut buf);
        }
        h.update(&buf);
        Ok(finish(h))
    }

    /// Bit-for-bit comparison of counts and parameters.
    pub fn counts_equal(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.tau == other.tau
            && self.items == other.items
            && self.live_bits == other.live_bits
            && self.view_size == other.view_size
            && self.counts == other.counts
    }

    /// Memory accounting at the current state; `active_per_item` is `k`.
    pub fn accounting(&self, active_per_item: usize) -> MemoryAccounting {
        let counter_bytes = self.dim * (C::BITS as usize / 8);
        let ledger_entries = self.ledger.len();
        let ledger_payload_bytes = ledger_entries * 12;
        let per_document_bytes_per_item = 4 * active_per_item;
        let per_document_store_bytes = self.items * per_document_bytes_per_item;
        let items_f = self.items.max(1) as f64;
        MemoryAccounting {
            dim: self.dim,
            items: self.items,
            active_per_item,
            counter_bits: C::BITS,
            counter_bytes,
            ledger_entries,
            ledger_payload_bytes,
            view_size: self.view_size,
            view_density: self.view_density(),
            view_bitset_bytes: self.dim.div_ceil(8),
            view_sparse_bytes: 4 * self.view_size,
            bytes_per_item_counter: counter_bytes as f64 / items_f,
            bytes_per_item_total: (counter_bytes + ledger_payload_bytes) as f64 / items_f,
            per_document_store_bytes,
            per_document_bytes_per_item,
            breakeven_items: counter_bytes.div_ceil(per_document_bytes_per_item.max(1)),
        }
    }
}

/// How a bit-only bundle can respond to a delete request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineDelete {
    /// Leave the bits: the deleted item remains fully present in the view.
    Noop,
    /// Clear the item's bits: survivors sharing those positions lose them too.
    Clear,
}

/// Today's mechanism: the thresholded (union, `tau = 1`) bundle as a bitset.
#[derive(Clone, Debug)]
pub struct ThresholdedBundle {
    dim: usize,
    bits: Vec<u64>,
    set_count: usize,
    items: usize,
}

impl ThresholdedBundle {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            bits: vec![0u64; dim.div_ceil(64)],
            set_count: 0,
            items: 0,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn items(&self) -> usize {
        self.items
    }
    pub fn set_count(&self) -> usize {
        self.set_count
    }
    pub fn density(&self) -> f64 {
        self.set_count as f64 / self.dim as f64
    }

    #[inline]
    fn is_set(&self, i: u32) -> bool {
        let w = (i / 64) as usize;
        self.bits.get(w).is_some_and(|word| word & (1u64 << (i % 64)) != 0)
    }

    /// OR the item into the view.
    pub fn insert(&mut self, item: &EntangledHVec) -> Result<(), BundleError> {
        let idx = validate(self.dim, item)?;
        for &i in idx {
            let w = (i / 64) as usize;
            let m = 1u64 << (i % 64);
            if self.bits[w] & m == 0 {
                self.bits[w] |= m;
                self.set_count += 1;
            }
        }
        self.items += 1;
        Ok(())
    }

    /// Apply the chosen delete semantics. Returns the number of bits cleared.
    pub fn delete(&mut self, item: &EntangledHVec, mode: BaselineDelete) -> Result<usize, BundleError> {
        let idx = validate(self.dim, item)?;
        let mut cleared = 0usize;
        if mode == BaselineDelete::Clear {
            for &i in idx {
                let w = (i / 64) as usize;
                let m = 1u64 << (i % 64);
                if self.bits[w] & m != 0 {
                    self.bits[w] &= !m;
                    self.set_count -= 1;
                    cleared += 1;
                }
            }
        }
        self.items = self.items.saturating_sub(1);
        Ok(cleared)
    }

    /// Number of the item's indices set in the view.
    pub fn overlap(&self, item: &EntangledHVec) -> usize {
        item.indices().iter().filter(|&&i| self.is_set(i)).count()
    }

    /// Sorted set indices.
    pub fn indices(&self) -> Vec<u32> {
        (0..self.dim as u32).filter(|&i| self.is_set(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(dim: usize, seed: u64) -> EntangledHVec {
        EntangledHVec::new_deterministic(dim, seed)
    }

    #[test]
    fn insert_then_delete_restores_the_empty_state() {
        let dim = 4096;
        let mut b = CounterBundle::<i16>::new(dim, 1, CapacityPolicy::UNBOUNDED).unwrap();
        let empty = b.state_digest();
        let items: Vec<_> = (0..50).map(|s| item(dim, s)).collect();
        for it in &items {
            b.insert(it).unwrap();
        }
        assert_eq!(b.items(), 50);
        for it in &items {
            b.delete(it).unwrap();
        }
        assert_eq!(b.items(), 0);
        assert_eq!(b.view_size(), 0);
        assert_eq!(b.live_bits(), 0);
        assert_eq!(b.ledger_entries(), 0);
        assert_eq!(b.state_digest(), empty);
        assert!(b.counts().iter().all(|&c| c == 0));
    }

    #[test]
    fn delete_matches_rebuild_bit_for_bit() {
        let dim = 4096;
        let items: Vec<_> = (0..300).map(|s| item(dim, 1000 + s)).collect();
        let mut b = CounterBundle::<i32>::new(dim, 2, CapacityPolicy::UNBOUNDED).unwrap();
        for it in &items {
            b.insert(it).unwrap();
        }
        let before = b.state_digest();
        for it in items.iter().step_by(3) {
            b.delete(it).unwrap();
        }
        let survivors: Vec<_> = items
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 != 0)
            .map(|(_, v)| v)
            .collect();
        let r = CounterBundle::<i32>::rebuild(dim, 2, CapacityPolicy::UNBOUNDED, survivors.into_iter())
            .unwrap();
        assert!(b.counts_equal(&r));
        assert_eq!(b.state_digest(), r.state_digest());
        assert_eq!(b.view_digest(), r.view_digest());
        assert_ne!(b.state_digest(), before);
    }

    #[test]
    fn view_at_tau_one_equals_bundle_bloom() {
        let dim = 4096;
        let items: Vec<_> = (0..40).map(|s| item(dim, 77 + s)).collect();
        let b = CounterBundle::<i16>::rebuild(dim, 1, CapacityPolicy::UNBOUNDED, items.iter()).unwrap();
        let bloom = EntangledHVec::bundle_bloom(&items);
        assert_eq!(b.view_indices(), bloom.indices());
    }

    #[test]
    fn non_member_delete_is_refused_and_leaves_state_unchanged() {
        let dim = 4096;
        let mut b = CounterBundle::<i16>::new(dim, 1, CapacityPolicy::UNBOUNDED).unwrap();
        b.insert(&item(dim, 1)).unwrap();
        let d = b.state_digest();
        let err = b.delete(&item(dim, 2)).unwrap_err();
        assert!(matches!(err, BundleError::NotPresent { .. }));
        assert_eq!(b.state_digest(), d);
        assert_eq!(b.items(), 1);
    }

    #[test]
    fn overflow_is_refused_atomically() {
        let dim = 512;
        let it = item(dim, 9);
        let mut b = CounterBundle::<i16>::new(dim, 1, CapacityPolicy::UNBOUNDED).unwrap();
        for _ in 0..i16::MAX {
            b.insert(&it).unwrap();
        }
        let d = b.state_digest();
        let err = b.insert(&it).unwrap_err();
        assert!(matches!(err, BundleError::CounterOverflow { .. }));
        assert_eq!(b.state_digest(), d);
        assert_eq!(b.items(), i16::MAX as usize);
    }

    #[test]
    fn capacity_policy_bounds_items_and_density() {
        let dim = 4096;
        let policy = CapacityPolicy {
            max_items: 2,
            max_view_density: 1.0,
        };
        let mut b = CounterBundle::<i16>::new(dim, 1, policy).unwrap();
        b.insert(&item(dim, 1)).unwrap();
        b.insert(&item(dim, 2)).unwrap();
        assert!(matches!(b.insert(&item(dim, 3)), Err(BundleError::ItemLimit { .. })));

        let tight = CapacityPolicy {
            max_items: usize::MAX,
            max_view_density: 0.001,
        };
        let mut t = CounterBundle::<i16>::new(dim, 1, tight).unwrap();
        assert!(matches!(t.insert(&item(dim, 1)), Err(BundleError::DensityLimit { .. })));
        assert_eq!(t.items(), 0);
        assert_eq!(t.view_size(), 0);
    }

    #[test]
    fn digest_with_item_added_equals_digest_after_insert() {
        let dim = 4096;
        let mut b = CounterBundle::<i32>::new(dim, 1, CapacityPolicy::UNBOUNDED).unwrap();
        for s in 0..10 {
            b.insert(&item(dim, s)).unwrap();
        }
        let x = item(dim, 99);
        let predicted = b.digest_with_item_added(&x).unwrap();
        b.insert(&x).unwrap();
        assert_eq!(predicted, b.state_digest());
    }

    #[test]
    fn thresholded_clear_damages_survivors_and_noop_keeps_the_deleted_item() {
        let dim = 4096;
        let items: Vec<_> = (0..200).map(|s| item(dim, 500 + s)).collect();
        let mut noop = ThresholdedBundle::new(dim);
        let mut clear = ThresholdedBundle::new(dim);
        for it in &items {
            noop.insert(it).unwrap();
            clear.insert(it).unwrap();
        }
        let k = items[0].indices().len();
        noop.delete(&items[0], BaselineDelete::Noop).unwrap();
        assert_eq!(noop.overlap(&items[0]), k);
        let cleared = clear.delete(&items[0], BaselineDelete::Clear).unwrap();
        assert_eq!(cleared, k);
        assert_eq!(clear.overlap(&items[0]), 0);
        let lost: usize = items[1..].iter().map(|it| k - clear.overlap(it)).sum();
        assert!(lost > 0, "200 items at k={k} in {dim} dims must share positions");
    }

    #[test]
    fn digest_hex_round_trip() {
        let d = item_digest(16, &[1, 2, 3]);
        assert_eq!(Digest32::parse_hex(&d.hex()), Some(d));
        assert!(Digest32::parse_hex("zz").is_none());
    }
}
