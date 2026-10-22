// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! On-disk form of a [`VGraph`]: every array in one file at a page-aligned offset, so an index
//! opens by mapping the file (or reading it into one buffer) without decoding or copying.
//!
//! Layout, little-endian: the magic, [`HEADER_WORDS`] `u64` header words, one `(offset, bytes)`
//! pair per section in [`SECTIONS`] order, then the sections, each starting on a 4096-byte
//! boundary. Sizes that follow from the header (code widths, strides, block geometry) are
//! recomputed on open rather than trusted, and every section length, neighbour id and upper-layer
//! id is checked before the index is returned: a damaged file is an error, never an
//! out-of-bounds read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use rayon::prelude::*;

use super::{
    bfs_order, code_at, put_varint, quantize, quantize4, zigzag, Graph, GraphView, VertexLayout,
};
use super::{block_record_bytes, get_varint, unzigzag, Line, Lines, Metric, Rotation, VGraph};
use super::{BLOCK, BLOCK_SEED, NONE, NONE3, SCREEN_SEED};

/// Element types an index array may hold.
///
/// # Safety
/// Implementors are plain data: every bit pattern is a valid value and there is no padding, so
/// any aligned, initialized bytes may be viewed as `[Self]` and every `[Self]` as bytes.
pub(super) unsafe trait Plain: Copy + 'static {}
// SAFETY: integers and floats of these widths have no invalid bit patterns and no padding.
unsafe impl Plain for u8 {}
// SAFETY: as above.
unsafe impl Plain for i8 {}
// SAFETY: as above.
unsafe impl Plain for u32 {}
// SAFETY: as above.
unsafe impl Plain for u64 {}
// SAFETY: as above.
unsafe impl Plain for f32 {}
// SAFETY: `Line` is `repr(C)` around `[i8; 128]`: size 128, no padding.
unsafe impl Plain for Line {}

/// Bytes of a plain array.
fn as_bytes<T: Plain>(xs: &[T]) -> &[u8] {
    // SAFETY: `T: Plain` has no padding, so its `len * size_of` bytes are initialized.
    unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs)) }
}

/// The read-only buffer an opened index's arrays point into.
enum Backing {
    Map(memmap2::Mmap),
    /// The file read into cache-line-aligned memory; `.1` is the file length.
    Heap(Vec<Line>, usize),
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Map(m) => m,
            Self::Heap(lines, len) => &as_bytes(lines)[..*len],
        }
    }
}

/// What keeps a [`Slab`]'s elements alive (only ever dropped).
enum Owner<T> {
    Vec { _buf: Vec<T> },
    File { _file: Arc<Backing> },
}

/// One array of a [`VGraph`]: its own vector (a built index) or a range of the shared buffer
/// of an opened file. The element pointer is resolved once, so an access costs what a slice
/// access does whatever holds the bytes.
pub(super) struct Slab<T: Plain> {
    ptr: NonNull<T>,
    len: usize,
    owner: Owner<T>,
}

// SAFETY: a `Slab` is a view of `len` initialized, aligned elements at `ptr` that `owner` keeps
// alive and in place (a vector buffer is never reallocated once wrapped; a backing buffer or
// mapping is never written). Moving or sharing it moves or shares no more than `&[T]` would.
unsafe impl<T: Plain + Send + Sync> Send for Slab<T> {}
// SAFETY: as for `Send`: shared access only reads.
unsafe impl<T: Plain + Sync> Sync for Slab<T> {}

impl<T: Plain> From<Vec<T>> for Slab<T> {
    fn from(mut v: Vec<T>) -> Self {
        let ptr = NonNull::new(v.as_mut_ptr()).expect("a vector pointer is never null");
        Self {
            ptr,
            len: v.len(),
            owner: Owner::Vec { _buf: v },
        }
    }
}

impl<T: Plain> Deref for Slab<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: see the `Send` impl.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Plain> Slab<T> {
    /// The elements of an index being built. Arrays of an opened file are read-only; asking
    /// for one is a logic error.
    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            matches!(self.owner, Owner::Vec { .. }),
            "the arrays of an opened index file are read-only"
        );
        // SAFETY: `ptr` and `len` describe the owned vector's buffer (see `From<Vec<T>>`), and
        // `&mut self` makes this the only access to it.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// `len` elements of `backing` at byte `offset`, which the caller has checked lie inside
    /// the buffer at an offset that is a multiple of [`ALIGN`].
    fn file(backing: &Arc<Backing>, offset: usize, len: usize) -> Self {
        let ptr = if len == 0 {
            NonNull::dangling()
        } else {
            let bytes = &backing.bytes()[offset..offset + len * std::mem::size_of::<T>()];
            let p = bytes.as_ptr().cast::<T>().cast_mut();
            // Buffers start page-aligned (a mapping) or line-aligned (`Vec<Line>`), and section
            // offsets are multiples of 4096, so every element type is aligned.
            assert!(p.is_aligned(), "index section is misaligned");
            NonNull::new(p).expect("a slice pointer is never null")
        };
        Self {
            ptr,
            len,
            owner: Owner::File {
                _file: Arc::clone(backing),
            },
        }
    }
}

/// Where the arrays of an index opened with [`VGraph::open`] live.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Residency {
    /// The file is mapped read-only. `open` reads the neighbour lists and id maps, which it
    /// checks; the codes are read on their first touch. Read pages count as file-backed
    /// resident memory, which the kernel may reclaim under pressure and read again.
    Mapped,
    /// As [`Residency::Mapped`], with every page touched before `open` returns, so no query
    /// pays a page fault.
    #[default]
    Prefaulted,
    /// The file is read into memory the index owns (anonymous memory).
    Heap,
}

const MAGIC: &[u8; 8] = b"HMSVGR01";
/// Written as a header word; reads back differently on a target of the other byte order.
const ORDER_MARK: u64 = 0x0102_0304_0506_0708;
/// Section alignment in the file: a page, so a mapping aligns every section for any element.
const ALIGN: usize = 4096;
const HEADER_WORDS: usize = 19;
/// Section names, in file order (the names appear in errors).
const SECTIONS: [&str; 15] = [
    "half_norms",
    "adj",
    "adj3",
    "adjv",
    "adjv_off",
    "codes",
    "scales",
    "residual",
    "res_scales",
    "ids",
    "screen",
    "blocks",
    "mean",
    "upper_ids",
    "layers",
];
const PREFIX: usize = 8 + 8 * HEADER_WORDS + 16 * SECTIONS.len();

fn bad(message: String) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("index file: {message}"),
    )
}

fn le_words(bytes: &[u8]) -> impl Iterator<Item = u64> + '_ {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
}

/// Upper layers as two `u32` arrays: per layer its vertex count then each list's length, and
/// all lists concatenated.
fn encode_layers(layers: &[Vec<Vec<u32>>]) -> Vec<u32> {
    let mut out = Vec::new();
    for layer in layers {
        out.push(layer.len() as u32);
        out.extend(layer.iter().map(|l| l.len() as u32));
    }
    for layer in layers {
        layer.iter().for_each(|l| out.extend_from_slice(l));
    }
    out
}

fn decode_layers(words: &[u32], count: usize, top: usize) -> std::io::Result<Vec<Vec<Vec<u32>>>> {
    let short = || bad("layers section is truncated".into());
    let mut shapes = Vec::with_capacity(count);
    let mut at = 0;
    for _ in 0..count {
        let size = *words.get(at).ok_or_else(short)? as usize;
        let lens = words.get(at + 1..at + 1 + size).ok_or_else(short)?;
        shapes.push(lens);
        at += 1 + size;
    }
    let mut layers = Vec::with_capacity(count);
    let mut prev = top;
    for lens in shapes {
        // Layer j + 1 is a prefix of layer j, so the descent's local ids stay valid downwards.
        if lens.is_empty() || lens.len() > prev {
            return Err(bad("upper layer sizes do not shrink upwards".into()));
        }
        prev = lens.len();
        let mut layer = Vec::with_capacity(lens.len());
        for &l in lens {
            let list = words.get(at..at + l as usize).ok_or_else(short)?;
            if list.iter().any(|&u| u as usize >= lens.len()) {
                return Err(bad("upper-layer id out of range".into()));
            }
            layer.push(list.to_vec());
            at += l as usize;
        }
        layers.push(layer);
    }
    if at != words.len() {
        return Err(bad("layers section has trailing data".into()));
    }
    Ok(layers)
}

/// `a * b`, or an error naming `what` when it overflows.
fn mul(a: usize, b: usize, what: &str) -> std::io::Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| bad(format!("{what} overflows")))
}

impl VGraph {
    pub(crate) fn release_pages(&self) -> std::io::Result<()> {
        #[cfg(target_os = "linux")]
        if let Owner::File { _file } = &self.codes.lines.owner {
            if let Backing::Map(map) = _file.as_ref() {
                // SAFETY: generation files are immutable and retained while readers exist;
                // shared read-only pages reload exactly the same bytes, including during search.
                unsafe { map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)? };
            }
        }
        Ok(())
    }

    pub(crate) fn save_from_graph(
        g: &Graph,
        path: &Path,
        bits: u8,
        residual_bits: u8,
        align: bool,
        reorder: bool,
        id_bytes: u8,
    ) -> std::io::Result<()> {
        save_graph(
            g.view(),
            path,
            bits,
            residual_bits,
            align,
            reorder,
            id_bytes,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn save_prepared(
        g: &super::super::PreparedGraph,
        path: &Path,
        bits: u8,
        residual_bits: u8,
        align: bool,
        reorder: bool,
        id_bytes: u8,
    ) -> std::io::Result<()> {
        save_graph(
            g.view(),
            path,
            bits,
            residual_bits,
            align,
            reorder,
            id_bytes,
        )
    }

    /// Dimension of the indexed vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric the index orders by.
    pub fn metric(&self) -> Metric {
        if self.half_norms.is_empty() {
            Metric::Angular
        } else {
            Metric::Euclidean
        }
    }

    /// Writes the index for [`VGraph::open`]: to a temporary file next to `path`, synced, then
    /// renamed over it, so a reader never sees a partial file. Little-endian targets only (the
    /// arrays are written as they lie in memory).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if cfg!(target_endian = "big") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "index files are little-endian; this target is big-endian",
            ));
        }
        let id_bytes = if !self.adjv_off.is_empty() {
            0
        } else if !self.adj3.is_empty() {
            3
        } else {
            4
        };
        let header: [u64; HEADER_WORDS] = [
            ORDER_MARK,
            self.n as u64,
            self.dim as u64,
            self.degree as u64,
            u64::from(self.bits),
            u64::from(self.res_bits),
            self.stride as u64,
            id_bytes,
            u64::from(self.entry),
            self.metric().code(),
            self.screen_words as u64,
            u64::from(self.screen_sigma.to_bits()),
            u64::from(self.block_bits),
            self.block_dim as u64,
            self.block_m as u64,
            self.blocks_per_vertex as u64,
            u64::from(self.block_kappa.to_bits()),
            u64::from(!self.ids.is_empty()),
            self.layers.len() as u64,
        ];
        let layers = encode_layers(&self.layers);
        let sections: [&[u8]; SECTIONS.len()] = [
            as_bytes(&self.half_norms),
            as_bytes(&self.adj),
            as_bytes(&self.adj3),
            as_bytes(&self.adjv),
            as_bytes(&self.adjv_off),
            as_bytes(&self.codes.lines),
            as_bytes(&self.scales),
            as_bytes(&self.residual),
            as_bytes(&self.res_scales),
            as_bytes(&self.ids),
            as_bytes(&self.screen),
            as_bytes(&self.blocks.lines),
            as_bytes(&self.mean),
            as_bytes(&self.upper_ids),
            as_bytes(&layers),
        ];
        let mut table = Vec::with_capacity(sections.len());
        let mut at = PREFIX;
        for s in &sections {
            at = at.next_multiple_of(ALIGN);
            table.push((at as u64, s.len() as u64));
            at += s.len();
        }
        let name = path
            .file_name()
            .ok_or_else(|| bad(format!("{} names no file", path.display())))?;
        let mut tmp_name = name.to_os_string();
        tmp_name.push(".tmp");
        let tmp = path.with_file_name(tmp_name);
        let written = (|| {
            let mut w = std::io::BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
            w.write_all(MAGIC)?;
            for x in header {
                w.write_all(&x.to_le_bytes())?;
            }
            for (off, len) in &table {
                w.write_all(&off.to_le_bytes())?;
                w.write_all(&len.to_le_bytes())?;
            }
            let mut pos = PREFIX;
            let zeros = [0u8; ALIGN];
            for (s, (off, _)) in sections.iter().zip(&table) {
                w.write_all(&zeros[..*off as usize - pos])?;
                w.write_all(s)?;
                pos = *off as usize + s.len();
            }
            let file = w.into_inner().map_err(|e| e.into_error())?;
            file.sync_all()?;
            crate::core::durable_file::sync_mapped_file(&file)
        })();
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        std::fs::rename(&tmp, path)?;
        crate::core::durable_file::sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(std::io::Error::other)
    }

    /// Opens an index written by [`VGraph::save`] with its arrays where `residency` puts them.
    /// The search reads the arrays in place, so its results are those of the saved index.
    ///
    /// A mapped index (the default, prefaulted) must not be modified or truncated while it is
    /// open: the mapping would change under the index, and a truncated page faults with
    /// `SIGBUS`. Replace the file by renaming a new one over it, as [`VGraph::save`] does.
    pub fn open(path: &Path, residency: Residency) -> std::io::Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "index files are little-endian; this target is big-endian",
            ));
        }
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| bad("larger than the address space".into()))?;
        if len < PREFIX {
            return Err(bad(format!("{len} bytes is shorter than the header")));
        }
        let backing = Arc::new(match residency {
            Residency::Heap => {
                let count = len.div_ceil(std::mem::size_of::<Line>());
                let mut lines = Vec::with_capacity(count);
                super::huge_capacity(lines.as_ptr(), count);
                lines.resize(count, Line([0; 128]));
                // SAFETY: `Line` is plain bytes (see `Plain`) and `lines` is initialized, so its
                // first `len <= 128 * count` bytes may be written as `u8`.
                let buf =
                    unsafe { std::slice::from_raw_parts_mut(lines.as_mut_ptr().cast::<u8>(), len) };
                file.read_exact(buf)?;
                Backing::Heap(lines, len)
            }
            // SAFETY: the mapping is read-only and lives in the `Backing`; the caller keeps the
            // file unmodified while the index is open (see above).
            Residency::Mapped | Residency::Prefaulted => {
                Backing::Map(unsafe { memmap2::Mmap::map(&file)? })
            }
        });
        let out = Self::from_backing(&backing)?;
        if residency == Residency::Prefaulted {
            let bytes = backing.bytes();
            let mut sum = 0u8;
            for at in (0..bytes.len()).step_by(ALIGN) {
                // SAFETY: `at` is inside `bytes`; a volatile read keeps the touch from being
                // optimized away.
                sum ^= unsafe { std::ptr::read_volatile(bytes.as_ptr().add(at)) };
            }
            std::hint::black_box(sum);
        }
        Ok(out)
    }

    fn from_backing(backing: &Arc<Backing>) -> std::io::Result<Self> {
        let bytes = backing.bytes();
        if &bytes[..8] != MAGIC {
            return Err(bad(
                "bad magic (not an index file, or another format)".into()
            ));
        }
        let h: Vec<u64> = le_words(&bytes[8..8 + 8 * HEADER_WORDS]).collect();
        if h[0] != ORDER_MARK {
            return Err(bad("byte-order mark mismatch".into()));
        }
        let small = |i: usize, max: u64, what: &str| -> std::io::Result<usize> {
            if h[i] > max {
                Err(bad(format!("{what} {} exceeds {max}", h[i])))
            } else {
                Ok(h[i] as usize)
            }
        };
        let n = small(1, u64::from(NONE) - 1, "vertex count")?;
        let dim = small(2, 1 << 20, "dimension")?;
        let degree = small(3, 1 << 16, "degree")?;
        let bits = small(4, 8, "code bits")? as u8;
        let res_bits = small(5, 8, "residual bits")? as u8;
        let stride = small(6, 1 << 24, "row stride")?;
        let id_bytes = small(7, 4, "id width")?;
        let entry = small(8, u64::from(NONE), "entry")?;
        let euclidean = small(9, 1, "metric")? == 1;
        let screen_words = small(10, 1 << 20, "screen words")?;
        let screen_sigma = f32::from_bits(h[11] as u32);
        let block_bits = small(12, 4, "block bits")? as u8;
        let block_dim = small(13, 1 << 22, "block dimension")?;
        let block_m = small(14, 1 << 22, "block groups")?;
        let blocks_per_vertex = small(15, 1 << 16, "blocks per vertex")?;
        let block_kappa = f32::from_bits(h[16] as u32);
        let reordered = small(17, 1, "reorder flag")? == 1;
        let layer_count = small(18, 64, "layer count")?;

        if n == 0 || dim == 0 || degree == 0 || !degree.is_multiple_of(2) || entry >= n {
            return Err(bad(
                "empty index, zero dimension, odd degree or entry out of range".into(),
            ));
        }
        if !matches!(bits, 4 | 8) || !matches!(res_bits, 0 | 4 | 8) {
            return Err(bad(format!("code bits {bits} / residual bits {res_bits}")));
        }
        if !(id_bytes == 4 || id_bytes == 0 || (id_bytes == 3 && (n as u64) < u64::from(NONE3))) {
            return Err(bad(format!("id width {id_bytes} for {n} vertices")));
        }
        // Widths as `from_graph_with` derives them.
        let padded = dim.next_multiple_of(if bits == 4 { 32 } else { 16 });
        let cbytes = if bits == 4 { padded / 2 } else { padded };
        let aligned = if cbytes <= 64 {
            cbytes.next_power_of_two()
        } else {
            cbytes.next_multiple_of(128)
        };
        if stride != cbytes && stride != aligned {
            return Err(bad(format!("row stride {stride} for {cbytes}-byte codes")));
        }
        let res_stride = match res_bits {
            8 => padded,
            4 => padded / 2,
            _ => 0,
        };
        let screen_rot = (screen_words > 0).then(|| Rotation::new(dim, 0, SCREEN_SEED));
        if let Some(rot) = &screen_rot {
            if euclidean || screen_words != rot.padded() / 64 {
                return Err(bad("screen shape does not match the dimension".into()));
            }
        }
        let block_rot = (block_bits > 0).then(|| Rotation::new(dim, 0, BLOCK_SEED));
        let record = match &block_rot {
            Some(rot) => {
                if euclidean
                    || !matches!(block_bits, 1 | 2 | 4)
                    || block_dim != rot.padded()
                    || block_m != block_dim * usize::from(block_bits) / 4
                    || blocks_per_vertex != degree.div_ceil(BLOCK)
                {
                    return Err(bad("neighbour-block shape does not match the graph".into()));
                }
                block_record_bytes(block_m)
            }
            None if block_dim == 0 && block_m == 0 && blocks_per_vertex == 0 => 0,
            None => return Err(bad("block shape without blocks".into())),
        };

        let table: Vec<(usize, usize)> = le_words(&bytes[8 + 8 * HEADER_WORDS..PREFIX])
            .collect::<Vec<_>>()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[o, l]| (o as usize, l as usize))
            .collect();
        let mut end = PREFIX;
        for (name, &(off, len)) in SECTIONS.iter().zip(&table) {
            let fits = off.is_multiple_of(ALIGN)
                && off >= end
                && off.checked_add(len).is_some_and(|e| e <= bytes.len());
            if !fits {
                return Err(bad(format!(
                    "section {name} lies outside the file or overlaps"
                )));
            }
            end = off + len;
        }
        let row_lines = mul(n, stride, "code rows")?.div_ceil(128);
        let block_bytes = mul(mul(n, blocks_per_vertex, "blocks")?, record, "blocks")?;
        let nd = mul(n, degree, "adjacency")?;
        let expect = |s: usize| -> Option<usize> {
            Some(match s {
                0 => 4 * if euclidean { n } else { 0 },
                1 => 4 * if id_bytes == 4 { nd } else { 0 },
                2 => 3 * if id_bytes == 3 { nd } else { 0 },
                3 if id_bytes == 0 => return None,
                3 => 0,
                4 => 4 * if id_bytes == 0 { n + 1 } else { 0 },
                5 => 128 * row_lines,
                6 => 4 * n,
                7 => n.checked_mul(res_stride)?,
                8 => 4 * if res_bits > 0 { n } else { 0 },
                9 => 4 * if reordered { n } else { 0 },
                10 => {
                    8 * if screen_words > 0 {
                        n.checked_mul(screen_words + 1)?
                    } else {
                        0
                    }
                }
                11 => 128 * block_bytes.div_ceil(128),
                12 => 4 * padded,
                _ => return None,
            })
        };
        for (s, (name, &(_, len))) in SECTIONS.iter().zip(&table).enumerate() {
            let element = match s {
                2 | 3 | 7 => 1,
                5 | 11 => 128,
                10 => 8,
                _ => 4,
            };
            let ok = match expect(s) {
                Some(want) => len == want,
                None => len.is_multiple_of(element),
            };
            if !ok {
                return Err(bad(format!(
                    "section {name} has {len} bytes, not the expected size"
                )));
            }
        }
        let sec = |s: usize| table[s];
        let slab = |s: usize, size: usize| (sec(s).0, sec(s).1 / size);
        macro_rules! view {
            ($s:expr, $t:ty) => {{
                let (off, count) = slab($s, std::mem::size_of::<$t>());
                Slab::<$t>::file(backing, off, count)
            }};
        }
        let adj = view!(1, u32);
        let adj3 = view!(2, u8);
        let adjv = view!(3, u8);
        let adjv_off = view!(4, u32);
        let ids = view!(9, u32);
        let upper_ids: Vec<u32> = view!(13, u32).to_vec();
        let layer_words = view!(14, u32);

        // Every id the search follows must name a vertex.
        let in_range = |u: u32, none: u32| (u as usize) < n || u == none;
        let ids_ok = match id_bytes {
            4 => adj.iter().all(|&u| in_range(u, NONE)),
            3 => adj3
                .as_chunks::<3>()
                .0
                .iter()
                .all(|b| in_range(u32::from_le_bytes([b[0], b[1], b[2], 0]), NONE3)),
            _ => delta_lists_ok(&adjv, &adjv_off, n, degree),
        };
        if !ids_ok {
            return Err(bad("neighbour id out of range or malformed list".into()));
        }
        if ids.iter().any(|&u| u as usize >= n) || upper_ids.iter().any(|&u| u as usize >= n) {
            return Err(bad("vertex id out of range".into()));
        }
        if upper_ids.len() > n {
            return Err(bad("more upper-layer vertices than vertices".into()));
        }
        let layers = decode_layers(&layer_words, layer_count, upper_ids.len())?;
        let mean: Vec<f32> = view!(12, f32).to_vec();
        Ok(Self {
            n,
            dim,
            padded,
            bits,
            cbytes,
            stride,
            degree,
            mean,
            half_norms: view!(0, f32),
            adj,
            adj3,
            adjv,
            adjv_off,
            codes: Lines {
                lines: view!(5, Line),
                len: n * stride,
            },
            scales: view!(6, f32),
            res_bits,
            res_stride,
            residual: view!(7, i8),
            res_scales: view!(8, f32),
            entry: entry as u32,
            upper_ids,
            layers,
            ids,
            screen: view!(10, u64),
            screen_words,
            screen_rot,
            screen_sigma,
            block_bits,
            block_dim,
            block_m,
            blocks_per_vertex,
            blocks: Lines {
                lines: view!(11, Line),
                len: block_bytes,
            },
            block_rot,
            block_kappa,
        })
    }
}

fn section_write<T: Plain>(
    w: &mut std::io::BufWriter<&mut File>,
    table: &[(usize, usize)],
    section: usize,
    offset: usize,
    values: &[T],
) -> std::io::Result<()> {
    let bytes = as_bytes(values);
    let (start, len) = table[section];
    if offset.checked_add(bytes.len()).is_none_or(|end| end > len) {
        return Err(bad("section write exceeds computed geometry".into()));
    }
    w.seek(SeekFrom::Start((start + offset) as u64))?;
    w.write_all(bytes)
}

fn save_graph(
    g: GraphView<'_>,
    path: &Path,
    bits: u8,
    residual_bits: u8,
    align: bool,
    reorder: bool,
    id_bytes: u8,
) -> std::io::Result<()> {
    if cfg!(target_endian = "big") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "index files are little-endian",
        ));
    }
    let VertexLayout {
        padded,
        cbytes,
        stride,
        res_stride,
    } = VertexLayout::new(g.n, g.dim, bits, residual_bits, align, id_bytes)
        .map_err(|e| bad(e.to_string()))?;
    let (n, dim, r) = (g.n, g.dim, g.degree);
    let ids = if reorder { bfs_order(g) } else { Vec::new() };
    let old = |v: usize| if reorder { ids[v] as usize } else { v };
    let mut new_ids = vec![0u32; if reorder { n } else { 0 }];
    for (v, &id) in ids.iter().enumerate() {
        new_ids[id as usize] = v as u32;
    }
    let new = |v: u32| if reorder { new_ids[v as usize] } else { v };
    let list = |v: usize| g.neighbors(old(v)).iter().take(r).map(|&v| new(v));
    let capacity = || bad("encoded index size overflows".into());
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(capacity)
    };
    let mut sorted = Vec::new();
    let mut encoded = Vec::new();
    let mut encode_list = |v: usize, dst: &mut Vec<u8>| {
        sorted.clear();
        sorted.extend(list(v));
        sorted.sort_unstable();
        let mut prev = v as u32;
        for (i, &u) in sorted.iter().enumerate() {
            let delta = if i == 0 {
                zigzag(i64::from(u) - i64::from(prev))
            } else {
                u64::from(u - prev - 1)
            };
            put_varint(dst, delta);
            prev = u;
        }
    };
    let mut adjv_len = 0usize;
    if id_bytes == 0 {
        for v in 0..n {
            encoded.clear();
            encode_list(v, &mut encoded);
            adjv_len = adjv_len
                .checked_add(encoded.len())
                .filter(|&n| n <= u32::MAX as usize)
                .ok_or_else(capacity)?;
        }
    }
    let layers_len = g.layers.iter().try_fold(0usize, |size, layer| {
        let words = layer.iter().try_fold(
            1usize.checked_add(layer.len()).ok_or_else(capacity)?,
            |words, list| words.checked_add(list.len()).ok_or_else(capacity),
        )?;
        size.checked_add(mul(words, 4)?).ok_or_else(capacity)
    })?;
    let codes_len = mul(n, stride)?
        .checked_next_multiple_of(128)
        .ok_or_else(capacity)?;
    let lengths = [
        if g.metric == Metric::Euclidean {
            mul(n, 4)?
        } else {
            0
        },
        if id_bytes == 4 {
            mul(mul(n, r)?, 4)?
        } else {
            0
        },
        if id_bytes == 3 {
            mul(mul(n, r)?, 3)?
        } else {
            0
        },
        adjv_len,
        if id_bytes == 0 {
            mul(n.checked_add(1).ok_or_else(capacity)?, 4)?
        } else {
            0
        },
        codes_len,
        mul(n, 4)?,
        mul(n, res_stride)?,
        if residual_bits > 0 { mul(n, 4)? } else { 0 },
        mul(ids.len(), 4)?,
        0,
        0,
        mul(padded, 4)?,
        mul(g.upper_ids.len(), 4)?,
        layers_len,
    ];
    let mut table = Vec::with_capacity(SECTIONS.len());
    let mut end = PREFIX;
    for len in lengths {
        end = end.checked_next_multiple_of(ALIGN).ok_or_else(capacity)?;
        table.push((end, len));
        end = end
            .checked_add(len)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(capacity)?;
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    temp.as_file().set_len(end as u64)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, temp.as_file_mut());
    w.write_all(MAGIC)?;
    for word in [
        ORDER_MARK,
        n as u64,
        dim as u64,
        r as u64,
        u64::from(bits),
        u64::from(residual_bits),
        stride as u64,
        u64::from(id_bytes),
        u64::from(new(g.entry)),
        g.metric.code(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        u64::from(reorder),
        g.layers.len() as u64,
    ] {
        w.write_all(&word.to_le_bytes())?;
    }
    for &(offset, len) in &table {
        w.write_all(&(offset as u64).to_le_bytes())?;
        w.write_all(&(len as u64).to_le_bytes())?;
    }
    let edge_batch = ((1 << 20) / r.saturating_mul(10).max(1)).max(1);
    let mut adjv_at = 0usize;
    for start in (0..n).step_by(edge_batch) {
        let stop = n.min(start + edge_batch);
        if id_bytes == 0 {
            let mut offsets = Vec::with_capacity(stop - start);
            encoded.clear();
            for v in start..stop {
                offsets.push((adjv_at + encoded.len()) as u32);
                encode_list(v, &mut encoded);
            }
            section_write(&mut w, &table, 3, adjv_at, &encoded)?;
            section_write(&mut w, &table, 4, start * 4, &offsets)?;
            adjv_at += encoded.len();
        } else if id_bytes == 3 {
            encoded.clear();
            for v in start..stop {
                let mut count = 0;
                for u in list(v) {
                    encoded.extend_from_slice(&u.min(NONE3).to_le_bytes()[..3]);
                    count += 1;
                }
                for _ in count..r {
                    encoded.extend_from_slice(&NONE3.to_le_bytes()[..3]);
                }
            }
            section_write(&mut w, &table, 2, start * r * 3, &encoded)?;
        } else {
            let mut edges = vec![NONE; (stop - start) * r];
            for (v, row) in (start..stop).zip(edges.chunks_exact_mut(r)) {
                for (dst, u) in row.iter_mut().zip(list(v)) {
                    *dst = u;
                }
            }
            section_write(&mut w, &table, 1, start * r * 4, &edges)?;
        }
    }
    if id_bytes == 0 {
        section_write(&mut w, &table, 4, n * 4, &[adjv_at as u32])?;
    }
    let mut sum = vec![0f64; dim];
    for row in g.vectors.chunks_exact(dim) {
        sum.iter_mut()
            .zip(row)
            .for_each(|(s, &v)| *s += f64::from(v));
    }
    let mut mean: Vec<f32> = sum.iter().map(|s| (s / n as f64) as f32).collect();
    mean.resize(padded, 0.0);
    g.release_rows()?;
    let batch = ((1 << 20) / (stride + res_stride + 16)).max(1);
    for start in (0..n).step_by(batch) {
        let count = batch.min(n - start);
        let mut codes = vec![0i8; count * stride];
        let mut scales = vec![0f32; count];
        let centred = |v: usize| -> Vec<f32> {
            g.vectors[old(v) * dim..(old(v) + 1) * dim]
                .iter()
                .zip(&mean)
                .map(|(a, m)| a - m)
                .collect()
        };
        codes
            .par_chunks_exact_mut(stride)
            .zip(scales.par_iter_mut())
            .enumerate()
            .for_each(|(v, (dst, scale))| {
                let x = centred(start + v);
                *scale = if bits == 8 {
                    quantize(&x, &mut dst[..cbytes])
                } else {
                    quantize4(&x, &mut dst[..cbytes])
                };
            });
        section_write(&mut w, &table, 5, start * stride, &codes)?;
        section_write(&mut w, &table, 6, start * 4, &scales)?;
        if residual_bits > 0 {
            let mut residual = vec![0i8; count * res_stride];
            let mut res_scales = vec![0f32; count];
            residual
                .par_chunks_exact_mut(res_stride)
                .zip(res_scales.par_iter_mut())
                .enumerate()
                .for_each(|(v, (dst, scale))| {
                    let mut x = centred(start + v);
                    let c = &codes[v * stride..v * stride + cbytes];
                    x.iter_mut()
                        .enumerate()
                        .for_each(|(i, a)| *a -= scales[v] * f32::from(code_at(c, bits, i)));
                    *scale = if residual_bits == 8 {
                        quantize(&x, dst)
                    } else {
                        quantize4(&x, dst)
                    };
                });
            section_write(&mut w, &table, 7, start * res_stride, &residual)?;
            section_write(&mut w, &table, 8, start * 4, &res_scales)?;
        }
        if g.metric == Metric::Euclidean {
            let half: Vec<f32> = (start..start + count)
                .map(|v| 0.5 * g.norms[old(v)])
                .collect();
            section_write(&mut w, &table, 0, start * 4, &half)?;
        }
        g.release_rows()?;
    }
    section_write(&mut w, &table, 9, 0, &ids)?;
    section_write(&mut w, &table, 12, 0, &mean)?;
    let upper: Vec<u32> = g.upper_ids.iter().map(|&v| new(v)).collect();
    section_write(&mut w, &table, 13, 0, &upper)?;
    w.seek(SeekFrom::Start(table[14].0 as u64))?;
    for layer in g.layers {
        w.write_all(&(layer.len() as u32).to_le_bytes())?;
        for list in layer {
            w.write_all(&(list.len() as u32).to_le_bytes())?;
        }
    }
    for layer in g.layers {
        for list in layer {
            w.write_all(as_bytes(list))?;
        }
    }
    w.flush()?;
    drop(w);
    temp.as_file().sync_all()?;
    crate::core::durable_file::sync_mapped_file(temp.as_file())?;
    temp.persist(path).map_err(|e| e.error)?;
    crate::core::durable_file::sync_directory(parent).map_err(std::io::Error::other)
}

/// Delta-coded lists (see `VGraph::for_each_delta_neighbour`): offsets rise from 0 to the end
/// of `adjv`, each list decodes inside its span to at most `degree` ids below `n`, with
/// varints of at most ten bytes.
fn delta_lists_ok(adjv: &[u8], off: &[u32], n: usize, degree: usize) -> bool {
    if off.len() != n + 1 || off[0] != 0 || off[n] as usize != adjv.len() {
        return false;
    }
    for v in 0..n {
        let (a, b) = (off[v] as usize, off[v + 1] as usize);
        if a > b {
            return false;
        }
        let span = &adjv[a..b];
        let (mut at, mut prev, mut count) = (0, v as i64, 0);
        while at < span.len() {
            let rest = &span[at..span.len().min(at + 10)];
            if rest.iter().all(|&b| b & 0x80 != 0) {
                return false;
            }
            let (delta, used) = get_varint(rest);
            at += used;
            let u = if count == 0 {
                prev.checked_add(unzigzag(delta))
            } else {
                i64::try_from(delta)
                    .ok()
                    .and_then(|d| prev.checked_add(d + 1))
            };
            match u {
                Some(u) if (0..n as i64).contains(&u) => prev = u,
                _ => return false,
            }
            count += 1;
            if count > degree {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::qgraph::tests::clustered;
    use crate::core::qgraph::{BuildParams, Graph, VSearchParams};

    /// Small indexes covering every array the file holds: both metrics, 4- and 8-bit codes,
    /// residuals 0/4/8, ids 4/3/delta, reordering, aligned rows, the screen and blocks.
    fn index_variants() -> Vec<(String, VGraph, Vec<f32>)> {
        let d = 40;
        let mut out = Vec::new();
        for metric in [Metric::Angular, Metric::Euclidean] {
            let data = clustered(2000, d, 7);
            let queries = clustered(40, d, 8);
            let params = BuildParams {
                metric,
                ..BuildParams::default()
            };
            let g = Graph::build(&data, d, &params).expect("build");
            type Config = (u8, u8, bool, bool, u8, bool);
            let configs: &[Config] = if metric == Metric::Angular {
                &[
                    (4, 8, true, true, 3, false),
                    (8, 4, false, false, 4, true),
                    (4, 0, true, true, 0, false),
                ]
            } else {
                &[(8, 8, true, true, 3, false), (4, 4, false, false, 0, false)]
            };
            for &(bits, res, align, reorder, id_bytes, screen) in configs {
                let v = VGraph::from_graph_with(&g, bits, res, align, reorder, id_bytes, screen)
                    .expect("encode");
                let name =
                    format!("{metric:?} b{bits} r{res} a{align} o{reorder} i{id_bytes} s{screen}");
                out.push((name, v, queries.clone()));
            }
            if metric == Metric::Angular {
                let v = VGraph::from_graph_with(&g, 4, 8, true, true, 3, false)
                    .and_then(|v| v.with_blocks(&g, 2))
                    .expect("encode");
                out.push(("angular blocks".into(), v, queries.clone()));
            }
        }
        out
    }

    fn same_results(what: &str, a: &VGraph, b: &VGraph, queries: &[f32]) {
        let (mut sa, mut sb) = (a.searcher(), b.searcher());
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        for q in queries.chunks_exact(a.dim()) {
            for (ef, rerank) in [(16, 0), (48, 20)] {
                let p = VSearchParams {
                    ef,
                    rerank,
                    ..VSearchParams::default()
                };
                let ea = sa.search(q, 10, p, &mut oa).expect("search");
                let eb = sb.search(q, 10, p, &mut ob).expect("search");
                assert_eq!((ea, &oa), (eb, &ob), "{what}");
            }
        }
    }

    #[test]
    fn streamed_and_mapped_builds_preserve_exact_vertex_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let d = 100;
        for metric in [Metric::Angular, Metric::Euclidean] {
            for n in [4095, 4096, 4097] {
                let data = clustered(n, d, 7);
                let params = BuildParams {
                    metric,
                    degree: 8,
                    build_ef: 16,
                    ..BuildParams::default()
                };
                let graph = Graph::build(&data, d, &params).unwrap();
                let mut rows = super::super::super::PreparedRows::new(n, d).unwrap();
                for chunk in data.chunks(17 * d) {
                    rows.append(chunk).unwrap();
                }
                let mapped = rows.build(&params).unwrap();
                for (bits, res, align, reorder, id_bytes) in [
                    (8, 8, true, false, 4),
                    (4, 8, true, true, 3),
                    (8, 4, false, false, 0),
                    (4, 0, true, true, 0),
                ] {
                    let owned = dir.path().join("owned");
                    let streamed = dir.path().join("streamed");
                    let spilled = dir.path().join("mapped");
                    let index =
                        VGraph::from_graph_with(&graph, bits, res, align, reorder, id_bytes, false)
                            .unwrap();
                    index.save(&owned).unwrap();
                    VGraph::save_from_graph(&graph, &streamed, bits, res, align, reorder, id_bytes)
                        .unwrap();
                    VGraph::save_prepared(&mapped, &spilled, bits, res, align, reorder, id_bytes)
                        .unwrap();
                    let expected = std::fs::read(&owned).unwrap();
                    assert_eq!(
                        std::fs::read(&streamed).unwrap(),
                        expected,
                        "{metric:?} {n} {bits}/{res} {id_bytes}"
                    );
                    assert_eq!(
                        std::fs::read(&spilled).unwrap(),
                        expected,
                        "mapped {metric:?} {n} {bits}/{res} {id_bytes}"
                    );
                    let reopened = VGraph::open(&spilled, Residency::Mapped).unwrap();
                    reopened.release_pages().unwrap();
                    same_results("mapped encoding", &index, &reopened, &data[..3 * d]);
                }
            }
        }
    }

    #[test]
    fn prepared_rows_reject_incomplete_excess_and_invalid_matrices() {
        use super::super::super::PreparedRows;
        let params = BuildParams::default();
        assert!(PreparedRows::new(1, 0).is_err());
        assert!(PreparedRows::new(usize::MAX, 4).is_err());
        for count in [1, 2, 3] {
            let mut rows = PreparedRows::new(2, 4).unwrap();
            let append = rows.append(&vec![1.; count * 4]);
            if count > 2 {
                assert!(append.is_err());
            } else {
                append.unwrap();
                assert_eq!(rows.build(&params).is_ok(), count == 2);
            }
        }
        let mut rows = PreparedRows::new(1, 4).unwrap();
        assert!(rows.append(&[1.; 3]).is_err());
        rows.append(&[f32::NAN; 4]).unwrap();
        assert!(rows.build(&params).is_err());
        let mut rows = PreparedRows::new(1, 4).unwrap();
        rows.append(&[f32::MAX; 4]).unwrap();
        assert!(rows
            .build(&BuildParams {
                metric: Metric::Euclidean,
                ..params
            })
            .is_err());
    }

    #[test]
    fn saved_index_opens_with_identical_results_in_every_residency() {
        let dir = tempfile::tempdir().unwrap();
        for (name, index, queries) in index_variants() {
            let path = dir.path().join("index");
            index.save(&path).unwrap();
            for residency in [Residency::Mapped, Residency::Prefaulted, Residency::Heap] {
                let opened = VGraph::open(&path, residency).unwrap();
                assert_eq!(opened.index_bytes(), index.index_bytes(), "{name}");
                assert_eq!(opened.metric(), index.metric(), "{name}");
                same_results(&format!("{name} {residency:?}"), &index, &opened, &queries);
            }
        }
    }

    #[test]
    fn damaged_files_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (_, index, _) = index_variants().swap_remove(0);
        let path = dir.path().join("a");
        index.save(&path).unwrap();
        let good = std::fs::read(&path).unwrap();
        let open = |bytes: &[u8]| {
            let p = dir.path().join("b");
            std::fs::write(&p, bytes).unwrap();
            VGraph::open(&p, Residency::Heap).map(|_| ())
        };
        let section = |s: usize| {
            let t = 8 + 8 * HEADER_WORDS + 16 * s;
            u64::from_le_bytes(good[t..t + 8].try_into().unwrap()) as usize
        };
        assert!(open(&good).is_ok());
        assert!(open(&good[..100]).is_err(), "truncated header");
        assert!(open(&good[..section(5) + 10]).is_err(), "truncated codes");
        let mut bytes = good.clone();
        let at = section(2);
        bytes[at..at + 3].copy_from_slice(&[0xfe, 0xff, 0xff]);
        let err = open(&bytes).unwrap_err().to_string();
        assert!(err.contains("neighbour id"), "{err}");
        let mut bytes = good.clone();
        bytes[8 + 8 * 6] ^= 1;
        assert!(open(&bytes).is_err(), "stride");
        let mut bytes = good;
        bytes[7] = b'9';
        assert!(open(&bytes).is_err(), "magic");
    }
}
