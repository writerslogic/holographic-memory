// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{
    validate_vector, Encoding, Manifest, Metric, Operation, UpdateError, MAX_DIM, MAX_OPS,
    MAX_PAYLOAD,
};
use crate::core::durable_file::{atomic_write, sync_mapped_file};

const ORDER: u64 = 0x0102_0304_0506_0708;
use crate::core::format::{
    VECTOR_LOG_MAGIC as LOG_MAGIC, VECTOR_MANIFEST_MAGIC as MANIFEST_MAGIC,
    VECTOR_ROWS_MAGIC as ROW_MAGIC,
};
pub(super) const LOG_HEADER: u64 = 60;
const FRAME_HEADER: usize = 24;

pub(super) fn corrupt(at: u64, reason: impl Into<String>) -> UpdateError {
    UpdateError::CorruptLog {
        offset: at,
        reason: reason.into(),
    }
}
fn bad(reason: impl Into<String>) -> UpdateError {
    UpdateError::CorruptBase(reason.into())
}
fn word(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("validated u64 span"))
}
fn small(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("validated u32 span"))
}
fn words(out: &mut Vec<u8>, xs: &[u64]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
fn seal(out: &mut Vec<u8>) {
    out.extend_from_slice(&crc32fast::hash(out).to_le_bytes());
}
fn check(bytes: &[u8], magic: &[u8; 8]) -> Result<(), UpdateError> {
    if bytes.len() < 20 {
        return Err(bad("short header"));
    }
    if &bytes[..8] != magic {
        return Err(UpdateError::UnsupportedFormat(
            "unknown magic or version newer than supported",
        ));
    }
    if cfg!(target_endian = "big") || word(&bytes[8..16]) != ORDER {
        return Err(UpdateError::UnsupportedFormat("byte order mismatch"));
    }
    if crc32fast::hash(&bytes[..bytes.len() - 4]) != small(&bytes[bytes.len() - 4..]) {
        return Err(bad("header checksum mismatch"));
    }
    Ok(())
}

impl Manifest {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = MANIFEST_MAGIC.to_vec();
        words(&mut out, &[ORDER]);
        out.extend_from_slice(&self.identity);
        let b = &self.build;
        let e = &self.encoding;
        words(
            &mut out,
            &[
                self.dim as u64,
                b.metric.code(),
                self.sequence,
                b.degree as u64,
                b.build_ef as u64,
                u64::from(b.alpha.to_bits()),
                b.seed,
                u64::from(b.quantized_search),
                u64::from(e.bits),
                u64::from(e.residual_bits),
                u64::from(e.align),
                u64::from(e.reorder),
                u64::from(e.id_bytes),
                u64::from(e.screen),
                u64::from(e.block_bits),
                0,
            ],
        );
        seal(&mut out);
        out
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, UpdateError> {
        if bytes.len() != 164 {
            return Err(bad("manifest length"));
        }
        check(bytes, MANIFEST_MAGIC)?;
        let h: Vec<_> = bytes[32..160]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| word(c))
            .collect();
        if h[0] == 0
            || h[0] > MAX_DIM as u64
            || h[1] > 1
            || h[3] == 0
            || h[3] > 65536
            || !h[3].is_multiple_of(2)
            || h[4] == 0
            || h[4] > 1_048_576
            || h[5] > u32::MAX as u64
            || h[7] > 1
            || !matches!(h[8], 4 | 8)
            || !matches!(h[9], 0 | 4 | 8)
            || h[10] > 1
            || h[11] > 1
            || !matches!(h[12], 0 | 3 | 4)
            || h[13] > 1
            || !matches!(h[14], 0 | 1 | 2 | 4)
            || h[15] != 0
        {
            return Err(bad("manifest parameter range"));
        }
        let alpha = f32::from_bits(h[5] as u32);
        if !alpha.is_finite() || alpha < 1.0 || (h[1] == 1 && (h[13] != 0 || h[14] != 0)) {
            return Err(bad("manifest metric or alpha"));
        }
        Ok(Self {
            identity: bytes[16..32].try_into().expect("identity"),
            dim: h[0] as usize,
            sequence: h[2],
            build: super::BuildParams {
                metric: if h[1] == 0 {
                    Metric::Angular
                } else {
                    Metric::Euclidean
                },
                degree: h[3] as usize,
                build_ef: h[4] as usize,
                alpha,
                seed: h[6],
                quantized_search: h[7] != 0,
                ..Default::default()
            },
            encoding: Encoding {
                bits: h[8] as u8,
                residual_bits: h[9] as u8,
                align: h[10] != 0,
                reorder: h[11] != 0,
                id_bytes: h[12] as u8,
                screen: h[13] != 0,
                block_bits: h[14] as u8,
            },
        })
    }
}

pub(super) struct BaseRows {
    map: memmap2::Mmap,
    pub ids: Vec<u32>,
    dim: usize,
    row_at: usize,
    metric: Metric,
}
impl BaseRows {
    pub fn write(
        path: &Path,
        manifest: &Manifest,
        ids: &[u32],
        vectors: &[f32],
    ) -> Result<(), UpdateError> {
        let row_at =
            (64 + ids.len().checked_mul(4).ok_or(UpdateError::Capacity)?).next_multiple_of(64);
        let mut header = ROW_MAGIC.to_vec();
        words(
            &mut header,
            &[
                ORDER,
                manifest.dim as u64,
                ids.len() as u64,
                manifest.build.metric.code(),
                manifest.sequence,
                row_at as u64,
            ],
        );
        seal(&mut header);
        header.extend_from_slice(&0u32.to_le_bytes());
        let parent = path.parent().ok_or(UpdateError::Capacity)?;
        let mut temp = tempfile::NamedTempFile::new_in(parent)?;
        let mut writer = std::io::BufWriter::new(temp.as_file_mut());
        writer.write_all(&header)?;
        for id in ids {
            writer.write_all(&id.to_le_bytes())?;
        }
        writer.write_all(&vec![0u8; row_at - 64 - ids.len() * 4])?;
        let mut bytes = Vec::with_capacity(manifest.dim * 4);
        for row in vectors.chunks_exact(manifest.dim) {
            bytes.clear();
            for x in row {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            writer.write_all(&bytes)?;
            writer.write_all(&crc32fast::hash(&bytes).to_le_bytes())?;
        }
        writer.flush()?;
        drop(writer);
        temp.as_file().sync_all()?;
        sync_mapped_file(temp.as_file())?;
        temp.persist(path).map_err(|e| e.error)?;
        Ok(())
    }
    pub fn open(path: &Path, manifest: &Manifest) -> Result<Self, UpdateError> {
        let file = File::open(path)?;
        if file.metadata()?.len() < 64 {
            return Err(bad("short rows file"));
        }
        // SAFETY: immutable generation files are only replaced by rename, never modified or
        // truncated while a mapping exists. The directory's exclusive lock excludes writers.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        check(&map[..60], ROW_MAGIC)?;
        let dim = word(&map[16..24]);
        let count = word(&map[24..32]);
        if dim != manifest.dim as u64
            || count >= u32::MAX as u64
            || word(&map[32..40]) != manifest.build.metric.code()
            || word(&map[40..48]) != manifest.sequence
            || small(&map[60..64]) != 0
        {
            return Err(bad("rows header mismatch"));
        }
        let count = usize::try_from(count).map_err(|_| UpdateError::Capacity)?;
        let row_at = 64usize
            .checked_add(count.checked_mul(4).ok_or(UpdateError::Capacity)?)
            .ok_or(UpdateError::Capacity)?
            .next_multiple_of(64);
        let stride = manifest
            .dim
            .checked_mul(4)
            .and_then(|v| v.checked_add(4))
            .ok_or(UpdateError::Capacity)?;
        let end = count
            .checked_mul(stride)
            .and_then(|v| v.checked_add(row_at))
            .ok_or(UpdateError::Capacity)?;
        if word(&map[48..56]) != row_at as u64 || end != map.len() {
            return Err(bad("rows length mismatch"));
        }
        let mut ids = Vec::new();
        ids.try_reserve_exact(count)
            .map_err(|_| UpdateError::Capacity)?;
        ids.extend(
            map[64..64 + count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| small(c)),
        );
        if ids.last() == Some(&u32::MAX) || ids.windows(2).any(|p| p[0] >= p[1]) {
            return Err(bad("rows ids not strictly ordered"));
        }
        Ok(Self {
            map,
            ids,
            dim: manifest.dim,
            row_at,
            metric: manifest.build.metric,
        })
    }
    pub fn row(&self, slot: usize) -> Result<&[f32], UpdateError> {
        let at = self.row_at + slot * (self.dim * 4 + 4);
        let bytes = &self.map[at..at + self.dim * 4];
        if crc32fast::hash(bytes) != small(&self.map[at + self.dim * 4..at + self.dim * 4 + 4]) {
            return Err(bad(format!("row {slot} checksum")));
        }
        // SAFETY: checked file geometry bounds the span; mmap and row starts align to f32,
        // all f32 bit patterns are valid, and check() rejects big-endian targets.
        let row = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), self.dim) };
        validate_vector(row, self.dim, self.metric)?;
        Ok(row)
    }
}

pub(super) fn log_header(manifest: &Manifest) -> Vec<u8> {
    let mut out = LOG_MAGIC.to_vec();
    words(&mut out, &[ORDER]);
    out.extend_from_slice(&manifest.identity);
    words(
        &mut out,
        &[
            manifest.dim as u64,
            manifest.build.metric.code(),
            manifest.sequence,
        ],
    );
    seal(&mut out);
    out
}
pub(super) fn rotate(path: &Path, manifest: &Manifest) -> Result<(), UpdateError> {
    atomic_write(path, &log_header(manifest)).map_err(UpdateError::Persistence)
}

pub(super) fn frame(ops: &[Operation], sequence: u64, dim: usize) -> Result<Vec<u8>, UpdateError> {
    let len = ops.iter().try_fold(0usize, |n, op| {
        n.checked_add(5 + op.vector().map_or(0, |_| dim * 4))
            .ok_or(UpdateError::Capacity)
    })?;
    if len > MAX_PAYLOAD || ops.len() > MAX_OPS {
        return Err(UpdateError::Capacity);
    }
    let mut out = Vec::with_capacity(FRAME_HEADER + len);
    out.extend_from_slice(&(len as u32).to_le_bytes());
    out.extend_from_slice(&(!(len as u32)).to_le_bytes());
    words(&mut out, &[sequence]);
    out.extend_from_slice(&(ops.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    for op in ops {
        out.push(op.tag());
        out.extend_from_slice(&op.id().to_le_bytes());
        if let Some(row) = op.vector() {
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let mut crc = crc32fast::Hasher::new();
    crc.update(&out[..20]);
    crc.update(&out[24..]);
    out[20..24].copy_from_slice(&crc.finalize().to_le_bytes());
    Ok(out)
}

pub(super) fn replay(
    path: &Path,
    manifest: &Manifest,
    mut apply: impl FnMut(Vec<Operation>) -> Result<(), UpdateError>,
) -> Result<(u64, u64), UpdateError> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length < LOG_HEADER {
        return Err(corrupt(0, "short log header"));
    }
    let mut header = [0u8; LOG_HEADER as usize];
    file.read_exact(&mut header)?;
    check(&header, LOG_MAGIC).map_err(|e| match e {
        UpdateError::UnsupportedFormat(_) => e,
        _ => corrupt(0, e.to_string()),
    })?;
    if header[16..32] != manifest.identity
        || word(&header[32..40]) != manifest.dim as u64
        || word(&header[40..48]) != manifest.build.metric.code()
    {
        return Err(corrupt(0, "log identity or shape mismatch"));
    }
    let mut sequence = word(&header[48..56]);
    if sequence > manifest.sequence {
        return Err(corrupt(0, "log origin exceeds base checkpoint"));
    }
    let mut at = LOG_HEADER;
    while length - at >= FRAME_HEADER as u64 {
        let mut h = [0u8; FRAME_HEADER];
        file.read_exact(&mut h)?;
        let len = small(&h[..4]) as usize;
        let count = small(&h[16..20]) as usize;
        if small(&h[4..8]) != !(len as u32)
            || len > MAX_PAYLOAD
            || count == 0
            || count > MAX_OPS
            || count > len / 5
        {
            return Err(corrupt(at, "invalid frame length or count"));
        }
        if length - at - (FRAME_HEADER as u64) < len as u64 {
            break;
        }
        let mut payload = vec![0; len];
        file.read_exact(&mut payload)?;
        let mut crc = crc32fast::Hasher::new();
        crc.update(&h[..20]);
        crc.update(&payload);
        if crc.finalize() != small(&h[20..24]) {
            return Err(corrupt(at, "frame checksum mismatch"));
        }
        let next = word(&h[8..16]);
        if sequence.checked_add(1) != Some(next) {
            return Err(corrupt(at, "nonconsecutive sequence"));
        }
        let mut cursor = 0usize;
        let mut ops = Vec::with_capacity(count);
        let mut ids = std::collections::HashSet::with_capacity(count);
        for _ in 0..count {
            let record = payload
                .get(cursor..cursor + 5)
                .ok_or_else(|| corrupt(at, "short operation"))?;
            let tag = record[0];
            let id = small(&record[1..5]);
            cursor += 5;
            if id == u32::MAX || !ids.insert(id) {
                return Err(corrupt(at, "invalid or duplicate id"));
            }
            let op = match tag {
                1 | 3 => {
                    let end = cursor
                        .checked_add(manifest.dim * 4)
                        .ok_or_else(|| corrupt(at, "vector size overflow"))?;
                    let bytes = payload
                        .get(cursor..end)
                        .ok_or_else(|| corrupt(at, "short vector"))?;
                    let row: Vec<_> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_bits(small(c)))
                        .collect();
                    cursor = end;
                    validate_vector(&row, manifest.dim, manifest.build.metric)
                        .map_err(|e| corrupt(at, e.to_string()))?;
                    if tag == 1 {
                        Operation::Insert(id, row)
                    } else {
                        Operation::Replace(id, row)
                    }
                }
                2 => Operation::Delete(id),
                _ => return Err(corrupt(at, "unknown operation")),
            };
            ops.push(op);
        }
        if cursor != payload.len() {
            return Err(corrupt(at, "trailing frame data"));
        }
        if next > manifest.sequence {
            apply(ops).map_err(|e| corrupt(at, e.to_string()))?;
        }
        sequence = next;
        at += FRAME_HEADER as u64 + len as u64;
    }
    Ok((sequence.max(manifest.sequence), at))
}

pub(super) fn append(path: &Path, end: u64, bytes: &[u8]) -> Result<(), UpdateError> {
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(end))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    sync_mapped_file(&file)?;
    Ok(())
}
