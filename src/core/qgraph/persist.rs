// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Raw little-endian serialization of a built [`QGraph`]. The block layout is recomputed from
//! the header, so a file can only describe a layout that `build` could have produced; every
//! length and id is checked before it is used.

use std::io::{self, Read, Write};

use super::kernels::Rotation;
use super::{Codes, Layout, QGraph, Store, NONE};

const MAGIC: &[u8; 8] = b"HMSQG\0\0\x01";
/// Largest padded dimension accepted from a file.
const MAX_PADDED: usize = 1 << 16;

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("qgraph file: {msg}"))
}

fn put_u64(w: &mut impl Write, x: u64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}

fn get_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn get_usize(r: &mut impl Read, max: usize, what: &str) -> io::Result<usize> {
    let x = get_u64(r)?;
    usize::try_from(x)
        .ok()
        .filter(|&x| x <= max)
        .ok_or_else(|| invalid(what))
}

fn put_u64s(w: &mut impl Write, xs: &[u64]) -> io::Result<()> {
    put_u64(w, xs.len() as u64)?;
    for chunk in xs.chunks(1 << 16) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|x| x.to_le_bytes()).collect();
        w.write_all(&bytes)?;
    }
    Ok(())
}

fn put_u32s(w: &mut impl Write, xs: &[u32]) -> io::Result<()> {
    put_u64(w, xs.len() as u64)?;
    let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.write_all(&bytes)
}

fn put_f32s(w: &mut impl Write, xs: &[f32]) -> io::Result<()> {
    put_u64(w, xs.len() as u64)?;
    let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.write_all(&bytes)
}

/// Reads a length prefix that must equal `expected`.
fn get_len(r: &mut impl Read, expected: usize, what: &str) -> io::Result<()> {
    if get_u64(r)? == expected as u64 {
        Ok(())
    } else {
        Err(invalid(what))
    }
}

fn get_u64s(r: &mut impl Read, len: usize, what: &str) -> io::Result<Vec<u64>> {
    get_len(r, len, what)?;
    let mut out = Vec::with_capacity(len);
    let mut buf = vec![0u8; 8 << 16];
    let mut left = len;
    while left > 0 {
        let take = left.min(1 << 16);
        let bytes = &mut buf[..take * 8];
        r.read_exact(bytes)?;
        out.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b)),
        );
        left -= take;
    }
    Ok(out)
}

fn get_u32s(r: &mut impl Read, len: usize, what: &str) -> io::Result<Vec<u32>> {
    get_len(r, len, what)?;
    get_u32s_body(r, len)
}

/// Reads `len` values whose length prefix the caller has already consumed.
fn get_u32s_body(r: &mut impl Read, len: usize) -> io::Result<Vec<u32>> {
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect())
}

fn get_f32s(r: &mut impl Read, len: usize, what: &str) -> io::Result<Vec<f32>> {
    Ok(get_u32s(r, len, what)?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}

impl QGraph {
    /// Writes the index in the format read by [`QGraph::read_from`].
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        let store = match self.store {
            Store::F32 => 0,
            Store::I16 => 1,
            Store::I8 => 2,
        };
        let codes = match self.codes {
            Codes::Edge => 0,
            Codes::Vertex => 1,
        };
        for x in [
            self.n,
            self.dim,
            self.degree,
            self.rotation.padded(),
            store,
            codes,
            self.entry as usize,
        ] {
            put_u64(w, x as u64)?;
        }
        put_f32s(w, self.rotation.signs())?;
        put_u64s(w, &self.blocks)?;
        put_u64s(w, &self.vcodes)?;
        put_f32s(w, &self.centroid)?;
        put_u32s(w, &self.upper_ids)?;
        put_u64(w, self.layers.len() as u64)?;
        for layer in &self.layers {
            put_u64(w, layer.len() as u64)?;
            for adj in layer {
                put_u32s(w, adj)?;
            }
        }
        Ok(())
    }

    /// Reads an index written by [`QGraph::write_to`], rejecting inconsistent files.
    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("bad magic or version"));
        }
        let n = get_usize(r, NONE as usize - 1, "n")?;
        let dim = get_usize(r, MAX_PADDED, "dim")?;
        let degree = get_usize(r, 1 << 12, "degree")?;
        let padded = get_usize(r, MAX_PADDED, "padded")?;
        let store = match get_u64(r)? {
            0 => Store::F32,
            1 => Store::I16,
            2 => Store::I8,
            _ => return Err(invalid("store")),
        };
        let codes = match get_u64(r)? {
            0 => Codes::Edge,
            1 => Codes::Vertex,
            _ => return Err(invalid("codes")),
        };
        let entry = get_usize(r, n.saturating_sub(1), "entry")? as u32;
        if n == 0 || dim == 0 || degree == 0 || !degree.is_multiple_of(2) {
            return Err(invalid("empty index or odd degree"));
        }
        let rotation = Rotation::from_signs(dim, get_f32s(r, 3 * padded, "rotation")?)
            .filter(|rot| rot.padded() == padded)
            .ok_or_else(|| invalid("rotation"))?;
        let words = padded / 64;
        let layout = Layout::new(dim, words, degree, store, codes);
        let blocks_len = n
            .checked_mul(layout.stride)
            .ok_or_else(|| invalid("size"))?;
        let blocks = get_u64s(r, blocks_len, "blocks")?;
        let (vcodes_len, centroid_len) = match codes {
            Codes::Edge => (0, 0),
            Codes::Vertex => (n * (words + 2), padded),
        };
        let vcodes = get_u64s(r, vcodes_len, "vertex codes")?;
        let centroid = get_f32s(r, centroid_len, "centroid")?;
        let n_upper = get_usize(r, n, "upper ids")?;
        let upper_ids = get_u32s_body(r, n_upper)?;
        if upper_ids.iter().any(|&u| u as usize >= n) {
            return Err(invalid("upper id out of range"));
        }
        let n_layers = get_usize(r, 64, "layers")?;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let size = get_usize(r, n_upper, "layer size")?;
            let mut layer = Vec::with_capacity(size);
            for _ in 0..size {
                let len = get_usize(r, size, "upper degree")?;
                let adj = get_u32s_body(r, len)?;
                if adj.iter().any(|&u| u as usize >= size) {
                    return Err(invalid("upper neighbour out of range"));
                }
                layer.push(adj);
            }
            layers.push(layer);
        }
        if !layers.is_empty() && layers.iter().any(|l| l.is_empty()) {
            return Err(invalid("empty layer"));
        }
        let index = Self {
            n,
            dim,
            degree,
            words,
            rotation,
            store,
            codes,
            vec_words: layout.vec_words,
            stride: layout.stride,
            codes_off: layout.codes_off,
            factors_off: layout.factors_off,
            ids_off: layout.ids_off,
            blocks,
            vcodes,
            centroid,
            entry,
            upper_ids,
            layers,
        };
        let ids_ok = (0..n as u32).all(|v| {
            let ids = super::kernels::as_u32(&index.block(v)[index.ids_off..]);
            ids[..degree].iter().all(|&u| u == NONE || (u as usize) < n)
        });
        if !ids_ok {
            return Err(invalid("neighbour id out of range"));
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernels::SplitMix;
    use super::super::{BuildParams, Codes, QGraph, SearchParams, Store};

    #[test]
    fn round_trip_gives_identical_results() {
        let (n, dim) = (1500, 40);
        let mut rng = SplitMix(7);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_f32() - 0.5).collect();
        for (store, codes) in [(Store::F32, Codes::Edge), (Store::I8, Codes::Vertex)] {
            let params = BuildParams {
                degree: 16,
                build_ef: 32,
                store,
                codes,
                ..BuildParams::default()
            };
            let built = QGraph::build(&data, dim, &params);
            let mut bytes = Vec::new();
            built.write_to(&mut bytes).unwrap();
            let loaded = QGraph::read_from(&mut bytes.as_slice()).unwrap();
            assert_eq!(loaded.index_bytes(), built.index_bytes());
            let sp = SearchParams {
                ef: 24,
                max_exact: 0,
            };
            let (mut a, mut b) = (built.searcher(), loaded.searcher());
            let (mut ra, mut rb) = (Vec::new(), Vec::new());
            for q in data.chunks_exact(dim).step_by(37) {
                assert_eq!(a.search(q, 10, sp, &mut ra), b.search(q, 10, sp, &mut rb));
                assert_eq!(ra, rb);
            }
            bytes.truncate(bytes.len() - 1);
            assert!(QGraph::read_from(&mut bytes.as_slice()).is_err());
        }
    }
}
