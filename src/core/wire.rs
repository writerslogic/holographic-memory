// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shared little-endian wire primitives for arena/log serialization.
//!
//! These primitives express the byte layout reused across the atom, composite,
//! triple, and relation serializers so the encoding lives in exactly one place.
//! Every helper is byte-for-byte compatible with the hand-inlined code it
//! replaces; do not change any layout here without a backward-compatible
//! migration.

/// Append a length-prefixed string: `[len:u16 LE][utf8 bytes]`.
///
/// Callers are responsible for ensuring `s` is at most `u16::MAX` bytes.
pub(crate) fn write_lp_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Read a length-prefixed string written by [`write_lp_str`], returning the
/// decoded value and the position just past it. Returns `None` on truncation or
/// invalid UTF-8.
pub(crate) fn read_lp_str(data: &[u8], pos: usize) -> Option<(String, usize)> {
    let len = u16::from_le_bytes(data.get(pos..pos + 2)?.try_into().ok()?) as usize;
    let start = pos + 2;
    let s = std::str::from_utf8(data.get(start..start + len)?)
        .ok()?
        .to_string();
    Some((s, start + len))
}

/// Append a delta run: `[count:u32 LE][delta:u32 LE ...]`.
pub(crate) fn write_deltas(buf: &mut Vec<u8>, deltas: &[u32]) {
    buf.extend_from_slice(&(deltas.len() as u32).to_le_bytes());
    for &d in deltas {
        buf.extend_from_slice(&d.to_le_bytes());
    }
}

/// Read a delta run written by [`write_deltas`], returning the deltas and the
/// position just past them. Returns `None` on truncation.
pub(crate) fn read_deltas(data: &[u8], pos: usize) -> Option<(Vec<u32>, usize)> {
    let count = u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?) as usize;
    let mut start = pos + 4;
    let mut deltas = Vec::with_capacity(count);
    for _ in 0..count {
        deltas.push(u32::from_le_bytes(
            data.get(start..start + 4)?.try_into().ok()?,
        ));
        start += 4;
    }
    Some((deltas, start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_str_roundtrip() {
        let mut buf = vec![0xAB]; // leading byte the reader must skip
        write_lp_str(&mut buf, "hello");
        let (s, pos) = read_lp_str(&buf, 1).unwrap();
        assert_eq!(s, "hello");
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn lp_str_bytes_match_manual() {
        let mut buf = Vec::new();
        write_lp_str(&mut buf, "abc");
        let mut manual = Vec::new();
        manual.extend_from_slice(&(3u16).to_le_bytes());
        manual.extend_from_slice(b"abc");
        assert_eq!(buf, manual);
    }

    #[test]
    fn deltas_roundtrip() {
        let mut buf = Vec::new();
        let deltas = vec![1u32, 5, 9, 42];
        write_deltas(&mut buf, &deltas);
        let (out, pos) = read_deltas(&buf, 0).unwrap();
        assert_eq!(out, deltas);
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn read_truncated_is_none() {
        assert!(read_lp_str(&[0x02, 0x00, b'a'], 0).is_none());
        assert!(read_deltas(&[0x01, 0x00, 0x00, 0x00], 0).is_none());
    }
}
