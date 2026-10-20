// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! File-backed chunk text: the store keeps `(offset, length)` per chunk and reads a chunk's
//! text back with a positioned read when a result needs it, so the serving process holds no
//! copy of the corpus text. The journal frames still carry every document's text, so the
//! segment is a cache rebuilt from the journal at open (truncated, then re-appended as the
//! documents are replayed); it needs no durability of its own. Positioned reads leave the
//! bytes in the page cache rather than in the process (a mapping would count every touched
//! page in RSS).

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{ensure, Context, Result};

/// Appends are serialized by the store's write lock; readers (captured queries that outlive
/// the lock) only touch bytes below the length they observed, so the segment needs no lock.
pub(crate) struct TextSegment {
    file: File,
    len: AtomicU64,
}

impl TextSegment {
    /// Creates or truncates the segment at `path`.
    pub(crate) fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("creating text segment {}", path.display()))?;
        Ok(Self {
            file,
            len: AtomicU64::new(0),
        })
    }

    /// Appends `text` and returns its location.
    pub(crate) fn append(&self, text: &str) -> Result<(u64, u32)> {
        let len = u32::try_from(text.len()).context("chunk text over 4 GiB")?;
        let at = self.len.load(Ordering::Acquire);
        positioned_write(&self.file, text.as_bytes(), at).context("appending chunk text")?;
        self.len.store(at + u64::from(len), Ordering::Release);
        Ok((at, len))
    }

    /// Reads the text stored at a location returned by [`TextSegment::append`].
    pub(crate) fn read(&self, at: u64, len: u32) -> Result<String> {
        ensure!(
            at.checked_add(u64::from(len))
                .is_some_and(|end| end <= self.len.load(Ordering::Acquire)),
            "chunk text location outside the text segment"
        );
        let mut bytes = vec![0u8; len as usize];
        positioned_read(&self.file, &mut bytes, at).context("reading chunk text")?;
        String::from_utf8(bytes).context("chunk text is not UTF-8")
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> u64 {
        self.len.load(Ordering::Acquire)
    }
}

#[cfg(unix)]
fn positioned_read(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, at)
}

#[cfg(unix)]
fn positioned_write(file: &File, buf: &[u8], at: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, at)
}

#[cfg(windows)]
fn positioned_write(file: &File, buf: &[u8], at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_write(&buf[done..], at + done as u64)?;
        if n == 0 {
            return Err(std::io::ErrorKind::WriteZero.into());
        }
        done += n;
    }
    Ok(())
}

#[cfg(windows)]
fn positioned_read(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], at + done as u64)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        done += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_round_trip_and_stay_inside_the_segment() {
        let dir = tempfile::tempdir().unwrap();
        let segment = TextSegment::create(&dir.path().join("t")).unwrap();
        let a = segment.append("").unwrap();
        let b = segment.append("héllo wörld").unwrap();
        let c = segment.append(&"x".repeat(70_000)).unwrap();
        assert_eq!(a, (0, 0));
        assert_eq!(b, (0, "héllo wörld".len() as u32));
        assert_eq!(c.0, u64::from(b.1));
        assert_eq!(segment.read(a.0, a.1).unwrap(), "");
        assert_eq!(segment.read(b.0, b.1).unwrap(), "héllo wörld");
        assert_eq!(segment.read(c.0, c.1).unwrap().len(), 70_000);
        assert_eq!(segment.len(), u64::from(b.1) + 70_000);
        // Past the end, and a length that overflows, are errors, never reads.
        assert!(segment.read(c.0, c.1 + 1).is_err());
        assert!(segment.read(segment.len(), 1).is_err());
        assert!(segment.read(u64::MAX, 1).is_err());
        // A slice that splits a multi-byte character is rejected as text.
        assert!(segment.read(b.0 + 1, 1).is_err());
        // Reopening truncates: the journal replay rebuilds it.
        let reopened = TextSegment::create(&dir.path().join("t")).unwrap();
        assert_eq!(reopened.len(), 0);
    }
}
