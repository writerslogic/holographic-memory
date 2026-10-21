// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Returning memory to the operating system that serving no longer needs: the resident pages
//! of file-backed mappings whose bytes stay on disk (journal frames replayed at open or just
//! committed), and freed heap the allocator keeps after a large transient build.
//!
//! Mechanism per platform:
//! - Linux: `madvise(MADV_DONTNEED)` on a `MAP_SHARED` file mapping unmaps the pages from
//!   the process; the page cache keeps them (dirty or clean), the file is untouched and the
//!   next access refaults. Heap: glibc `malloc_trim(0)` returns free arena memory to the
//!   kernel (musl has no equivalent; no-op).
//! - macOS: `madvise(MADV_DONTNEED)` and `MADV_FREE` only deactivate file-backed pages (they
//!   stay resident until memory pressure), so the pages are discarded with
//!   `msync(MS_SYNC | MS_INVALIDATE)`: dirty pages are written first, then the resident
//!   copies are freed and the next access refaults from the file. Heap:
//!   `malloc_zone_pressure_relief(NULL, 0)` returns the free pages of every malloc zone.
//! - elsewhere (Windows, wasm): no-op.
//!
//! Everything here is a hint: an error is ignored and the data is unaffected. The symbols
//! are declared by hand (the `libc` crate is an optional dependency of this crate), with the
//! platform constants of `<sys/mman.h>` and `<malloc/malloc.h>` given at each use.
//!
//! Knob: `HMS_RELEASE_JOURNAL_PAGES` (`0`, `false`, `off` or `no` disable; default on), read
//! once per process.

use std::sync::OnceLock;

/// Whether journal pages are released after replay and after each committed frame.
pub(crate) fn release_journal_pages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HMS_RELEASE_JOURNAL_PAGES").map_or(true, |value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
    })
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn page_size() -> usize {
    extern "C" {
        fn getpagesize() -> std::os::raw::c_int;
    }
    // SAFETY: `getpagesize` takes no arguments and has no side effects.
    let size = unsafe { getpagesize() };
    usize::try_from(size).unwrap_or(4096).max(1)
}

/// The page-aligned span covering `[ptr, ptr + len)`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn aligned(ptr: *const u8, len: usize) -> (*mut std::os::raw::c_void, usize) {
    let page = page_size();
    let start = (ptr as usize) & !(page - 1);
    let len = len + (ptr as usize - start);
    (start as *mut std::os::raw::c_void, len)
}

/// Drops the resident pages of `[ptr, ptr + len)`, a span inside a `MAP_SHARED` mapping of a
/// file, after the bytes are on disk or in the page cache; the next access refaults them.
/// Callers hold the mapping alive and, for a writable mapping, the writer's lock.
pub(crate) fn release_file_pages(ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn madvise(
                addr: *mut std::os::raw::c_void,
                len: usize,
                advice: std::os::raw::c_int,
            ) -> std::os::raw::c_int;
        }
        const MADV_DONTNEED: std::os::raw::c_int = 4;
        let (addr, len) = aligned(ptr, len);
        // SAFETY: the span lies inside a live shared file mapping (the caller's invariant);
        // MADV_DONTNEED on a shared mapping only drops the process's page-table entries, the
        // page cache and file keep the bytes.
        let _ = unsafe { madvise(addr, len, MADV_DONTNEED) };
    }
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn msync(
                addr: *mut std::os::raw::c_void,
                len: usize,
                flags: std::os::raw::c_int,
            ) -> std::os::raw::c_int;
        }
        const MS_INVALIDATE: std::os::raw::c_int = 0x0002;
        const MS_SYNC: std::os::raw::c_int = 0x0010;
        let (addr, len) = aligned(ptr, len);
        // SAFETY: the span lies inside a live shared file mapping; MS_SYNC writes any dirty
        // page before MS_INVALIDATE discards the resident copies.
        let _ = unsafe { msync(addr, len, MS_SYNC | MS_INVALIDATE) };
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = ptr;
    }
}

/// Returns heap memory the allocator holds free (after a large transient structure was
/// dropped) to the operating system.
pub(crate) fn release_free_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> std::os::raw::c_int;
        }
        // SAFETY: glibc's `malloc_trim` is safe to call at any time from any thread.
        let _ = unsafe { malloc_trim(0) };
    }
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::os::raw::c_void, goal: usize) -> usize;
        }
        // SAFETY: a null zone means every zone; goal 0 means as much as possible. The call is
        // thread-safe and only returns free pages.
        let _ = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn releasing_a_mapping_keeps_its_bytes_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pages.bin");
        let pattern: Vec<u8> = (0..(256 * 1024)).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &pattern).unwrap();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut map = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
        map[7] = 0xAB;
        map.flush_range(0, 4096).unwrap();
        release_file_pages(map.as_ptr(), map.len());
        assert_eq!(map[7], 0xAB);
        assert_eq!(&map[8..], &pattern[8..]);
        release_file_pages(map[100..].as_ptr(), 10);
        assert_eq!(map[100], pattern[100]);
        drop(map);
        assert_eq!(std::fs::read(&path).unwrap()[7], 0xAB);
        release_free_heap();
    }
}
