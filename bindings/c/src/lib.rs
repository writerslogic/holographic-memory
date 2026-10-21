// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stable C ABI for the Holographic Memory System document store.
//!
//! Conventions, stated once because every function follows them:
//!
//! * Every fallible function returns an `hms_status_t`; `HMS_OK` is zero. On failure the
//!   calling thread's `hms1_last_error_code` and `hms1_last_error_message` describe it.
//! * No panic crosses the boundary: each entry point catches unwinds and reports
//!   `HMS_ERR_PANIC`.
//! * Strings and byte buffers are `hms_str_t` (pointer plus length, UTF-8, not NUL
//!   terminated). Every length is validated against a documented limit before use and
//!   nothing is allocated from a caller-supplied size.
//! * Handles are opaque. `hms_store_t` is thread-safe: any number of threads may search,
//!   ingest, delete and read statistics concurrently, and `hms1_store_close` may run while
//!   other calls are in flight (they finish first; the store is released after the last).
//!   Closed or never-issued store handles are rejected with `HMS_ERR_INVALID_HANDLE`.
//! * `hms_results_t` and `hms_chunks_t` are immutable after creation: any number of threads
//!   may read one concurrently, but its `free` must not race with any read, and freeing one
//!   twice is undefined behaviour (NULL is accepted and ignored).
//! * Symbols carry the ABI major version (`hms1_*`). The header also defines unversioned
//!   `hms_*` aliases. Struct arguments that begin with `struct_size` grow compatibly: append-
//!   only, older callers keep working.
//!
//! The ABI is versioned by `hms1_abi_version`: major changes break the symbol names, minor
//! changes only add functions and struct fields, patch changes are behavioural fixes.

mod error;
mod store;
mod types;

pub use error::*;
pub use store::*;
pub use types::*;

use std::os::raw::c_char;

/// The ABI version, packed as `major << 16 | minor << 8 | patch`.
pub const HMS_ABI_VERSION: u32 = 0x01_00_00;
/// The ABI major version, equal to the numeric suffix of the `hms1_` symbols.
pub const HMS_ABI_MAJOR: u32 = 1;

static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// The ABI version the loaded library implements, packed as `major << 16 | minor << 8 |
/// patch`. A caller built against `HMS_ABI_VERSION` is compatible when the majors are equal
/// and the library's minor is at least the caller's.
#[no_mangle]
pub extern "C" fn hms1_abi_version() -> u32 {
    HMS_ABI_VERSION
}

/// The library's own version string (the `holographic-memory-c` crate version), a static
/// NUL-terminated UTF-8 string. Never NULL; never freed.
#[no_mangle]
pub extern "C" fn hms1_version() -> *const c_char {
    VERSION.as_ptr().cast()
}
