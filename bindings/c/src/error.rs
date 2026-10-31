// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Status codes and the per-thread last-error slot.

use std::cell::RefCell;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Status returned by every fallible function. Zero is success; every failure also leaves a
/// message readable with `hms1_last_error_message` on the calling thread.
#[allow(non_camel_case_types)]
pub type warm_status_t = i32;

/// Success.
pub const WARM_OK: warm_status_t = 0;
/// A required pointer argument was NULL (or a non-empty buffer had a NULL pointer).
pub const WARM_ERR_NULL_POINTER: warm_status_t = 1;
/// An argument was malformed or out of range, or the engine rejected the request.
pub const WARM_ERR_INVALID_ARGUMENT: warm_status_t = 2;
/// A string argument was not valid UTF-8.
pub const WARM_ERR_INVALID_UTF8: warm_status_t = 3;
/// A handle is not a live handle of the expected kind (closed, freed, or never issued).
pub const WARM_ERR_INVALID_HANDLE: warm_status_t = 4;
/// An operating-system I/O failure.
pub const WARM_ERR_IO: warm_status_t = 5;
/// The store is open in another engine (this or another process).
pub const WARM_ERR_LOCKED: warm_status_t = 6;
/// The encryption key is missing, wrong, or the store cannot be decrypted with it.
pub const WARM_ERR_ENCRYPTION: warm_status_t = 7;
/// Stored data failed an integrity check.
pub const WARM_ERR_CORRUPT: warm_status_t = 8;
/// A length or count exceeds a documented limit.
pub const WARM_ERR_LIMIT_EXCEEDED: warm_status_t = 9;
/// A `struct_size` field is smaller than the first field it must cover.
pub const WARM_ERR_ABI_MISMATCH: warm_status_t = 10;
/// A Rust panic was caught at the boundary; the store should be considered suspect.
pub const WARM_ERR_PANIC: warm_status_t = 98;
/// An unclassified engine failure.
pub const WARM_ERR_INTERNAL: warm_status_t = 99;

/// Largest single input buffer accepted, in bytes.
pub const WARM_MAX_BUFFER_BYTES: usize = 1 << 30;
/// Largest element count accepted for any input array.
pub const WARM_MAX_ARRAY_LEN: usize = 1 << 24;

pub(crate) struct ApiError {
    pub code: warm_status_t,
    pub message: String,
}

impl ApiError {
    pub fn new(code: warm_status_t, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Classify an engine error. `default` applies when nothing more specific is recognised.
    pub fn from_engine(error: &anyhow::Error, default: warm_status_t) -> Self {
        let message = format!("{error:#}");
        let lower = message.to_ascii_lowercase();
        let code = if lower.contains("already open") {
            WARM_ERR_LOCKED
        } else if lower.contains("decrypt")
            || lower.contains("encryption")
            || lower.contains("passphrase")
            || lower.contains("aes-256")
        {
            WARM_ERR_ENCRYPTION
        } else if lower.contains("checksum") || lower.contains("corrupt") {
            WARM_ERR_CORRUPT
        } else if error
            .chain()
            .any(|cause| cause.downcast_ref::<std::io::Error>().is_some())
        {
            WARM_ERR_IO
        } else {
            default
        };
        Self { code, message }
    }
}

pub(crate) type ApiResult<T> = Result<T, ApiError>;

struct Slot {
    code: warm_status_t,
    /// UTF-8 message, always NUL terminated.
    message: Vec<u8>,
}

thread_local! {
    static LAST: RefCell<Slot> = const { RefCell::new(Slot { code: 0, message: Vec::new() }) };
}

fn clear() {
    LAST.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.code = WARM_OK;
        slot.message.clear();
        slot.message.push(0);
    });
}

fn record(error: &ApiError) -> warm_status_t {
    LAST.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.code = error.code;
        slot.message.clear();
        slot.message
            .extend(error.message.bytes().map(|b| if b == 0 { b' ' } else { b }));
        slot.message.push(0);
    });
    error.code
}

/// Run `body` with the last-error slot reset, converting failures and panics to a status.
pub(crate) fn guard(body: impl FnOnce() -> ApiResult<()>) -> warm_status_t {
    clear();
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => WARM_OK,
        Ok(Err(error)) => record(&error),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_owned());
            record(&ApiError::new(
                WARM_ERR_PANIC,
                format!("internal panic: {detail}"),
            ))
        }
    }
}

/// The code of the last failed call on this thread, or `WARM_OK` if the last call succeeded.
#[no_mangle]
pub extern "C" fn hms1_last_error_code() -> warm_status_t {
    LAST.with(|slot| slot.borrow().code)
}

/// The message of the last failed call on this thread as a NUL-terminated UTF-8 string, empty
/// if the last call succeeded. Never NULL. The pointer stays valid until the next `hms1_*`
/// call on the same thread (any call resets it); copy it if it must outlive that.
#[no_mangle]
pub extern "C" fn hms1_last_error_message() -> *const c_char {
    LAST.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.message.is_empty() {
            slot.message.push(0);
        }
        slot.message.as_ptr().cast()
    })
}
