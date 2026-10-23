// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `std` locks that ignore poisoning: a panic while a lock is held leaves the data reachable
//! instead of turning every later access into a panic. The server catches panics per request
//! and keeps serving, so a poisoned lock would fail every later request on that store.

use std::fmt;
use std::sync::{self, PoisonError, TryLockError};

pub(crate) use std::sync::{MutexGuard, RwLockReadGuard, RwLockWriteGuard};

#[derive(Default)]
pub(crate) struct Mutex<T>(sync::Mutex<T>);

impl<T> Mutex<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(sync::Mutex::new(value))
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        match self.0.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Default)]
pub(crate) struct RwLock<T>(sync::RwLock<T>);

impl<T> RwLock<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(sync::RwLock::new(value))
    }

    pub(crate) fn read(&self) -> RwLockReadGuard<'_, T> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn write(&self) -> RwLockWriteGuard<'_, T> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        match self.0.try_read() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
