// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use parking_lot::{Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Default)]
struct WriterState {
    next: u64,
    serving: u64,
}

#[derive(Default)]
struct WriterQueue {
    state: Mutex<WriterState>,
    changed: Condvar,
}

pub(super) struct WriterLease<'a> {
    queue: &'a WriterQueue,
    held: std::cell::Cell<bool>,
}

impl WriterQueue {
    fn acquire(&self) -> WriterLease<'_> {
        let mut state = self.state.lock();
        let ticket = state.next;
        state.next = ticket
            .checked_add(1)
            .expect("writer reservation counter exhausted");
        while state.serving != ticket {
            self.changed.wait(&mut state);
        }
        WriterLease {
            queue: self,
            held: std::cell::Cell::new(true),
        }
    }
}

impl WriterLease<'_> {
    /// Hands the queue to the next writer now; dropping the lease afterwards does nothing.
    /// Maintenance uses it once its arena pointer has moved, so the work that follows (index
    /// persistence, the audit record) no longer holds writers back.
    pub(super) fn release(&self) {
        if !self.held.replace(false) {
            return;
        }
        let mut state = self.queue.state.lock();
        state.serving = state
            .serving
            .checked_add(1)
            .expect("writer reservation counter overflow");
        self.queue.changed.notify_all();
    }
}

impl Drop for WriterLease<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Default)]
pub(super) struct MutationGate {
    writers: WriterQueue,
    readers: RwLock<()>,
}

pub(super) struct WriteGuard<'a> {
    _readers: RwLockWriteGuard<'a, ()>,
    _writer: WriterLease<'a>,
}

pub(super) struct TransactionGuard<'a> {
    gate: &'a MutationGate,
    writer: WriterLease<'a>,
}

impl<'a> TransactionGuard<'a> {
    pub(super) fn publish(self) -> WriteGuard<'a> {
        WriteGuard {
            _readers: self.gate.readers.write(),
            _writer: self.writer,
        }
    }
}

impl MutationGate {
    pub(super) fn read(&self) -> RwLockReadGuard<'_, ()> {
        self.readers.read()
    }

    pub(super) fn write(&self) -> WriteGuard<'_> {
        self.begin_write().publish()
    }

    pub(super) fn begin_write(&self) -> TransactionGuard<'_> {
        TransactionGuard {
            gate: self,
            writer: self.writers.acquire(),
        }
    }

    pub(super) fn maintenance(&self) -> WriterLease<'_> {
        self.writers.acquire()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    #[test]
    fn maintenance_allows_reads_but_excludes_mutations_and_other_maintenance() {
        let gate = Arc::new(MutationGate::default());
        let maintenance = gate.maintenance();
        let (started_tx, started_rx) = mpsc::channel();
        let (written_tx, written_rx) = mpsc::channel();
        let writer_gate = Arc::clone(&gate);
        let writer = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let _write = writer_gate.write();
            written_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        let (read_tx, read_rx) = mpsc::channel();
        let reader_gate = Arc::clone(&gate);
        let reader = std::thread::spawn(move || {
            let _read = reader_gate.read();
            read_tx.send(()).unwrap();
        });
        let read = read_rx.recv_timeout(Duration::from_secs(5));
        assert!(written_rx.try_recv().is_err());
        drop(maintenance);
        writer.join().unwrap();
        reader.join().unwrap();
        read.expect("maintenance blocked a query");
        written_rx.recv().unwrap();
    }

    #[test]
    fn queries_progress_while_compaction_waits_and_a_mutation_is_queued() -> anyhow::Result<()> {
        use crate::{DocumentInput, HmsCore};
        let directory = tempfile::tempdir()?;
        let path = directory.path().display().to_string();
        let core = Arc::new(HmsCore::new(1024, Some(path.clone()), None)?);
        let vector = core.encode_text("reader remains available");
        core.memorize("original".into(), vector.clone())?;
        let documents = core.documents.write();
        let compact_core = Arc::clone(&core);
        let compactor = std::thread::spawn(move || compact_core.compact());
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let maintenance_active = || {
            let state = core.mutation_gate.writers.state.lock();
            state.serving != state.next
        };
        while !maintenance_active() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        let maintenance_started = maintenance_active();
        let writer_core = Arc::clone(&core);
        let writer = std::thread::spawn(move || {
            writer_core.memorize_document(serde_json::from_value::<DocumentInput>(
                serde_json::json!({"id":"after", "text":"durable after compaction"}),
            )?)
        });
        let reader_core = Arc::clone(&core);
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::spawn(move || tx.send(reader_core.query(&vector, 1)));
        let observation = rx.recv_timeout(Duration::from_secs(5));
        drop(documents);
        compactor.join().unwrap()?;
        writer.join().unwrap()?;
        reader.join().unwrap().unwrap();
        assert!(maintenance_started);
        assert_eq!(
            observation.expect("compaction blocked the reader")[0].id,
            "original"
        );
        drop(core);
        let restored = HmsCore::new(1024, Some(path), None)?;
        assert_eq!(
            restored.search_documents("durable", &Default::default())?[0].document_id,
            "after"
        );
        Ok(())
    }

    #[test]
    fn mutations_still_exclude_readers() {
        let gate = Arc::new(MutationGate::default());
        let write = gate.write();
        assert!(gate.readers.try_read().is_none());
        drop(write);
        assert!(gate.readers.try_read().is_some());
    }

    #[test]
    fn reservations_preserve_queue_order_through_publication_waits() {
        let gate = Arc::new(MutationGate::default());
        let maintenance = gate.maintenance();
        let read = gate.read();
        assert!(gate.writers.state.try_lock().is_some());
        let (tx, rx) = mpsc::channel();
        let mut workers = Vec::new();
        let mut registered = true;
        for id in 0..4 {
            let worker_gate = Arc::clone(&gate);
            let tx = tx.clone();
            workers.push(std::thread::spawn(move || {
                if id % 2 == 0 {
                    let _lease = worker_gate.maintenance();
                    tx.send(id).unwrap();
                } else {
                    let transaction = worker_gate.begin_write();
                    // `begin_write` must not hold the state mutex. The test thread polls that
                    // mutex while registering workers, so a single `try_lock` can observe its
                    // momentary hold; the invariant is that the lock becomes free while the
                    // transaction is open.
                    let deadline = std::time::Instant::now() + Duration::from_secs(5);
                    while worker_gate.writers.state.try_lock().is_none() {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "begin_write holds the state mutex"
                        );
                        std::thread::yield_now();
                    }
                    let _published = transaction.publish();
                    tx.send(id).unwrap();
                }
            }));
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while gate.writers.state.lock().next < id + 2 && std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
            registered &= gate.writers.state.lock().next == id + 2;
        }
        drop(maintenance);
        let first = rx.recv_timeout(Duration::from_secs(5));
        let blocked = rx.recv_timeout(Duration::from_millis(50));
        drop(read);
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(registered);
        assert_eq!(first.unwrap(), 0);
        assert!(matches!(blocked, Err(mpsc::RecvTimeoutError::Timeout)));
        assert_eq!(rx.try_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    /// A released lease hands the queue over at once, and the later drop advances nothing
    /// (otherwise a ticket would be skipped and the next writer would wait forever).
    #[test]
    fn released_lease_hands_the_queue_over_exactly_once() {
        let gate = Arc::new(MutationGate::default());
        let lease = gate.maintenance();
        let (sender, receiver) = mpsc::channel();
        let worker = {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || {
                let _write = gate.write();
                sender.send(()).unwrap();
            })
        };
        assert!(receiver.recv_timeout(Duration::from_millis(200)).is_err());
        lease.release();
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("writer admitted after release");
        worker.join().unwrap();
        lease.release();
        drop(lease);
        let state = gate.writers.state.lock();
        assert_eq!(state.serving, state.next, "queue advanced once per lease");
        drop(state);
        drop(gate.write());
    }

    #[test]
    fn reservation_counter_rejects_exhaustion_without_wrapping() {
        for next in [u64::MAX - 2, u64::MAX - 1, u64::MAX] {
            let gate = MutationGate::default();
            *gate.writers.state.lock() = WriterState {
                next,
                serving: next,
            };
            let lease =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gate.maintenance()));
            if next == u64::MAX {
                assert!(lease.is_err());
            } else {
                let lease = lease.unwrap();
                assert!(gate.writers.state.try_lock().is_some());
                drop(lease);
            }
        }
    }
}
