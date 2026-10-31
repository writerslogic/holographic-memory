// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Write};
use std::sync::Arc;

#[derive(Clone)]
enum Backing {
    Owned(Vec<f32>),
    Mapped(Arc<MappedRows>),
}

struct MappedRows {
    // Drop the mapping before removing its Windows directory and delete-on-close file.
    map: memmap2::Mmap,
    #[cfg(windows)]
    _directory: tempfile::TempDir,
}

#[derive(Clone)]
pub(super) struct DeltaRows(Arc<Backing>);

impl Default for DeltaRows {
    fn default() -> Self {
        Self(Arc::new(Backing::Owned(Vec::new())))
    }
}

impl std::ops::Deref for DeltaRows {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match self.0.as_ref() {
            Backing::Owned(rows) => rows,
            Backing::Mapped(rows) => bytemuck::cast_slice(rows.map.as_ref()),
        }
    }
}

impl DeltaRows {
    pub(super) fn extend_from_slice(&mut self, values: &[f32]) {
        let backing = Arc::make_mut(&mut self.0);
        if let Backing::Mapped(rows) = backing {
            *backing = Backing::Owned(bytemuck::cast_slice::<u8, f32>(rows.map.as_ref()).to_vec());
        }
        match backing {
            Backing::Owned(rows) => rows.extend_from_slice(values),
            Backing::Mapped(_) => unreachable!("mapped rows were made mutable"),
        }
    }

    pub(super) fn spill_source(&self) -> Option<Self> {
        matches!(self.0.as_ref(), Backing::Owned(rows) if !rows.is_empty()).then(|| self.clone())
    }

    pub(super) fn same_backing(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(super) fn mapped(&self) -> io::Result<Self> {
        let context = |operation, error: io::Error| {
            io::Error::new(
                error.kind(),
                format!("delta row spill {operation}: {error}"),
            )
        };
        #[cfg(windows)]
        let directory =
            tempfile::tempdir().map_err(|e| context("create temporary directory", e))?;
        #[cfg(windows)]
        let file = tempfile::tempfile_in(directory.path());
        #[cfg(not(windows))]
        let file = tempfile::tempfile();
        let mut file = file.map_err(|e| context("create temporary file", e))?;
        file.write_all(bytemuck::cast_slice(self))
            .map_err(|e| context("write temporary file", e))?;
        // SAFETY: this exclusively owned scratch file is fully written, and no
        // writable handle is published or used after creating the mapping.
        let map =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| context("map temporary file", e))?;
        #[cfg(unix)]
        map.advise(memmap2::Advice::Random)
            .map_err(|e| context("advise mapped rows", e))?;
        Ok(Self(Arc::new(Backing::Mapped(Arc::new(MappedRows {
            map,
            #[cfg(windows)]
            _directory: directory,
        })))))
    }

    pub(super) fn release_pages(&self) -> io::Result<()> {
        if let Backing::Mapped(rows) = self.0.as_ref() {
            #[cfg(target_os = "linux")]
            // SAFETY: the unlinked scratch file has no writable handles and its bytes
            // remain unchanged when a concurrent reader faults discarded pages back in.
            unsafe {
                rows.map
                    .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)?;
            }
            #[cfg(not(target_os = "linux"))]
            let _ = rows;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn spill_avoids_temp_directory_collisions() {
        const CHILD: &str = "WARM_SPILL_COLLISION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let test = format!(
                "{}::spill_avoids_temp_directory_collisions",
                module_path!().split_once("::").unwrap().1
            );
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &test])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(output.status.success(), "{output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        let root = tempfile::tempdir().unwrap();
        tempfile::env::override_temp_dir(root.path()).unwrap();
        fastrand::seed(20261030);
        let occupied = tempfile::tempdir().unwrap();
        let mut rows = DeltaRows::default();
        rows.extend_from_slice(&[-0.0, 1.0, 2.0]);
        // Replay the occupied directory's random name during spill creation.
        fastrand::seed(20261030);
        let mut mapped = rows.mapped().unwrap();
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&mapped),
            bytemuck::cast_slice::<f32, u8>(&rows)
        );
        let scratch = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path != occupied.path())
            .unwrap();
        assert!(scratch.is_dir());
        let pinned = mapped.clone();
        mapped.extend_from_slice(&[3.0]);
        assert_eq!(&*pinned, &*rows);
        assert_eq!(&*mapped, &[-0.0, 1.0, 2.0, 3.0]);
        assert!(scratch.is_dir());
        drop(pinned);
        assert!(!scratch.exists());
        assert!(occupied.path().is_dir());
    }

    #[cfg(windows)]
    #[test]
    fn concurrent_spill_lifetimes() {
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..512 {
                        spilled_rows_keep_bits_and_detach_on_append();
                    }
                });
            }
        });
    }

    #[test]
    fn spilled_rows_keep_bits_and_detach_on_append() {
        for len in [1, 1023, 1024, 1025] {
            let values: Vec<f32> = (0..len)
                .map(|i| if i % 2 == 0 { -0.0 } else { i as f32 / 7.0 })
                .collect();
            let mut rows = DeltaRows::default();
            rows.extend_from_slice(&values);
            let source = rows.spill_source().unwrap();
            let mut mapped = source.mapped().unwrap();
            assert!(mapped.spill_source().is_none());
            mapped.release_pages().unwrap();
            assert_eq!(
                bytemuck::cast_slice::<f32, u8>(&mapped),
                bytemuck::cast_slice::<f32, u8>(&values)
            );
            let previous = mapped.clone();
            mapped.extend_from_slice(&[13.0]);
            assert_eq!(&mapped[..len], &*previous);
            assert_eq!(mapped[len], 13.0);
            assert_eq!(previous.len(), len);
            assert!(mapped.spill_source().is_some());
        }
        assert!(DeltaRows::default().spill_source().is_none());
    }
}
