// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Write};
use std::sync::Arc;

#[derive(Clone)]
enum Backing {
    Owned(Vec<f32>),
    Mapped(Arc<memmap2::Mmap>),
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
            Backing::Mapped(map) => bytemuck::cast_slice(map.as_ref()),
        }
    }
}

impl DeltaRows {
    pub(super) fn extend_from_slice(&mut self, values: &[f32]) {
        let backing = Arc::make_mut(&mut self.0);
        if let Backing::Mapped(map) = backing {
            *backing = Backing::Owned(bytemuck::cast_slice::<u8, f32>(map.as_ref()).to_vec());
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
        let mut file = tempfile::tempfile()?;
        file.write_all(bytemuck::cast_slice(self))?;
        // SAFETY: the unlinked scratch file is exclusively owned here, fully written,
        // and its only writable handle is dropped before the mapping is published.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        #[cfg(unix)]
        map.advise(memmap2::Advice::Random)?;
        Ok(Self(Arc::new(Backing::Mapped(Arc::new(map)))))
    }

    pub(super) fn release_pages(&self) -> io::Result<()> {
        if let Backing::Mapped(map) = self.0.as_ref() {
            #[cfg(target_os = "linux")]
            // SAFETY: the unlinked scratch file has no writable handles and its bytes
            // remain unchanged when a concurrent reader faults discarded pages back in.
            unsafe {
                map.unchecked_advise(memmap2::UncheckedAdvice::DontNeed)?;
            }
            #[cfg(not(target_os = "linux"))]
            let _ = map;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
