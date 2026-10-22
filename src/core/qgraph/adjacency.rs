//! Construction-time adjacency: every list in one fixed-slot allocation.

use rayon::prelude::*;

/// Adjacency lists of at most `degree` ids each: `degree` slots per vertex in one flat array
/// plus a length per vertex. Construction rewrites lists millions of times from many threads;
/// fixed slots keep those rewrites from allocating, and the structure is freed as two blocks
/// that go straight back to the system instead of a heap fragmented by per-list buffers.
#[derive(Clone, Default)]
pub(crate) struct Adjacency {
    degree: usize,
    slots: Vec<u32>,
    lens: Vec<u32>,
}

impl Adjacency {
    /// `n` empty lists of capacity `degree`.
    pub(crate) fn new(n: usize, degree: usize) -> Self {
        Self {
            degree,
            slots: vec![0; n * degree],
            lens: vec![0; n],
        }
    }

    /// Lists from their lengths and their concatenation; `None` when the lengths do not
    /// account for `flat` exactly or a list is longer than `degree`.
    pub(crate) fn from_flat(lens: Vec<u32>, flat: &[u32], degree: usize) -> Option<Self> {
        if lens.iter().any(|&l| l as usize > degree)
            || lens.iter().map(|&l| l as usize).sum::<usize>() != flat.len()
        {
            return None;
        }
        let mut adj = Self::new(lens.len(), degree);
        let mut at = 0;
        for (v, &l) in lens.iter().enumerate() {
            adj.set(v, &flat[at..at + l as usize]);
            at += l as usize;
        }
        Some(adj)
    }

    /// Appends `list` as a new vertex; an empty `Adjacency` takes `degree` from this call.
    pub(crate) fn push(&mut self, list: &[u32], degree: usize) {
        if self.lens.is_empty() {
            self.degree = degree;
        }
        assert!(list.len() <= self.degree, "list longer than the degree");
        let at = self.slots.len();
        self.slots.resize(at + self.degree, 0);
        self.slots[at..at + list.len()].copy_from_slice(list);
        self.lens.push(list.len() as u32);
    }

    /// Number of vertices.
    pub(crate) fn len(&self) -> usize {
        self.lens.len()
    }

    /// The list of `v`.
    #[inline]
    pub(crate) fn get(&self, v: usize) -> &[u32] {
        let at = v * self.degree;
        &self.slots[at..at + self.lens[v] as usize]
    }

    /// The length word of `v`, for a prefetch ahead of [`Adjacency::get`].
    #[inline]
    pub(crate) fn len_word(&self, v: usize) -> &u32 {
        &self.lens[v]
    }

    /// Replaces the list of `v`.
    pub(crate) fn set(&mut self, v: usize, list: &[u32]) {
        assert!(list.len() <= self.degree, "list longer than the degree");
        let at = v * self.degree;
        self.slots[at..at + list.len()].copy_from_slice(list);
        self.lens[v] = list.len() as u32;
    }

    /// Every list in vertex order.
    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = &[u32]> + Clone + '_ {
        (0..self.len()).map(|v| self.get(v))
    }

    /// Sets lists `0..count` in parallel, list `i` to `f(i)`.
    pub(crate) fn par_set(&mut self, count: usize, f: impl Fn(usize) -> Vec<u32> + Sync) {
        let r = self.degree;
        self.slots[..count * r]
            .par_chunks_mut(r)
            .zip(self.lens[..count].par_iter_mut())
            .enumerate()
            .for_each(|(i, (dst, len))| {
                let list = f(i);
                assert!(list.len() <= r, "list longer than the degree");
                dst[..list.len()].copy_from_slice(&list);
                *len = list.len() as u32;
            });
    }

    /// Appends `extra[v]` to every list `v` in parallel, where `extra` sees the lists as they
    /// were before any append. Work and scratch go in blocks of vertices so the scratch stays
    /// bounded; the result equals computing every `extra` first and appending afterwards,
    /// since an append writes only past the old length, which `extra` never reads.
    pub(crate) fn par_append(&mut self, f: impl Fn(&Self, usize) -> Vec<u32> + Sync) {
        const BLOCK: usize = 1 << 15;
        let (n, r) = (self.len(), self.degree);
        let mut new_lens = self.lens.clone();
        let mut buf = vec![0u32; BLOCK.min(n) * r];
        let mut counts = vec![0u32; BLOCK.min(n)];
        for base in (0..n).step_by(BLOCK) {
            let size = BLOCK.min(n - base);
            let this = &*self;
            buf[..size * r]
                .par_chunks_mut(r)
                .zip(counts[..size].par_iter_mut())
                .enumerate()
                .for_each(|(i, (dst, count))| {
                    let extra = f(this, base + i);
                    let room = r - this.lens[base + i] as usize;
                    assert!(extra.len() <= room, "list longer than the degree");
                    dst[..extra.len()].copy_from_slice(&extra);
                    *count = extra.len() as u32;
                });
            for i in 0..size {
                let v = base + i;
                let (at, k) = (v * r + self.lens[v] as usize, counts[i] as usize);
                self.slots[at..at + k].copy_from_slice(&buf[i * r..i * r + k]);
                new_lens[v] += counts[i];
            }
        }
        self.lens = new_lens;
    }

    /// Mutable `(slots, length)` of each vertex in `ids`, which must be strictly increasing.
    pub(crate) fn rows_mut(
        &mut self,
        ids: impl Iterator<Item = usize>,
    ) -> impl Iterator<Item = (&mut [u32], &mut u32)> {
        let r = self.degree;
        let mut slots: &mut [u32] = &mut self.slots;
        let mut lens: &mut [u32] = &mut self.lens;
        let mut next = 0;
        ids.map(move |v| {
            assert!(v >= next, "ids must be strictly increasing");
            let row = std::mem::take(&mut slots)[(v - next) * r..].split_at_mut(r);
            slots = row.1;
            let len = std::mem::take(&mut lens)[v - next..]
                .split_first_mut()
                .expect("id in range");
            lens = len.1;
            next = v + 1;
            (row.0, len.0)
        })
    }

    /// The lists as separate vectors.
    pub(crate) fn to_lists(&self) -> Vec<Vec<u32>> {
        self.iter().map(<[u32]>::to_vec).collect()
    }
}

impl std::ops::Index<usize> for Adjacency {
    type Output = [u32];

    fn index(&self, v: usize) -> &[u32] {
        self.get(v)
    }
}

impl PartialEq for Adjacency {
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree && self.lens == other.lens && self.iter().eq(other.iter())
    }
}

/// Hashes as the equivalent `Vec<Vec<u32>>` does.
impl std::hash::Hash for Adjacency {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        self.iter().for_each(|list| list.hash(state));
    }
}

impl std::fmt::Debug for Adjacency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_mut_reaches_exactly_the_listed_vertices() {
        let mut a = Adjacency::new(6, 3);
        for (row, len) in a.rows_mut([0, 2, 5].into_iter()) {
            row[0] = 7;
            *len = 1;
        }
        let lists = a.to_lists();
        assert_eq!(
            lists,
            vec![vec![7], vec![], vec![7], vec![], vec![], vec![7]]
        );
    }

    #[test]
    fn par_append_reads_the_lists_from_before_any_append() {
        let n = (1 << 15) + 3;
        let mut a = Adjacency::new(n, 4);
        for v in 0..n {
            a.set(v, &[((v + 1) % n) as u32]);
        }
        // Every vertex appends its successor's list; a block that saw an earlier block's
        // appends would append two ids for some vertex.
        a.par_append(|g, v| g.get((v + 1) % n).to_vec());
        assert!((0..n).all(|v| a.get(v) == [((v + 1) % n) as u32, ((v + 2) % n) as u32]));
    }
}
