use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;
use anyhow::Result;

impl HmsCore {
    /// Encode a (head, relation, tail) triplet as `h XOR r XOR t` and memorize it.
    pub fn memorize_triplet(&self, id: String, h: String, r: String, t: String) -> Result<()> {
        let vec_h = self.encode_text(&h);
        let vec_r = self.encode_text(&r);
        let vec_t = self.encode_text(&t);
        let triplet = vec_h.bind(&vec_r).bind(&vec_t);
        self.memorize(id, triplet)
    }

    /// Encode an ordered sequence using position-based permutation and memorize it.
    pub fn memorize_sequence(&self, id: String, sequence: &[String]) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        let mut vecs = Vec::new();
        for (i, item) in sequence.iter().enumerate() {
            let v = self.encode_text(item).permute(i);
            vecs.push(v);
        }
        let trajectory = EntangledHVec::bundle(&vecs);
        self.memorize(id, trajectory)
    }

    /// Query for tails matching `head XOR relation`.
    pub fn query_triplet(&self, h: String, r: String, k: u32) -> Result<Vec<RetrievalResult>> {
        let vec_h = self.encode_text(&h);
        let vec_r = self.encode_text(&r);
        let query_vec = vec_h.bind(&vec_r);
        Ok(self.query(&query_vec, k))
    }

    /// Find analogy: A is to B as C is to ? Returns k closest matches.
    pub fn find_analogy(&self, a: &str, b: &str, c: &str, k: u32) -> Vec<RetrievalResult> {
        let vec_a = self.encode_text(a);
        let vec_b = self.encode_text(b);
        let vec_c = self.encode_text(c);
        let target = vec_a.bind(&vec_b).bind(&vec_c);
        self.query(&target, k)
    }

    /// Query for sequences matching a partial sequence prefix.
    pub fn query_sequence(&self, partial: &[String], k: u32) -> Result<Vec<RetrievalResult>> {
        if partial.is_empty() {
            return Ok(vec![]);
        }
        let vecs: Vec<EntangledHVec> = partial
            .iter()
            .enumerate()
            .map(|(i, item)| self.encode_text(item).permute(i))
            .collect();
        let query_vec = EntangledHVec::bundle(&vecs);
        Ok(self.query(&query_vec, k))
    }
}
