use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;
use anyhow::Result;

impl HmsCore {
    pub fn memorize_triplet(&self, id: String, h: String, r: String, t: String) -> Result<()> {
        let vec_h = self.encode_text(&h);
        let vec_r = self.encode_text(&r);
        let vec_t = self.encode_text(&t);
        let triplet = vec_h.bind(&vec_r).bind(&vec_t);
        self.memorize(id, triplet)
    }

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

    pub fn query_triplet(&self, h: String, r: String, k: u32) -> Result<Vec<RetrievalResult>> {
        let vec_h = self.encode_text(&h);
        let vec_r = self.encode_text(&r);
        let query_vec = vec_h.bind(&vec_r);
        Ok(self.query(&query_vec, k))
    }

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
