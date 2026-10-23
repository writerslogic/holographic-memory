// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::HmsCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;
use anyhow::Result;

impl HmsCore {
    /// Encode a (head, relation, tail) triplet as `h XOR r XOR t` and memorize it.
    pub fn memorize_triplet(&self, id: String, h: String, r: String, t: String) -> Result<()> {
        super::mutation::validate_public_id(&id)?;
        if !self.meaning_enabled() {
            let triplet = self
                .encode_text(&h)
                .bind(&self.encode_text(&r))
                .bind(&self.encode_text(&t));
            return self.memorize(id, triplet);
        }
        self.commit(&[super::mutation::Mutation::Triplet {
            id,
            subject: h,
            relation: r,
            object: t,
        }])
    }

    /// Encode an ordered sequence using position-based permutation and memorize it.
    pub fn memorize_sequence(&self, id: String, sequence: &[String]) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        self.memorize(id, self.encode_sequence(sequence))
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
        Ok(self.query(&self.encode_sequence(partial), k))
    }

    /// Bundles `items`, each permuted by its position.
    fn encode_sequence(&self, items: &[String]) -> EntangledHVec {
        let vecs: Vec<EntangledHVec> = items
            .iter()
            .enumerate()
            .map(|(i, item)| self.encode_text(item).permute(i))
            .collect();
        self.bundle(&vecs)
    }
}
