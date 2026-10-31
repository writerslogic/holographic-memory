// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::WarmCore;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;
use anyhow::Result;

impl WarmCore {
    /// Encode a (head, relation, tail) triplet as `h XOR r XOR t` and memorize it.
    pub fn memorize_triplet(&self, id: String, h: String, r: String, t: String) -> Result<()> {
        super::mutation::validate_public_id(&id)?;
        if !self.meaning_enabled() {
            let triplet = self
                .try_encode_text(&h)?
                .bind(&self.try_encode_text(&r)?)
                .bind(&self.try_encode_text(&t)?);
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
        self.memorize(id, self.encode_sequence(sequence)?)
    }

    /// Query for tails matching `head XOR relation`.
    pub fn query_triplet(&self, h: String, r: String, k: u32) -> Result<Vec<RetrievalResult>> {
        let vec_h = self.try_encode_text(&h)?;
        let vec_r = self.try_encode_text(&r)?;
        let query_vec = vec_h.bind(&vec_r);
        self.query(&query_vec, k)
    }

    /// Find analogy: A is to B as C is to ? Returns k closest matches.
    pub fn find_analogy(&self, a: &str, b: &str, c: &str, k: u32) -> Vec<RetrievalResult> {
        self.try_find_analogy(a, b, c, k).expect(
            "analogy encoding failed; use try_find_analogy to handle model or resource errors",
        )
    }

    pub fn try_find_analogy(
        &self,
        a: &str,
        b: &str,
        c: &str,
        k: u32,
    ) -> Result<Vec<RetrievalResult>> {
        let vec_a = self.try_encode_text(a)?;
        let vec_b = self.try_encode_text(b)?;
        let vec_c = self.try_encode_text(c)?;
        let target = vec_a.bind(&vec_b).bind(&vec_c);
        self.query(&target, k)
    }

    /// Query for sequences matching a partial sequence prefix.
    pub fn query_sequence(&self, partial: &[String], k: u32) -> Result<Vec<RetrievalResult>> {
        if partial.is_empty() {
            return Ok(vec![]);
        }
        self.query(&self.encode_sequence(partial)?, k)
    }

    /// Bundles `items`, each permuted by its position.
    fn encode_sequence(&self, items: &[String]) -> Result<EntangledHVec> {
        let texts: Vec<_> = items.iter().map(String::as_str).collect();
        let vecs: Vec<EntangledHVec> = self
            .try_encode_texts(&texts)?
            .into_iter()
            .enumerate()
            .map(|(i, vector)| vector.permute(i))
            .collect();
        Ok(self.bundle(&vecs))
    }
}
