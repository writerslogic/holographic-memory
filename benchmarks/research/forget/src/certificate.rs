// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deletion certificates for [`CounterBundle`].
//!
//! # Mechanism
//!
//! A certificate commits to the counter state before the delete, the state
//! after, and the deleted item, all as SHA-256 digests, and records a residual
//! bound: the deleted item's overlap with the post-delete view, together with
//! the exact null distribution of that overlap for a random item of the same
//! weight. Verification needs only the certificate, the item and the
//! post-delete bundle: it recomputes `state_after`, then reproduces
//! `state_before` by digesting `after + item` in a streaming pass
//! ([`CounterBundle::digest_with_item_added`]). Because the counts form a
//! group under addition, `digest(after + item) == state_before` proves that
//! the transition was exactly the subtraction of that one item.
//!
//! # Residual bound
//!
//! After an exact delete the item's own contribution is gone; whatever overlap
//! remains comes from surviving items that happen to share positions. For an
//! item of weight `k` drawn independently of the survivors, the view overlap
//! is hypergeometric: `k` positions sampled without replacement from `dim`, of
//! which `view_size` are set. The certificate records that null's mean and
//! standard deviation and the item's z-score; `within_bound` is
//! `overlap <= mean + sigma * sd`. The unthresholded readout `count_dot` gets
//! the same treatment with the exact finite-population variance of the
//! counts. The bound is a statement about the item's own contribution, not
//! about near-duplicates: a surviving item that resembles the deleted one is
//! legitimately still in the memory, and an overlap above the bound is
//! evidence of such a survivor, not of an inexact delete (the hash chain is
//! the exactness proof).
//!
//! # Data layout
//!
//! Plain `serde` structs; digests serialise as 64 hex characters. One
//! certificate per delete; a chain of certificates is consistent when each
//! `state_after` equals the next `state_before`.

use writerslogic_warm::EntangledHVec;
use serde::{Deserialize, Serialize};

use crate::bundle::{hvec_digest, BundleError, Counter, CounterBundle, Digest32};

/// Certificate schema identifier.
pub const CERTIFICATE_SCHEMA: &str = "warm-forget/certificate-v1";
/// Default residual bound width in standard deviations.
pub const DEFAULT_SIGMA: f64 = 3.0;

/// Hypergeometric null of the view overlap for a random weight-`k` item.
pub fn view_null(dim: usize, view_size: usize, k: usize) -> (f64, f64) {
    if dim == 0 || k == 0 {
        return (0.0, 0.0);
    }
    let d = dim as f64;
    let p = view_size as f64 / d;
    let kk = k as f64;
    let fpc = if dim > 1 { (d - kk) / (d - 1.0) } else { 0.0 };
    let var = kk * p * (1.0 - p) * fpc;
    (kk * p, var.max(0.0).sqrt())
}

/// Finite-population null of `sum(counts[i])` over `k` random positions.
pub fn count_null(dim: usize, sum: f64, sum_sq: f64, k: usize) -> (f64, f64) {
    if dim == 0 || k == 0 {
        return (0.0, 0.0);
    }
    let d = dim as f64;
    let kk = k as f64;
    let mean_c = sum / d;
    let var_c = (sum_sq / d - mean_c * mean_c).max(0.0);
    let fpc = if dim > 1 { (d - kk) / (d - 1.0) } else { 0.0 };
    (kk * mean_c, (kk * var_c * fpc).sqrt())
}

fn z(x: f64, mean: f64, sd: f64) -> f64 {
    if sd > 0.0 {
        (x - mean) / sd
    } else {
        0.0
    }
}

/// Post-delete residual of the deleted item against the view and the counts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidualBound {
    pub active: usize,
    pub view_size: usize,
    pub view_density: f64,
    /// Indices of the item at or above `tau` in the post-delete view.
    pub view_overlap: usize,
    pub overlap_fraction: f64,
    /// `view_overlap / sqrt(active * view_size)`.
    pub cosine: f64,
    pub view_null_mean: f64,
    pub view_null_sd: f64,
    pub view_z: f64,
    /// `sum(counts[i])` over the item's indices.
    pub count_dot: i64,
    pub count_null_mean: f64,
    pub count_null_sd: f64,
    pub count_z: f64,
    pub sigma: f64,
    /// `view_overlap <= view_null_mean + sigma * view_null_sd` and the same for `count_dot`.
    pub within_bound: bool,
}

/// Residual of `item` against the current state of `bundle`.
pub fn residual<C: Counter>(bundle: &CounterBundle<C>, item: &EntangledHVec, sigma: f64) -> ResidualBound {
    let k = item.indices().len();
    let view_size = bundle.view_size();
    let view_overlap = bundle.view_overlap(item);
    let (vm, vs) = view_null(bundle.dim(), view_size, k);
    let count_dot = bundle.count_dot(item);
    let (sum, sum_sq) = bundle.count_moments();
    let (cm, cs) = count_null(bundle.dim(), sum, sum_sq, k);
    let denom = ((k * view_size) as f64).sqrt();
    let cosine = if denom > 0.0 {
        view_overlap as f64 / denom
    } else {
        0.0
    };
    let within_bound = (view_overlap as f64) <= vm + sigma * vs && (count_dot as f64) <= cm + sigma * cs;
    ResidualBound {
        active: k,
        view_size,
        view_density: bundle.view_density(),
        view_overlap,
        overlap_fraction: if k > 0 { view_overlap as f64 / k as f64 } else { 0.0 },
        cosine,
        view_null_mean: vm,
        view_null_sd: vs,
        view_z: z(view_overlap as f64, vm, vs),
        count_dot,
        count_null_mean: cm,
        count_null_sd: cs,
        count_z: z(count_dot as f64, cm, cs),
        sigma,
        within_bound,
    }
}

/// Proof-carrying record of one exact delete.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeletionCertificate {
    pub schema: String,
    pub dim: usize,
    pub tau: u32,
    pub counter_bits: u32,
    pub item: Digest32,
    pub state_before: Digest32,
    pub state_after: Digest32,
    pub view_after: Digest32,
    pub items_before: usize,
    pub items_after: usize,
    pub residual: ResidualBound,
}

/// Delete `item` and issue its certificate. Costs two state digests
/// (`O(dim)` each) on top of the `O(k)` delete.
pub fn delete_with_certificate<C: Counter>(
    bundle: &mut CounterBundle<C>,
    item: &EntangledHVec,
    sigma: f64,
) -> Result<DeletionCertificate, BundleError> {
    let state_before = bundle.state_digest();
    let items_before = bundle.items();
    let item_digest = bundle.delete(item)?;
    let state_after = bundle.state_digest();
    let residual = residual(bundle, item, sigma);
    Ok(DeletionCertificate {
        schema: CERTIFICATE_SCHEMA.to_string(),
        dim: bundle.dim(),
        tau: bundle.tau(),
        counter_bits: C::BITS,
        item: item_digest,
        state_before,
        state_after,
        view_after: bundle.view_digest(),
        items_before,
        items_after: bundle.items(),
        residual,
    })
}

/// Outcome of [`verify`]; `ok` is the conjunction of the structural checks.
/// `residual_recomputed_within_bound` is informational (see module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verification {
    pub schema_ok: bool,
    pub parameters_ok: bool,
    pub item_matches: bool,
    pub state_after_matches: bool,
    pub state_before_reproduced: bool,
    pub view_after_matches: bool,
    pub item_count_consistent: bool,
    pub residual_recomputed_within_bound: bool,
    pub ok: bool,
}

/// Check `cert` against the deleted `item` and the post-delete `after` bundle.
pub fn verify<C: Counter>(
    cert: &DeletionCertificate,
    item: &EntangledHVec,
    after: &CounterBundle<C>,
) -> Result<Verification, BundleError> {
    let schema_ok = cert.schema == CERTIFICATE_SCHEMA;
    let parameters_ok =
        cert.dim == after.dim() && cert.tau == after.tau() && cert.counter_bits == C::BITS;
    let item_matches = hvec_digest(after.dim(), item) == cert.item;
    let state_after_matches = after.state_digest() == cert.state_after;
    let state_before_reproduced = after.digest_with_item_added(item)? == cert.state_before;
    let view_after_matches = after.view_digest() == cert.view_after;
    let item_count_consistent =
        cert.items_after == after.items() && cert.items_before == cert.items_after + 1;
    let residual_recomputed_within_bound = residual(after, item, cert.residual.sigma).within_bound;
    let ok = schema_ok
        && parameters_ok
        && item_matches
        && state_after_matches
        && state_before_reproduced
        && view_after_matches
        && item_count_consistent;
    Ok(Verification {
        schema_ok,
        parameters_ok,
        item_matches,
        state_after_matches,
        state_before_reproduced,
        view_after_matches,
        item_count_consistent,
        residual_recomputed_within_bound,
        ok,
    })
}

/// A sequence of certificates is a chain when each `state_after` is the next
/// `state_before`. Returns the index of the first break, if any.
pub fn chain_break(certs: &[DeletionCertificate]) -> Option<usize> {
    certs
        .windows(2)
        .position(|w| w[0].state_after != w[1].state_before)
        .map(|i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::CapacityPolicy;

    fn item(dim: usize, seed: u64) -> EntangledHVec {
        EntangledHVec::new_deterministic(dim, seed)
    }

    #[test]
    fn certificate_verifies_against_post_state_and_rejects_a_wrong_item() {
        let dim = 4096;
        let items: Vec<_> = (0..64).map(|s| item(dim, 300 + s)).collect();
        let mut b = CounterBundle::<i16>::rebuild(dim, 1, CapacityPolicy::UNBOUNDED, items.iter()).unwrap();
        let before = b.state_digest();
        let cert = delete_with_certificate(&mut b, &items[5], DEFAULT_SIGMA).unwrap();
        assert_eq!(cert.state_before, before);
        assert_eq!(cert.items_before, 64);
        assert_eq!(cert.items_after, 63);
        let v = verify(&cert, &items[5], &b).unwrap();
        assert!(v.ok, "{v:?}");
        let wrong = verify(&cert, &items[6], &b).unwrap();
        assert!(!wrong.item_matches);
        assert!(!wrong.state_before_reproduced);
        assert!(!wrong.ok);
    }

    #[test]
    fn certificate_chain_is_consistent() {
        let dim = 4096;
        let items: Vec<_> = (0..32).map(|s| item(dim, 900 + s)).collect();
        let mut b = CounterBundle::<i32>::rebuild(dim, 1, CapacityPolicy::UNBOUNDED, items.iter()).unwrap();
        let certs: Vec<_> = items[..8]
            .iter()
            .map(|it| delete_with_certificate(&mut b, it, DEFAULT_SIGMA).unwrap())
            .collect();
        assert_eq!(chain_break(&certs), None);
        let mut broken = certs.clone();
        broken[3].state_before = broken[0].state_before;
        assert_eq!(chain_break(&broken), Some(3));
    }

    #[test]
    fn null_moments_are_sane() {
        let (m, s) = view_null(16384, 8192, 64);
        assert!((m - 32.0).abs() < 1e-9);
        assert!(s > 3.9 && s < 4.1, "sd {s}");
        let (m0, s0) = view_null(16384, 0, 64);
        assert_eq!((m0, s0), (0.0, 0.0));
        let (cm, cs) = count_null(16384, 16384.0, 16384.0, 64);
        assert!((cm - 64.0).abs() < 1e-9);
        assert_eq!(cs, 0.0);
    }
}
