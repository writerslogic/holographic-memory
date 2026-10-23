// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! A consolidation that keeps the previous base graph: deleted vertices are spliced out of
//! the lists that pointed at them, and the delta rows are linked in by the build's insertion
//! pass, so the cost follows the rows that changed instead of every live row.

use std::cell::RefCell;

use rayon::prelude::*;

use super::super::{
    adjacency::Adjacency, kernels::SplitMix, medoid, norms_opt, prof, upper_layers, BuildCodes,
    BuildParams, GraphStructure, Rows, VGraph, BUILD_CODES_MIN_DIM, FIRST_PASS_EF, LAYER_RATIO,
    NONE,
};
use crate::core::error::HmsError;

/// Where each row of the merged generation comes from.
pub(super) struct MergePlan<'a> {
    /// The previous base graph, stored without renumbering (stored id = base slot).
    pub old: &'a VGraph,
    /// Per new slot: its previous base slot, or `NONE` for a row from the delta.
    pub new_to_old: Vec<u32>,
    /// Per previous base slot: its new slot, or `NONE` when the row was deleted.
    pub old_to_new: Vec<u32>,
}

#[derive(Default)]
struct RepairScratch {
    old: Vec<u32>,
    hop: Vec<u32>,
    ids: Vec<u32>,
    d: Vec<f32>,
    cands: Vec<(f32, u32)>,
}

thread_local! { static REPAIR: RefCell<RepairScratch> = RefCell::default(); }

/// The merged graph over the prepared rows `vectors` (with `norms`, empty for angular).
pub(super) fn merge_structure(
    vectors: &[f32],
    norms: &[f32],
    dim: usize,
    params: &BuildParams,
    plan: &MergePlan<'_>,
) -> Result<GraphStructure, HmsError> {
    let n = vectors.len() / dim;
    let r = params.degree;
    if let Some(at) = norms.iter().position(|v| !v.is_finite()) {
        return Err(super::super::invalid(format!(
            "row {at} overflows its squared norm"
        )));
    }
    let codes = (params.quantized_search || dim >= BUILD_CODES_MIN_DIM)
        .then(|| BuildCodes::new(vectors, dim, norms_opt(norms)));
    let rows = Rows {
        data: vectors,
        dim,
        norms: norms_opt(norms),
        codes: codes.as_ref(),
        quantized: params.quantized_search,
    };
    let t = prof::start();
    let mut adj = Adjacency::new(n, r);
    // FreshDiskANN's delete consolidation: a list that lost a deleted neighbour takes that
    // neighbour's live out-neighbours as candidates and is pruned again; other lists are
    // renumbered as they are.
    adj.par_set(n, |s, out| {
        let o = plan.new_to_old[s];
        if o == NONE {
            return;
        }
        REPAIR.with(|cell| {
            let RepairScratch {
                old,
                hop,
                ids,
                d,
                cands,
            } = &mut *cell.borrow_mut();
            old.clear();
            plan.old.neighbours_into(o as usize, old);
            if old.iter().all(|&u| plan.old_to_new[u as usize] != NONE) {
                out.extend(old.iter().map(|&u| plan.old_to_new[u as usize]));
                return;
            }
            ids.clear();
            for &u in old.iter() {
                let mapped = plan.old_to_new[u as usize];
                if mapped != NONE {
                    ids.push(mapped);
                    continue;
                }
                hop.clear();
                plan.old.neighbours_into(u as usize, hop);
                ids.extend(
                    hop.iter()
                        .map(|&w| plan.old_to_new[w as usize])
                        .filter(|&w| w != NONE && w as usize != s),
                );
            }
            ids.sort_unstable();
            ids.dedup();
            rows.d2_stream(s as u32, ids, d);
            cands.clear();
            cands.extend(d.iter().copied().zip(ids.iter().copied()));
            rows.prune_into(s as u32, cands, params.alpha, r, out);
        });
    });
    prof::report("merge-repair", t);
    let inserted: Vec<u32> = (0..n as u32)
        .filter(|&s| plan.new_to_old[s as usize] == NONE)
        .collect();
    let base_entry = plan.old_to_new[plan.old.entry_vertex() as usize];
    let start = if base_entry != NONE {
        base_entry
    } else {
        // The previous entry was deleted: start from the live row nearest the mean.
        let live: Vec<u32> = (0..n as u32)
            .filter(|&s| plan.new_to_old[s as usize] != NONE)
            .collect();
        let sub: Vec<f32> = live
            .iter()
            .flat_map(|&s| rows.row(s).iter().copied())
            .collect();
        let sub_norms: Vec<f32> = rows
            .norms
            .map(|n| live.iter().map(|&s| n[s as usize]).collect())
            .unwrap_or_default();
        let sub_rows = Rows {
            data: &sub,
            dim,
            norms: norms_opt(&sub_norms),
            codes: None,
            quantized: false,
        };
        match live.get(medoid(&sub_rows, live.len(), params.metric) as usize) {
            Some(&s) => s,
            None => return Err(super::super::invalid("merge without live base rows".into())),
        }
    };
    let mut order = inserted;
    shuffle(&mut order, params.seed ^ n as u64);
    let t = prof::start();
    rows.insert_pass(
        &mut adj,
        &order,
        start,
        params.build_ef.min(FIRST_PASS_EF),
        1.0,
        r,
    );
    prof::report("merge-insert", t);
    // The build's second pass, over the delta rows and the base rows whose neighbourhood
    // changed: those the delta now links to and those that lost a deleted neighbour. A full
    // rebuild searches again for every row; here only these can have better neighbours.
    let mut again = vec![false; n];
    for &p in &order {
        again[p as usize] = true;
        for &u in adj.get(p as usize) {
            again[u as usize] = true;
        }
    }
    let repaired: Vec<usize> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |list, s| {
            let o = plan.new_to_old[s];
            list.clear();
            if o != NONE {
                plan.old.neighbours_into(o as usize, list);
            }
            let dead = list.iter().any(|&u| plan.old_to_new[u as usize] == NONE);
            (s, dead)
        })
        .filter_map(|(s, dead)| dead.then_some(s))
        .collect();
    for s in repaired {
        again[s] = true;
    }
    // The previous graph is not read again: drop its pages before the passes allocate.
    plan.old
        .release_pages()
        .map_err(|e| super::super::invalid(format!("release previous base pages: {e}")))?;
    let mut order: Vec<u32> = (0..n as u32).filter(|&s| again[s as usize]).collect();
    drop(again);
    shuffle(&mut order, params.seed ^ n as u64 ^ 1);
    let t = prof::start();
    rows.insert_pass(&mut adj, &order, start, params.build_ef, params.alpha, r);
    prof::report("merge-relink", t);
    let t = prof::start();
    rows.fill(&mut adj, r);
    prof::report("fill", t);
    let t = prof::start();
    let entry = medoid(&rows, n, params.metric);
    let mut order: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut order, params.seed ^ 0xA5A5_A5A5 ^ n as u64);
    let layers = upper_layers(&rows, &order, params);
    order.truncate(n / LAYER_RATIO);
    prof::report("upper", t);
    Ok(GraphStructure {
        adj,
        entry,
        upper_ids: order,
        layers,
    })
}

fn shuffle(order: &mut [u32], seed: u64) {
    let mut rng = SplitMix(seed);
    for i in (1..order.len()).rev() {
        order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
    }
}

/// Slot maps for a merge: `live_base[slot]` says whether a previous base row survives, and
/// `ids` are the merged generation's sorted ids, of which `base_ids` (sorted, one per previous
/// slot) are a subset where live.
pub(super) fn slot_maps(
    base_ids: &[u32],
    live_base: impl Fn(usize) -> bool + Sync,
    ids: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let mut new_to_old = vec![NONE; ids.len()];
    let mut old_to_new = vec![NONE; base_ids.len()];
    let mut at = 0;
    for (slot, &id) in base_ids.iter().enumerate() {
        if !live_base(slot) {
            continue;
        }
        at += ids[at..].partition_point(|&x| x < id);
        debug_assert_eq!(ids[at], id);
        new_to_old[at] = slot as u32;
        old_to_new[slot] = at as u32;
    }
    (new_to_old, old_to_new)
}
