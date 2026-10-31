use serde::{Deserialize, Serialize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

#[allow(dead_code)]
#[path = "../../contrast/src/bm25.rs"]
mod bm25;
#[allow(dead_code)]
#[path = "../../contrast/src/hv.rs"]
mod hv;
#[allow(dead_code)]
#[path = "../../../../src/core/phase_hvec.rs"]
mod phase;

struct Meter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let p = unsafe { System.realloc(p, l, n) };
        if !p.is_null() {
            LIVE.fetch_add(n as isize - l.size() as isize, Ordering::Relaxed);
        }
        p
    }
}
#[global_allocator]
static ALLOC: Meter = Meter;
fn live() -> isize {
    LIVE.load(Ordering::Relaxed)
}
fn measured<T>(f: impl FnOnce() -> T) -> (T, f64, usize) {
    let before = live();
    let start = Instant::now();
    let value = f();
    (
        value,
        start.elapsed().as_secs_f64() * 1e6,
        (live() - before).max(0) as usize,
    )
}
type Triple = [u16; 3];
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
struct Doc {
    id: usize,
    tenant: u8,
    start: i32,
    end: i32,
    active: bool,
    version: u32,
    facts: Vec<Triple>,
    text: String,
    spans: Vec<(usize, usize)>,
}
fn render(d: &mut Doc) {
    d.text.clear();
    d.spans.clear();
    for f in &d.facts {
        let start = d.text.len();
        d.text
            .push_str(&format!("Café λ: E{} P{} E{}.\n", f[0], f[1], f[2]));
        d.spans.push((start, d.text.len()));
    }
}
fn corpus(seed: u64, n: usize, nf: usize) -> Vec<Doc> {
    let mut state = seed;
    let mut docs = Vec::with_capacity(n);
    for pair in 0..n / 2 {
        let a = (hv::splitmix64(&mut state) % 128) as u16;
        let b = (a + 1 + (hv::splitmix64(&mut state) % 127) as u16) % 128;
        let c = (hv::splitmix64(&mut state) % 128) as u16;
        let p = (hv::splitmix64(&mut state) % 16) as u16;
        let mut facts = vec![[a, p, b], [b, (p + 1) % 16, c]];
        for _ in 2..nf {
            facts.push([
                (hv::splitmix64(&mut state) % 128) as u16,
                (hv::splitmix64(&mut state) % 16) as u16,
                (hv::splitmix64(&mut state) % 128) as u16,
            ]);
        }
        for reverse in 0..2 {
            let mut d = Doc {
                id: pair * 2 + reverse,
                tenant: (pair % 2) as u8,
                start: 0,
                end: if pair % 7 == 0 { 50 } else { 100 },
                active: true,
                version: 1,
                facts: facts.clone(),
                text: String::new(),
                spans: Vec::new(),
            };
            if reverse == 1 {
                for f in &mut d.facts {
                    f.swap(0, 2);
                }
            }
            render(&mut d);
            docs.push(d);
        }
    }
    docs
}
#[derive(Clone, Serialize, Deserialize, Debug)]
struct Clause {
    mask: u8,
    value: Triple,
}
#[derive(Clone, Serialize, Deserialize, Debug)]
struct Query {
    class: usize,
    tenant: u8,
    time: i32,
    clauses: Vec<Clause>,
}
fn queries(ds: &[Doc]) -> Vec<Query> {
    (0..96)
        .map(|i| {
            let class = i % 6;
            let d = &ds[((i / 6) * 14 + 2) % ds.len()];
            let mut f = d.facts[0];
            if class == 4 {
                f.swap(0, 2);
            }
            if class == 5 {
                f[0] = 999;
            }
            let mask = match class {
                1 => 3,
                2 => 6,
                _ => 7,
            };
            let mut clauses = vec![Clause { mask, value: f }];
            if class == 3 {
                clauses.push(Clause {
                    mask: 7,
                    value: d.facts[1],
                });
            }
            Query {
                class,
                tenant: d.tenant,
                time: if i % 4 == 0 { 50 } else { 49 },
                clauses,
            }
        })
        .collect()
}
fn eligible(d: &Doc, q: &Query) -> bool {
    d.active && d.tenant == q.tenant && d.start <= q.time && q.time < d.end
}
fn matched(f: &Triple, c: &Clause) -> bool {
    (0..3).all(|r| c.mask & (1 << r) == 0 || c.value[r] == f[r])
}
#[derive(Debug, PartialEq, Eq)]
struct Hit {
    id: usize,
    version: u32,
    quotes: Vec<(usize, usize, String)>,
}
fn verify(d: &Doc, q: &Query) -> Option<Hit> {
    if !eligible(d, q) {
        return None;
    }
    let mut quotes = Vec::new();
    for c in &q.clauses {
        let i = d.facts.iter().position(|f| matched(f, c))?;
        let (a, b) = d.spans[i];
        quotes.push((a, b, d.text[a..b].to_owned()));
    }
    Some(Hit {
        id: d.id,
        version: d.version,
        quotes,
    })
}
fn oracle(ds: &[Doc], q: &Query) -> Vec<usize> {
    ds.iter()
        .filter(|d| {
            if !d.active || d.tenant != q.tenant || q.time < d.start || q.time >= d.end {
                return false;
            }
            for c in &q.clauses {
                let mut found = false;
                for f in &d.facts {
                    let mut ok = true;
                    for (r, v) in f.iter().enumerate() {
                        if c.mask & (1 << r) != 0 && *v != c.value[r] {
                            ok = false;
                        }
                    }
                    found |= ok;
                }
                if !found {
                    return false;
                }
            }
            true
        })
        .map(|d| d.id)
        .collect()
}
fn keys(d: &Doc) -> BTreeSet<(u8, u16)> {
    d.facts
        .iter()
        .flat_map(|f| (0..3).map(move |r| (r as u8, f[r])))
        .collect()
}
struct Exact {
    post: BTreeMap<(u8, u16), Vec<usize>>,
}
impl Exact {
    fn build(ds: &[Doc]) -> Self {
        let mut s = Self {
            post: BTreeMap::new(),
        };
        for d in ds {
            s.insert(d);
        }
        s
    }
    fn insert(&mut self, d: &Doc) {
        if !d.active {
            return;
        }
        for key in keys(d) {
            let v = self.post.entry(key).or_default();
            let p = v.partition_point(|x| *x < d.id);
            v.insert(p, d.id);
        }
    }
    fn remove(&mut self, d: &Doc) {
        for key in keys(d) {
            if let Some(v) = self.post.get_mut(&key) {
                if let Ok(p) = v.binary_search(&d.id) {
                    v.remove(p);
                }
            }
        }
    }
    fn search(&self, ds: &[Doc], q: &Query) -> Vec<Hit> {
        let mut lists = Vec::new();
        for c in &q.clauses {
            for r in 0..3 {
                if c.mask & (1 << r) != 0 {
                    match self.post.get(&(r as u8, c.value[r])) {
                        Some(v) => lists.push(v),
                        None => return Vec::new(),
                    }
                }
            }
        }
        lists.sort_by_key(|l| l.len());
        if lists.is_empty() {
            return Vec::new();
        }
        lists[0]
            .iter()
            .filter(|id| lists[1..].iter().all(|l| l.binary_search(id).is_ok()))
            .filter_map(|id| verify(&ds[*id], q))
            .take(5)
            .collect()
    }
}
fn pack(v: &phase::PhaseHVec) -> hv::BitVec {
    let mut out = [0; hv::WORDS];
    for (i, p) in v.phases().iter().enumerate() {
        out[i / 64] |= (*p as u64) << (i % 64);
    }
    out
}
fn atom(r: u8, v: u16) -> phase::PhaseHVec {
    phase::PhaseHVec::new_random(hv::DIM, 2, 0x31415926 ^ (r as u64) << 32 ^ v as u64)
}
struct Vsa {
    code: BTreeMap<(u8, u16), phase::PhaseHVec>,
    rows: Vec<[hv::BitVec; 7]>,
}
impl Vsa {
    fn new() -> Self {
        Self {
            code: BTreeMap::new(),
            rows: Vec::new(),
        }
    }
    fn projection(&self, c: &Clause) -> phase::PhaseHVec {
        let mut out = phase::PhaseHVec::from_phases(vec![0; hv::DIM], 2);
        for r in 0..3 {
            if c.mask & (1 << r) != 0 {
                let missing;
                let v = if let Some(v) = self.code.get(&(r as u8, c.value[r])) {
                    v
                } else {
                    missing = atom(r as u8, c.value[r]);
                    &missing
                };
                out = out.bind(v);
            }
        }
        out
    }
    fn encode(&mut self, d: &Doc) -> [hv::BitVec; 7] {
        for (r, v) in keys(d) {
            self.code.entry((r, v)).or_insert_with(|| atom(r, v));
        }
        std::array::from_fn(|m| {
            let ps: Vec<_> = d
                .facts
                .iter()
                .map(|f| {
                    self.projection(&Clause {
                        mask: (m + 1) as u8,
                        value: *f,
                    })
                })
                .collect();
            pack(&phase::PhaseHVec::bundle(&ps))
        })
    }
    fn build(ds: &[Doc]) -> Self {
        let mut s = Self::new();
        s.rows = Vec::with_capacity(ds.len());
        for d in ds {
            let row = s.encode(d);
            s.rows.push(row);
        }
        s
    }
    fn search(&self, ds: &[Doc], q: &Query, limit: usize) -> Vec<Hit> {
        let qs: Vec<_> = q
            .clauses
            .iter()
            .map(|c| (c.mask as usize - 1, pack(&self.projection(c))))
            .collect();
        let mut ranked: Vec<_> = ds
            .iter()
            .filter(|d| eligible(d, q))
            .map(|d| {
                let score: i32 = qs
                    .iter()
                    .map(|(m, v)| hv::dot_bits(&self.rows[d.id][*m], v))
                    .sum();
                (score, d.id)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        ranked
            .iter()
            .take(limit)
            .filter_map(|(_, id)| verify(&ds[*id], q))
            .take(5)
            .collect()
    }
}
fn token(r: usize, v: u16) -> u64 {
    v as u64 + if r == 1 { 10000 } else { 0 }
}
fn tokens(d: &Doc) -> Vec<u64> {
    d.facts
        .iter()
        .flat_map(|f| (0..3).map(move |r| token(r, f[r])))
        .collect()
}
fn bm_build(ds: &[Doc]) -> bm25::Bm25Index {
    bm25::Bm25Index::build(
        &ds.iter()
            .map(|d| if d.active { tokens(d) } else { Vec::new() })
            .collect::<Vec<_>>(),
    )
}
fn bm_search(b: &bm25::Bm25Index, ds: &[Doc], q: &Query) -> Vec<Hit> {
    let ts: Vec<_> = q
        .clauses
        .iter()
        .flat_map(|c| {
            (0..3)
                .filter(|r| c.mask & (1 << r) != 0)
                .map(|r| token(r, c.value[r]))
        })
        .collect();
    let scores = b.score_all(&bm25::counts(&ts));
    let mut ranked: Vec<_> = ds
        .iter()
        .filter(|d| eligible(d, q))
        .map(|d| (scores[d.id], d.id))
        .collect();
    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked
        .iter()
        .take(32)
        .filter_map(|(_, id)| verify(&ds[*id], q))
        .take(5)
        .collect()
}
const NAMES: [&str; 5] = [
    "exact_role_postings",
    "bm25_verified",
    "vsa_verified_32",
    "vsa_verified_128",
    "vsa_verified_32_fallback",
];
fn search(
    m: usize,
    e: &Exact,
    v: &Vsa,
    b: &bm25::Bm25Index,
    ds: &[Doc],
    q: &Query,
) -> (Vec<Hit>, bool) {
    match m {
        0 => (e.search(ds, q), false),
        1 => (bm_search(b, ds, q), false),
        2 => (v.search(ds, q, 32), false),
        3 => (v.search(ds, q, 128), false),
        _ => {
            let h = v.search(ds, q, 32);
            if h.len() < 5 {
                (e.search(ds, q), true)
            } else {
                (h, false)
            }
        }
    }
}
fn percentile(v: &[f64], p: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}
#[derive(Serialize)]
struct Metrics {
    method: &'static str,
    recall: f64,
    by_class: Vec<Option<f64>>,
    positive_queries: usize,
    empty_queries: usize,
    safety_errors: usize,
    fallback_fraction: f64,
    p50_us: f64,
    p95_us: f64,
    repeat_p50_us: Vec<f64>,
}
fn evaluate(e: &Exact, v: &Vsa, b: &bm25::Bm25Index, ds: &[Doc], qs: &[Query]) -> Vec<Metrics> {
    let truths: Vec<_> = qs.iter().map(|q| oracle(ds, q)).collect();
    let mut times = vec![vec![Vec::<f64>::new(); 3]; 5];
    for rep in 0..4 {
        let order: Vec<usize> = if rep % 2 == 0 {
            (0..5).collect()
        } else {
            (0..5).rev().collect()
        };
        for m in order {
            for q in qs {
                let start = Instant::now();
                black_box(search(m, e, v, b, ds, q));
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                if rep > 0 {
                    times[m][rep - 1].push(elapsed);
                }
            }
        }
    }
    (0..5)
        .map(|m| {
            let mut sums = [0.; 6];
            let mut counts = [0; 6];
            let mut errors = 0;
            let mut fallbacks = 0;
            for (q, truth) in qs.iter().zip(&truths) {
                let (h, fb) = search(m, e, v, b, ds, q);
                fallbacks += fb as usize;
                let mut seen = BTreeSet::new();
                for hit in &h {
                    if !truth.contains(&hit.id) || !seen.insert(hit.id) {
                        errors += 1;
                    }
                    let d = &ds[hit.id];
                    if hit.version != d.version || hit.quotes.len() != q.clauses.len() {
                        errors += 1;
                    }
                    for (j, (a, z, quote)) in hit.quotes.iter().enumerate() {
                        let valid = d.spans.iter().position(|span| span == &(*a, *z));
                        if d.text.get(*a..*z) != Some(quote.as_str())
                            || valid.is_none_or(|i| !matched(&d.facts[i], &q.clauses[j]))
                        {
                            errors += 1;
                        }
                    }
                }
                if !truth.is_empty() {
                    sums[q.class] += h.iter().filter(|h| truth.contains(&h.id)).count() as f64
                        / truth.len().min(5) as f64;
                    counts[q.class] += 1;
                }
            }
            let total: usize = counts.iter().sum();
            let flat: Vec<_> = times[m].iter().flatten().copied().collect();
            Metrics {
                method: NAMES[m],
                recall: sums.iter().sum::<f64>() / total as f64,
                by_class: sums
                    .iter()
                    .zip(counts)
                    .map(|(s, n)| if n == 0 { None } else { Some(s / n as f64) })
                    .collect(),
                positive_queries: total,
                empty_queries: qs.len() - total,
                safety_errors: errors,
                fallback_fraction: fallbacks as f64 / qs.len() as f64,
                p50_us: percentile(&flat, 0.5),
                p95_us: percentile(&flat, 0.95),
                repeat_p50_us: times[m].iter().map(|t| percentile(t, 0.5)).collect(),
            }
        })
        .collect()
}
#[derive(Serialize)]
struct Case {
    seed: u64,
    documents: usize,
    facts_per_document: usize,
    canonical_bytes: usize,
    index_bytes: Vec<usize>,
    mutated_canonical_bytes: usize,
    mutated_index_bytes: Vec<usize>,
    canonical_create_us: f64,
    canonical_update_us: f64,
    build_us: Vec<f64>,
    update_us: Vec<f64>,
    reopen_build_us: Vec<f64>,
    snapshot_write_us: f64,
    snapshot_read_us: f64,
    snapshot_bytes: usize,
    fixture_fnv64: String,
    initial: Vec<Metrics>,
    mutated: Vec<Metrics>,
    reopen_parity: bool,
}
fn arms<T: Copy + std::ops::Add<Output = T>>(e: T, b: T, v: T) -> Vec<T> {
    vec![e, b, v, v, v + e]
}
fn run_case(seed: u64, n: usize, nf: usize, out: &std::path::Path) -> Case {
    let (mut ds, canonical_create_us, canonical_bytes) = measured(|| corpus(seed, n, nf));
    let qs = queries(&ds);
    let fixture = serde_json::to_vec(&(&ds, &qs)).unwrap();
    let fixture_fnv64 = format!("{:016x}", hv::fnv1a64(&fixture));
    std::fs::write(out.join(format!("fixture-{seed}-{n}-{nf}.json")), &fixture).unwrap();
    drop(fixture);
    let (mut e, et, eb) = measured(|| Exact::build(&ds));
    let (mut b, bt, bb) = measured(|| bm_build(&ds));
    let (mut v, vt, vb) = measured(|| Vsa::build(&ds));
    let initial = evaluate(&e, &v, &b, &ds, &qs);
    let mut eu = 0.;
    let mut vu = 0.;
    let mut canonical_update_us = 0.;
    let mut em = eb as isize;
    let mut vm = vb as isize;
    for i in 0..32 {
        let start = Instant::now();
        let mut new = ds[i].clone();
        new.version += 1;
        if i < 16 {
            new.facts[0][0] = 500 + i as u16;
            render(&mut new);
        } else {
            new.active = false;
        }
        canonical_update_us += start.elapsed().as_secs_f64() * 1e6;
        let before = live();
        let start = Instant::now();
        e.remove(&ds[i]);
        e.insert(&new);
        eu += start.elapsed().as_secs_f64() * 1e6;
        em += live() - before;
        let before = live();
        let start = Instant::now();
        if new.active {
            v.rows[i] = v.encode(&new);
        } else {
            v.rows[i] = [[0; hv::WORDS]; 7];
        }
        vu += start.elapsed().as_secs_f64() * 1e6;
        vm += live() - before;
        ds[i] = new;
    }
    let before = live();
    let start = Instant::now();
    b = bm_build(&ds);
    let bu = start.elapsed().as_secs_f64() * 1e6;
    let bm = bb as isize + live() - before;
    let mutated_canonical_bytes = ds.capacity() * std::mem::size_of::<Doc>()
        + ds.iter()
            .map(|d| {
                d.facts.capacity() * std::mem::size_of::<Triple>()
                    + d.text.capacity()
                    + d.spans.capacity() * std::mem::size_of::<(usize, usize)>()
            })
            .sum::<usize>();
    let mutated = evaluate(&e, &v, &b, &ds, &qs);
    let snapshot = out.join(format!("snapshot-{seed}-{n}-{nf}.json"));
    let start = Instant::now();
    let data = serde_json::to_vec(&ds).unwrap();
    std::fs::write(&snapshot, &data).unwrap();
    let snapshot_write_us = start.elapsed().as_secs_f64() * 1e6;
    let snapshot_bytes = data.len();
    drop(data);
    let start = Instant::now();
    let reopened: Vec<Doc> = serde_json::from_slice(&std::fs::read(snapshot).unwrap()).unwrap();
    let snapshot_read_us = start.elapsed().as_secs_f64() * 1e6;
    assert_eq!(ds, reopened);
    let (er, ert, _) = measured(|| Exact::build(&reopened));
    let (br, brt, _) = measured(|| bm_build(&reopened));
    let (vr, vrt, _) = measured(|| Vsa::build(&reopened));
    let mut parity = true;
    for q in &qs {
        for m in 0..5 {
            parity &= search(m, &e, &v, &b, &ds, q) == search(m, &er, &vr, &br, &reopened, q);
        }
    }
    assert!(parity);
    assert_eq!(initial[0].recall, 1.);
    assert_eq!(mutated[0].recall, 1.);
    assert!(initial.iter().chain(&mutated).all(|m| m.safety_errors == 0));
    Case {
        seed,
        documents: n,
        facts_per_document: nf,
        canonical_bytes,
        index_bytes: arms(eb, bb, vb),
        mutated_canonical_bytes,
        mutated_index_bytes: arms(em as usize, bm as usize, vm as usize),
        canonical_create_us,
        canonical_update_us,
        build_us: arms(et, bt, vt),
        update_us: arms(eu, bu, vu),
        reopen_build_us: arms(ert, brt, vrt),
        snapshot_write_us,
        snapshot_read_us,
        snapshot_bytes,
        fixture_fnv64,
        initial,
        mutated,
        reopen_parity: parity,
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let dev = args.iter().any(|a| a == "--dev");
    let out = std::path::PathBuf::from(args.last().expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    let seeds: &[u64] = if dev { &[11] } else { &[101, 211, 307] };
    let sizes: &[usize] = if dev { &[256] } else { &[256, 1024] };
    let mut results = Vec::new();
    for &seed in seeds {
        for &n in sizes {
            for nf in [2, 8, 32] {
                eprintln!("case seed={seed} documents={n} facts={nf}");
                results.push(run_case(seed, n, nf, &out));
                std::fs::write(
                    out.join("results.json"),
                    serde_json::to_vec_pretty(&results).unwrap(),
                )
                .unwrap();
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn candidate_limits_and_replacement() {
        let mut ds = corpus(11, 256, 2);
        let mut v = Vsa::build(&ds);
        ds[2].facts[0] = [500, 3, 501];
        ds[2].version += 1;
        render(&mut ds[2]);
        v.rows[2] = v.encode(&ds[2]);
        let q = Query {
            class: 0,
            tenant: ds[2].tenant,
            time: 1,
            clauses: vec![Clause {
                mask: 7,
                value: ds[2].facts[0],
            }],
        };
        for limit in [31, 32, 33, 127, 128, 129] {
            let hits = v.search(&ds, &q, limit);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].id, 2);
            assert_eq!(hits[0].version, 2);
            assert_eq!(hits[0].quotes[0].2, "Café λ: E500 P3 E501.\n");
        }
        ds[2].active = false;
        assert!(v.search(&ds, &q, 128).is_empty());
    }
    #[test]
    fn parity_and_binding() {
        let a = atom(0, 4);
        let b = atom(2, 4);
        assert!(
            (a.similarity(&b) - hv::dot_bits(&pack(&a), &pack(&b)) as f64 / hv::DIM as f64).abs()
                < 1e-12
        );
        assert_eq!(a.bind(&b).unbind(&b).phases(), a.phases());
        let bundle = phase::PhaseHVec::bundle(&[a.clone(), b]);
        assert!(
            (bundle.similarity(&a)
                - hv::dot_bits(&pack(&bundle), &pack(&a)) as f64 / hv::DIM as f64)
                .abs()
                < 1e-12
        );
    }
    #[test]
    fn exact_oracle_and_lifecycle() {
        let mut ds = corpus(11, 256, 8);
        let mut e = Exact::build(&ds);
        for q in queries(&ds) {
            let truth = oracle(&ds, &q);
            let hits = e.search(&ds, &q);
            assert_eq!(
                hits.iter().map(|h| h.id).collect::<Vec<_>>(),
                truth.into_iter().take(5).collect::<Vec<_>>()
            );
        }
        let old = ds[2].clone();
        e.remove(&old);
        ds[2].active = false;
        for q in queries(&ds) {
            assert!(e.search(&ds, &q).iter().all(|h| h.id != 2));
        }
        let reopened: Vec<Doc> = serde_json::from_slice(&serde_json::to_vec(&ds).unwrap()).unwrap();
        let e2 = Exact::build(&reopened);
        for q in queries(&ds) {
            assert_eq!(e.search(&ds, &q), e2.search(&reopened, &q));
        }
    }
    #[test]
    fn role_source_and_boundaries() {
        let mut ds = corpus(11, 2, 2);
        ds[0].facts = vec![[1, 2, 3], [3, 4, 5]];
        ds[1].facts = vec![[3, 2, 1], [5, 4, 3]];
        for d in &mut ds {
            render(d);
        }
        let mut a = tokens(&ds[0]);
        a.sort();
        let mut b = tokens(&ds[1]);
        b.sort();
        assert_eq!(a, b);
        let mut q = Query {
            class: 0,
            tenant: 0,
            time: 49,
            clauses: vec![Clause {
                mask: 7,
                value: [1, 2, 3],
            }],
        };
        assert!(verify(&ds[0], &q).is_some());
        assert!(verify(&ds[1], &q).is_none());
        for time in [-1, 0, 1, 49, 50, 51] {
            q.time = time;
            assert_eq!(verify(&ds[0], &q).is_some(), (0..50).contains(&time));
        }
        q.time = 1;
        q.tenant = 1;
        assert!(verify(&ds[0], &q).is_none());
        q.tenant = 0;
        q.clauses[0].value = [1, 4, 5];
        assert!(verify(&ds[0], &q).is_none());
    }
}
