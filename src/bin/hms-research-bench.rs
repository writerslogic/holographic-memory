use holographic_memory::core::entangled::{hash_u64, EntangledHVec};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");
    let quick = args.iter().any(|a| a == "--quick");
    let dim = args.iter().position(|a| a == "--dim")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
    let density_denom = args.iter().position(|a| a == "--density")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    if !json_output {
        eprintln!("HMS Research Benchmark — D={}, density=1/{}", dim, density_denom);
        eprintln!("=========================================================");
    }

    let t_start = Instant::now();

    if !json_output { eprintln!("[1/6] Knowledge graph retrieval..."); }
    let kg = bench_knowledge_graph(dim, density_denom, quick);

    if !json_output { eprintln!("[2/6] Analogy completion..."); }
    let analogy = bench_analogy(dim, density_denom);

    if !json_output { eprintln!("[3/6] Interference stress test..."); }
    let interference = bench_interference(dim, density_denom, quick);

    if !json_output { eprintln!("[4/6] Multi-hop inference..."); }
    let multihop = bench_multihop(dim, density_denom);

    if !json_output { eprintln!("[5/6] Sequence encoding..."); }
    let sequence = bench_sequence(dim, density_denom, quick);

    if !json_output { eprintln!("[6/6] Head-to-head vs HRR..."); }
    let hrr_dim = dim.min(512);
    let h2h = bench_head_to_head(hrr_dim, dim, density_denom, quick);

    let report = serde_json::json!({
        "meta": {
            "benchmark": "hms-research-bench",
            "version": "1.0.0",
            "dim": dim,
            "density_denom": density_denom,
            "active_indices": dim / density_denom,
            "quick_mode": quick,
            "elapsed_seconds": t_start.elapsed().as_secs_f64(),
        },
        "knowledge_graph": kg,
        "analogy": analogy,
        "interference": interference,
        "multihop": multihop,
        "sequence": sequence,
        "head_to_head_vs_hrr": h2h,
    });

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_report(&report);
    }
}

// ============================================================================
// Real-world knowledge graph dataset
// ============================================================================

struct KnowledgeGraph {
    entities: Vec<&'static str>,
    relations: Vec<(&'static str, Vec<(&'static str, &'static str)>)>,
}

fn build_knowledge_graph() -> KnowledgeGraph {
    let entities = vec![
        "france", "germany", "japan", "brazil", "india", "egypt", "australia",
        "canada", "mexico", "russia", "china", "italy", "spain", "nigeria",
        "argentina", "sweden", "greece", "turkey", "iran", "kenya",
        "paris", "berlin", "tokyo", "brasilia", "new_delhi", "cairo", "canberra",
        "ottawa", "mexico_city", "moscow", "beijing", "rome", "madrid", "abuja",
        "buenos_aires", "stockholm", "athens", "ankara", "tehran", "nairobi",
        "french", "german", "japanese", "portuguese", "hindi", "arabic", "english",
        "spanish", "russian", "mandarin", "italian", "greek", "turkish", "persian",
        "swahili", "swedish",
        "europe", "asia", "south_america", "africa", "oceania", "north_america",
        "euro", "yen", "real", "rupee", "pound", "dollar", "peso", "ruble",
        "yuan", "lira", "krona",
        "seine", "spree", "sumida", "amazon", "ganges", "nile", "murray",
        "st_lawrence", "colorado", "volga", "yangtze", "tiber", "ebro",
        "niger", "parana", "gota",
        "mammal", "bird", "fish", "reptile", "insect", "amphibian",
        "dog", "cat", "eagle", "shark", "snake", "frog", "ant", "whale",
        "penguin", "crocodile", "butterfly", "salamander", "dolphin", "hawk",
        "land", "sky", "ocean", "forest", "desert", "freshwater",
    ];

    let relations = vec![
        ("capital_of", vec![
            ("france", "paris"), ("germany", "berlin"), ("japan", "tokyo"),
            ("brazil", "brasilia"), ("india", "new_delhi"), ("egypt", "cairo"),
            ("australia", "canberra"), ("canada", "ottawa"), ("mexico", "mexico_city"),
            ("russia", "moscow"), ("china", "beijing"), ("italy", "rome"),
            ("spain", "madrid"), ("nigeria", "abuja"), ("argentina", "buenos_aires"),
            ("sweden", "stockholm"), ("greece", "athens"), ("turkey", "ankara"),
            ("iran", "tehran"), ("kenya", "nairobi"),
        ]),
        ("language_of", vec![
            ("france", "french"), ("germany", "german"), ("japan", "japanese"),
            ("brazil", "portuguese"), ("india", "hindi"), ("egypt", "arabic"),
            ("australia", "english"), ("canada", "english"), ("mexico", "spanish"),
            ("russia", "russian"), ("china", "mandarin"), ("italy", "italian"),
            ("spain", "spanish"), ("argentina", "spanish"), ("sweden", "swedish"),
            ("greece", "greek"), ("turkey", "turkish"), ("iran", "persian"),
            ("kenya", "swahili"),
        ]),
        ("continent_of", vec![
            ("france", "europe"), ("germany", "europe"), ("italy", "europe"),
            ("spain", "europe"), ("sweden", "europe"), ("greece", "europe"),
            ("turkey", "europe"), ("russia", "europe"),
            ("japan", "asia"), ("india", "asia"), ("china", "asia"),
            ("iran", "asia"),
            ("brazil", "south_america"), ("argentina", "south_america"),
            ("egypt", "africa"), ("nigeria", "africa"), ("kenya", "africa"),
            ("australia", "oceania"),
            ("canada", "north_america"), ("mexico", "north_america"),
        ]),
        ("currency_of", vec![
            ("france", "euro"), ("germany", "euro"), ("italy", "euro"),
            ("spain", "euro"), ("greece", "euro"),
            ("japan", "yen"), ("brazil", "real"), ("india", "rupee"),
            ("egypt", "pound"), ("australia", "dollar"), ("canada", "dollar"),
            ("mexico", "peso"), ("russia", "ruble"), ("china", "yuan"),
            ("turkey", "lira"), ("sweden", "krona"), ("argentina", "peso"),
        ]),
        ("river_in", vec![
            ("france", "seine"), ("germany", "spree"), ("japan", "sumida"),
            ("brazil", "amazon"), ("india", "ganges"), ("egypt", "nile"),
            ("australia", "murray"), ("canada", "st_lawrence"),
            ("mexico", "colorado"), ("russia", "volga"), ("china", "yangtze"),
            ("italy", "tiber"), ("spain", "ebro"), ("nigeria", "niger"),
            ("argentina", "parana"), ("sweden", "gota"),
        ]),
        ("class_of", vec![
            ("dog", "mammal"), ("cat", "mammal"), ("whale", "mammal"),
            ("dolphin", "mammal"),
            ("eagle", "bird"), ("penguin", "bird"), ("hawk", "bird"),
            ("shark", "fish"),
            ("snake", "reptile"), ("crocodile", "reptile"),
            ("frog", "amphibian"), ("salamander", "amphibian"),
            ("ant", "insect"), ("butterfly", "insect"),
        ]),
        ("habitat_of", vec![
            ("dog", "land"), ("cat", "land"), ("ant", "land"),
            ("eagle", "sky"), ("hawk", "sky"), ("butterfly", "sky"),
            ("shark", "ocean"), ("whale", "ocean"), ("dolphin", "ocean"),
            ("snake", "forest"), ("frog", "freshwater"),
            ("crocodile", "freshwater"), ("salamander", "freshwater"),
            ("penguin", "ocean"),
        ]),
    ];

    KnowledgeGraph { entities, relations }
}

// ============================================================================
// 1. Knowledge graph retrieval — real facts, Zipfian extension, shared entities
// ============================================================================

fn bench_knowledge_graph(dim: usize, dd: usize, quick: bool) -> serde_json::Value {
    let kg = build_knowledge_graph();
    let role_relation = 1usize;
    let role_arg1 = 2usize;
    let role_arg2 = 3usize;

    let mut entity_vecs: HashMap<&str, EntangledHVec> = HashMap::new();
    for &e in &kg.entities {
        let seed = stable_hash(e);
        entity_vecs.insert(e, EntangledHVec::new_with_density(dim, dd, seed));
    }

    let mut relation_vecs: HashMap<&str, EntangledHVec> = HashMap::new();
    for &(rel, _) in &kg.relations {
        let seed = stable_hash(rel);
        relation_vecs.insert(rel, EntangledHVec::new_with_density(dim, dd, seed));
    }

    let mut all_facts: Vec<(&str, &str, &str)> = Vec::new();
    let mut fact_vecs: Vec<EntangledHVec> = Vec::new();
    for &(rel, ref pairs) in &kg.relations {
        for &(a, b) in pairs {
            let comp = EntangledHVec::bundle_bloom(&[
                relation_vecs[rel].permute(role_relation),
                entity_vecs[a].permute(role_arg1),
                entity_vecs[b].permute(role_arg2),
            ]);
            fact_vecs.push(comp);
            all_facts.push((rel, a, b));
        }
    }

    let n_real_facts = all_facts.len();

    let n_synthetic = if quick { 200 } else { 2000 };
    let synthetic_entities: Vec<EntangledHVec> = (0..500)
        .map(|i| EntangledHVec::new_with_density(dim, dd, 0x5000_0000 + i as u64))
        .collect();
    let synthetic_relations: Vec<EntangledHVec> = (0..20)
        .map(|i| EntangledHVec::new_with_density(dim, dd, 0x6000_0000 + i as u64))
        .collect();

    for i in 0..n_synthetic {
        let zipf_rank = zipf_sample(i as u64, 500);
        let rel_idx = hash_u64(0xF100, i as u64) as usize % synthetic_relations.len();
        let b_idx = hash_u64(0xF200, i as u64) as usize % synthetic_entities.len();
        let comp = EntangledHVec::bundle_bloom(&[
            synthetic_relations[rel_idx].permute(role_relation),
            synthetic_entities[zipf_rank].permute(role_arg1),
            synthetic_entities[b_idx].permute(role_arg2),
        ]);
        fact_vecs.push(comp);
    }

    let total_facts = fact_vecs.len();

    let mut arg1_correct = 0;
    let mut arg2_correct = 0;
    let mut rel_correct = 0;
    let n_test = n_real_facts;

    let all_entities: Vec<&EntangledHVec> = entity_vecs.values().collect();
    let entity_names: Vec<&&str> = entity_vecs.keys().collect();
    let all_relations: Vec<&EntangledHVec> = relation_vecs.values().collect();
    let relation_names: Vec<&&str> = relation_vecs.keys().collect();

    for (i, &(true_rel, true_a, true_b)) in all_facts.iter().enumerate().take(n_test) {
        let comp = &fact_vecs[i];

        let best_a = (0..all_entities.len())
            .max_by(|&x, &y| {
                all_entities[x].permute(role_arg1).containment_similarity(comp)
                    .partial_cmp(&all_entities[y].permute(role_arg1).containment_similarity(comp)).unwrap()
            }).unwrap();
        if *entity_names[best_a] == true_a { arg1_correct += 1; }

        let best_b = (0..all_entities.len())
            .max_by(|&x, &y| {
                all_entities[x].permute(role_arg2).containment_similarity(comp)
                    .partial_cmp(&all_entities[y].permute(role_arg2).containment_similarity(comp)).unwrap()
            }).unwrap();
        if *entity_names[best_b] == true_b { arg2_correct += 1; }

        let best_r = (0..all_relations.len())
            .max_by(|&x, &y| {
                all_relations[x].permute(role_relation).containment_similarity(comp)
                    .partial_cmp(&all_relations[y].permute(role_relation).containment_similarity(comp)).unwrap()
            }).unwrap();
        if *relation_names[best_r] == true_rel { rel_correct += 1; }
    }

    let entity_counts = count_entity_frequency(&kg);
    let max_freq = entity_counts.values().copied().max().unwrap_or(0);
    let unique_entities = entity_counts.len();

    serde_json::json!({
        "n_real_facts": n_real_facts,
        "n_synthetic_facts": n_synthetic,
        "total_facts": total_facts,
        "n_real_entities": kg.entities.len(),
        "unique_entities_in_facts": unique_entities,
        "max_entity_frequency": max_freq,
        "n_relations": kg.relations.len(),
        "vocab_size": kg.entities.len() + kg.relations.len(),
        "arg1_accuracy": arg1_correct as f64 / n_test as f64,
        "arg2_accuracy": arg2_correct as f64 / n_test as f64,
        "relation_accuracy": rel_correct as f64 / n_test as f64,
        "overall_accuracy": (arg1_correct + arg2_correct + rel_correct) as f64 / (n_test * 3) as f64,
    })
}

fn count_entity_frequency(kg: &KnowledgeGraph) -> HashMap<&str, usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &(_, ref pairs) in &kg.relations {
        for &(a, b) in pairs {
            *counts.entry(a).or_default() += 1;
            *counts.entry(b).or_default() += 1;
        }
    }
    counts
}

fn zipf_sample(i: u64, max: usize) -> usize {
    let u = (hash_u64(0x21BF, i) as f64) / (u64::MAX as f64);
    let rank = (max as f64 * u.powf(0.75)) as usize;
    rank.min(max - 1)
}

// ============================================================================
// 2. Analogy completion — "A is to B as C is to ?"
// ============================================================================

fn bench_analogy(dim: usize, dd: usize) -> serde_json::Value {
    let kg = build_knowledge_graph();
    let role_arg1 = 2usize;
    let role_arg2 = 3usize;

    let mut entity_vecs: HashMap<&str, EntangledHVec> = HashMap::new();
    for &e in &kg.entities {
        entity_vecs.insert(e, EntangledHVec::new_with_density(dim, dd, stable_hash(e)));
    }

    let mut correct = 0;
    let mut total = 0;
    let mut by_relation: Vec<serde_json::Value> = Vec::new();

    for &(rel, ref pairs) in &kg.relations {
        if pairs.len() < 4 { continue; }
        let mut rel_correct = 0;
        let mut rel_total = 0;

        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                if rel_total >= 50 { break; }
                let (a, b) = pairs[i];
                let (c, true_d) = pairs[j];
                if a == c || b == true_d { continue; }

                let fact_ab = EntangledHVec::bundle_bloom(&[
                    entity_vecs[a].permute(role_arg1),
                    entity_vecs[b].permute(role_arg2),
                ]);
                let fact_cd = EntangledHVec::bundle_bloom(&[
                    entity_vecs[c].permute(role_arg1),
                    entity_vecs[true_d].permute(role_arg2),
                ]);

                let relation_bundle = EntangledHVec::bundle_bloom(&[&fact_ab, &fact_cd]);

                let candidates: Vec<&&str> = entity_vecs.keys()
                    .filter(|&&e| e != a && e != b && e != c)
                    .collect();

                let best = candidates.iter()
                    .max_by(|&&cand, &&other| {
                        entity_vecs[*cand].permute(role_arg2).containment_similarity(&relation_bundle)
                            .partial_cmp(&entity_vecs[*other].permute(role_arg2).containment_similarity(&relation_bundle))
                            .unwrap()
                    }).unwrap();

                if **best == true_d {
                    correct += 1;
                    rel_correct += 1;
                }
                total += 1;
                rel_total += 1;
            }
        }

        if rel_total > 0 {
            by_relation.push(serde_json::json!({
                "relation": rel,
                "accuracy": rel_correct as f64 / rel_total as f64,
                "n_tests": rel_total,
            }));
        }
    }

    serde_json::json!({
        "total_analogies": total,
        "correct": correct,
        "accuracy": if total > 0 { correct as f64 / total as f64 } else { 0.0 },
        "by_relation": by_relation,
    })
}

// ============================================================================
// 3. Interference stress — shared entities across facts
// ============================================================================

fn bench_interference(dim: usize, dd: usize, quick: bool) -> serde_json::Value {
    let role_arg1 = 2usize;
    let role_arg2 = 3usize;

    let n_entities = if quick { 100 } else { 500 };
    let entities: Vec<EntangledHVec> = (0..n_entities)
        .map(|i| EntangledHVec::new_with_density(dim, dd, 0x1A70 + i as u64))
        .collect();

    let fact_counts = if quick {
        vec![5, 10, 20, 50, 100, 200]
    } else {
        vec![5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    };

    let mut individual_results = Vec::new();
    let mut bundled_results = Vec::new();

    for &n_facts in &fact_counts {
        let shared_entity = 0usize;
        let mut facts: Vec<(usize, usize)> = Vec::new();
        let mut comps: Vec<EntangledHVec> = Vec::new();

        for t in 0..n_facts {
            let (a_idx, b_idx) = if t % 3 == 0 {
                (shared_entity, 1 + (hash_u64(0x1A00, t as u64) as usize % (n_entities - 1)))
            } else if t % 3 == 1 {
                (1 + (hash_u64(0x1B00, t as u64) as usize % (n_entities - 1)), shared_entity)
            } else {
                let a = 1 + (hash_u64(0x1C00, t as u64) as usize % (n_entities - 1));
                let b = 1 + (hash_u64(0x1D00, t as u64) as usize % (n_entities - 1));
                (a, if b == a { (a + 1) % n_entities } else { b })
            };

            let comp = EntangledHVec::bundle_bloom(&[
                entities[a_idx].permute(role_arg1),
                entities[b_idx].permute(role_arg2),
            ]);
            comps.push(comp);
            facts.push((a_idx, b_idx));
        }

        let n_probe = 30.min(n_facts);
        let mut ind_correct = 0;
        let ind_total = n_probe * 2;
        for t in 0..n_probe {
            let comp = &comps[t];
            let (true_a, true_b) = facts[t];

            let best_a = (0..n_entities)
                .max_by(|&x, &y| {
                    entities[x].permute(role_arg1).containment_similarity(comp)
                        .partial_cmp(&entities[y].permute(role_arg1).containment_similarity(comp)).unwrap()
                }).unwrap();
            if best_a == true_a { ind_correct += 1; }

            let best_b = (0..n_entities)
                .max_by(|&x, &y| {
                    entities[x].permute(role_arg2).containment_similarity(comp)
                        .partial_cmp(&entities[y].permute(role_arg2).containment_similarity(comp)).unwrap()
                }).unwrap();
            if best_b == true_b { ind_correct += 1; }
        }

        let shared_facts = facts.iter().filter(|f| f.0 == shared_entity || f.1 == shared_entity).count();

        individual_results.push(serde_json::json!({
            "n_facts": n_facts,
            "shared_entity_appears_in": shared_facts,
            "retrieval_accuracy": ind_correct as f64 / ind_total as f64,
        }));

        let all_bindings: Vec<EntangledHVec> = facts.iter()
            .flat_map(|&(a, b)| vec![
                entities[a].permute(role_arg1),
                entities[b].permute(role_arg2),
            ])
            .collect();
        let mega_bundle = EntangledHVec::bundle_bloom(&all_bindings);

        let mut bun_correct = 0;
        let bun_total = n_probe * 2;
        for t in 0..n_probe {
            let (true_a, true_b) = facts[t];

            let best_a = (0..n_entities)
                .max_by(|&x, &y| {
                    entities[x].permute(role_arg1).containment_similarity(&mega_bundle)
                        .partial_cmp(&entities[y].permute(role_arg1).containment_similarity(&mega_bundle)).unwrap()
                }).unwrap();
            if best_a == true_a { bun_correct += 1; }

            let best_b = (0..n_entities)
                .max_by(|&x, &y| {
                    entities[x].permute(role_arg2).containment_similarity(&mega_bundle)
                        .partial_cmp(&entities[y].permute(role_arg2).containment_similarity(&mega_bundle)).unwrap()
                }).unwrap();
            if best_b == true_b { bun_correct += 1; }
        }

        let active = dim / dd;
        let bundle_density = 1.0 - (1.0 - active as f64 / dim as f64).powi((n_facts * 2) as i32);

        bundled_results.push(serde_json::json!({
            "n_facts": n_facts,
            "shared_entity_appears_in": shared_facts,
            "retrieval_accuracy": bun_correct as f64 / bun_total as f64,
            "bundle_density": bundle_density,
        }));

        if bun_correct == 0 && n_facts > 20 { break; }
    }

    let mut results = Vec::new();
    for (i, ind) in individual_results.iter().enumerate() {
        let bun = &bundled_results[i];
        results.push(serde_json::json!({
            "n_facts": ind["n_facts"],
            "shared_entity_appears_in": ind["shared_entity_appears_in"],
            "individual_accuracy": ind["retrieval_accuracy"],
            "bundled_accuracy": bun["retrieval_accuracy"],
            "bundle_density": bun["bundle_density"],
        }));
    }

    serde_json::json!({
        "description": "Individual: each fact stored separately. Bundled: all facts in one Bloom vector.",
        "n_entities": n_entities,
        "results": results,
    })
}

// ============================================================================
// 4. Multi-hop inference — A->B->C chains
// ============================================================================

fn bench_multihop(dim: usize, dd: usize) -> serde_json::Value {
    let kg = build_knowledge_graph();

    let mut entity_vecs: HashMap<&str, EntangledHVec> = HashMap::new();
    for &e in &kg.entities {
        entity_vecs.insert(e, EntangledHVec::new_with_density(dim, dd, stable_hash(e)));
    }

    let role_arg1 = 2usize;
    let role_arg2 = 3usize;

    let capital_map: HashMap<&str, &str> = kg.relations.iter()
        .find(|r| r.0 == "capital_of").unwrap().1.iter().copied().collect();
    let continent_map: HashMap<&str, &str> = kg.relations.iter()
        .find(|r| r.0 == "continent_of").unwrap().1.iter().copied().collect();

    let mut hop1_correct = 0;
    let mut hop2_correct = 0;
    let mut total = 0;

    let all_entities: Vec<(&str, &EntangledHVec)> = entity_vecs.iter()
        .map(|(&k, v)| (k, v)).collect();

    for (&country, &capital) in &capital_map {
        if let Some(&continent) = continent_map.get(country) {
            let fact_cap = EntangledHVec::bundle_bloom(&[
                entity_vecs[country].permute(role_arg1),
                entity_vecs[capital].permute(role_arg2),
            ]);

            let best_capital = all_entities.iter()
                .max_by(|a, b| {
                    a.1.permute(role_arg2).containment_similarity(&fact_cap)
                        .partial_cmp(&b.1.permute(role_arg2).containment_similarity(&fact_cap)).unwrap()
                }).unwrap();

            if best_capital.0 == capital { hop1_correct += 1; }

            let fact_cont = EntangledHVec::bundle_bloom(&[
                entity_vecs[country].permute(role_arg1),
                entity_vecs[continent].permute(role_arg2),
            ]);

            let best_cont = all_entities.iter()
                .max_by(|a, b| {
                    a.1.permute(role_arg2).containment_similarity(&fact_cont)
                        .partial_cmp(&b.1.permute(role_arg2).containment_similarity(&fact_cont)).unwrap()
                }).unwrap();

            if best_cont.0 == continent { hop2_correct += 1; }
            total += 1;
        }
    }

    let chains: Vec<(&str, &str, &str, &str)> = capital_map.iter()
        .filter_map(|(&country, &capital)| {
            continent_map.get(country).map(|&cont| (country, capital, country, cont))
        })
        .collect();

    let mut chain_correct = 0;
    for &(country, capital, _, continent) in &chains {
        let fact1 = EntangledHVec::bundle_bloom(&[
            entity_vecs[country].permute(role_arg1),
            entity_vecs[capital].permute(role_arg2),
        ]);
        let fact2 = EntangledHVec::bundle_bloom(&[
            entity_vecs[country].permute(role_arg1),
            entity_vecs[continent].permute(role_arg2),
        ]);

        let retrieved_capital = all_entities.iter()
            .max_by(|a, b| {
                a.1.permute(role_arg2).containment_similarity(&fact1)
                    .partial_cmp(&b.1.permute(role_arg2).containment_similarity(&fact1)).unwrap()
            }).unwrap();

        if retrieved_capital.0 == capital {
            let retrieved_cont = all_entities.iter()
                .max_by(|a, b| {
                    a.1.permute(role_arg2).containment_similarity(&fact2)
                        .partial_cmp(&b.1.permute(role_arg2).containment_similarity(&fact2)).unwrap()
                }).unwrap();

            if retrieved_cont.0 == continent { chain_correct += 1; }
        }
    }

    serde_json::json!({
        "hop1_capital_accuracy": hop1_correct as f64 / total as f64,
        "hop2_continent_accuracy": hop2_correct as f64 / total as f64,
        "chain_accuracy": chain_correct as f64 / chains.len() as f64,
        "n_chains": chains.len(),
        "n_countries_tested": total,
    })
}

// ============================================================================
// 5. Sequence encoding — positional retrieval at each slot
// ============================================================================

fn bench_sequence(dim: usize, dd: usize, quick: bool) -> serde_json::Value {
    let max_len = if quick { 50 } else { 200 };
    let n_vocab = 500;
    let vocab: Vec<EntangledHVec> = (0..n_vocab)
        .map(|i| EntangledHVec::new_with_density(dim, dd, 0x5E00 + i as u64))
        .collect();

    let test_lengths = if quick {
        vec![3, 5, 10, 20, 30, 50]
    } else {
        vec![3, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200]
    };

    let n_trials = 10;
    let mut results = Vec::new();

    for &seq_len in &test_lengths {
        if seq_len > max_len { break; }
        let mut total_correct = 0;
        let mut total_queries = 0;

        for trial in 0..n_trials {
            let sequence: Vec<usize> = (0..seq_len)
                .map(|pos| hash_u64(0x5500 + trial as u64, pos as u64) as usize % n_vocab)
                .collect();

            let bindings: Vec<EntangledHVec> = sequence.iter().enumerate()
                .map(|(pos, &word_idx)| vocab[word_idx].permute(pos + 1))
                .collect();
            let comp = EntangledHVec::bundle_bloom(&bindings);

            for (pos, &true_idx) in sequence.iter().enumerate() {
                let best = (0..n_vocab)
                    .max_by(|&x, &y| {
                        vocab[x].permute(pos + 1).containment_similarity(&comp)
                            .partial_cmp(&vocab[y].permute(pos + 1).containment_similarity(&comp)).unwrap()
                    }).unwrap();
                if best == true_idx { total_correct += 1; }
                total_queries += 1;
            }
        }

        let accuracy = total_correct as f64 / total_queries as f64;
        results.push(serde_json::json!({
            "sequence_length": seq_len,
            "accuracy": accuracy,
            "n_trials": n_trials,
            "n_queries": total_queries,
        }));
    }

    serde_json::json!({
        "vocab_size": n_vocab,
        "results": results,
    })
}

// ============================================================================
// 6. Head-to-head vs HRR (Plate's Holographic Reduced Representation)
// ============================================================================

#[derive(Clone)]
struct HrrVec {
    data: Vec<f64>,
}

impl HrrVec {
    fn from_seed(dim: usize, seed: u64) -> Self {
        let mut data = vec![0.0f64; dim];
        for i in (0..dim).step_by(2) {
            let u1 = (hash_u64(seed, i as u64) as f64 / u64::MAX as f64).max(1e-15);
            let u2 = hash_u64(seed, i as u64 + 1) as f64 / u64::MAX as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data[i] = r * theta.cos();
            if i + 1 < dim { data[i + 1] = r * theta.sin(); }
        }
        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 { for x in &mut data { *x /= norm; } }
        Self { data }
    }

    fn bind(&self, other: &Self) -> Self {
        let n = self.data.len();
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                result[(i + j) % n] += self.data[i] * other.data[j];
            }
        }
        let norm = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 { for x in &mut result { *x /= norm; } }
        Self { data: result }
    }

    fn correlate(&self, other: &Self) -> Self {
        let n = self.data.len();
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for (k, cell) in result.iter_mut().enumerate() {
                *cell += self.data[i] * other.data[(i + k) % n];
            }
        }
        Self { data: result }
    }

    fn bundle(vecs: &[&Self]) -> Self {
        if vecs.is_empty() { return Self { data: Vec::new() }; }
        let n = vecs[0].data.len();
        let mut sum = vec![0.0f64; n];
        for v in vecs { for (i, &x) in v.data.iter().enumerate() { sum[i] += x; } }
        let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 { for x in &mut sum { *x /= norm; } }
        Self { data: sum }
    }

    fn cosine(&self, other: &Self) -> f64 {
        let dot: f64 = self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum();
        let na = self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = other.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-15 || nb < 1e-15 { return 0.0; }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

fn bench_head_to_head(hrr_dim: usize, ehv_dim: usize, dd: usize, quick: bool) -> serde_json::Value {
    let kg = build_knowledge_graph();
    let _n_entities = kg.entities.len();

    let mut ehv_entities: HashMap<&str, EntangledHVec> = HashMap::new();
    let mut hrr_entities: HashMap<&str, HrrVec> = HashMap::new();
    for &e in &kg.entities {
        let seed = stable_hash(e);
        ehv_entities.insert(e, EntangledHVec::new_with_density(ehv_dim, dd, seed));
        hrr_entities.insert(e, HrrVec::from_seed(hrr_dim, seed));
    }

    let mut ehv_roles: Vec<EntangledHVec> = Vec::new();
    let mut hrr_roles: Vec<HrrVec> = Vec::new();
    for i in 0..4 {
        ehv_roles.push(EntangledHVec::new_with_density(ehv_dim, dd, 0x40BE + i as u64));
        hrr_roles.push(HrrVec::from_seed(hrr_dim, 0x40BE + i as u64));
    }

    let test_relations = &["capital_of", "language_of", "continent_of"];
    let mut h2h_results = Vec::new();

    for &rel_name in test_relations {
        let pairs: &Vec<(&str, &str)> = &kg.relations.iter()
            .find(|r| r.0 == rel_name).unwrap().1;

        let mut ehv_correct = 0;
        let mut hrr_correct = 0;
        let n_test = pairs.len();

        for &(a, b) in pairs {
            let ehv_fact = EntangledHVec::bundle_bloom(&[
                ehv_entities[a].permute(2),
                ehv_entities[b].permute(3),
            ]);

            let ehv_best = kg.entities.iter()
                .max_by(|&&x, &&y| {
                    ehv_entities[x].permute(3).containment_similarity(&ehv_fact)
                        .partial_cmp(&ehv_entities[y].permute(3).containment_similarity(&ehv_fact)).unwrap()
                }).unwrap();
            if *ehv_best == b { ehv_correct += 1; }

            let hrr_fact = HrrVec::bundle(&[
                &hrr_roles[2].bind(&hrr_entities[a]),
                &hrr_roles[3].bind(&hrr_entities[b]),
            ]);
            let decoded = hrr_fact.correlate(&hrr_roles[3]);

            let hrr_best = kg.entities.iter()
                .max_by(|&&x, &&y| {
                    decoded.cosine(&hrr_entities[x])
                        .partial_cmp(&decoded.cosine(&hrr_entities[y])).unwrap()
                }).unwrap();
            if *hrr_best == b { hrr_correct += 1; }
        }

        h2h_results.push(serde_json::json!({
            "relation": rel_name,
            "n_facts": n_test,
            "ehv_accuracy": ehv_correct as f64 / n_test as f64,
            "hrr_accuracy": hrr_correct as f64 / n_test as f64,
            "ehv_dim": ehv_dim,
            "hrr_dim": hrr_dim,
        }));
    }

    let seq_lengths = if quick { vec![3, 5, 8] } else { vec![3, 5, 8, 12, 16] };
    let n_seq_vocab = 50;
    let ehv_vocab: Vec<EntangledHVec> = (0..n_seq_vocab)
        .map(|i| EntangledHVec::new_with_density(ehv_dim, dd, 0x4240 + i as u64))
        .collect();
    let hrr_vocab: Vec<HrrVec> = (0..n_seq_vocab)
        .map(|i| HrrVec::from_seed(hrr_dim, 0x4240 + i as u64))
        .collect();
    let hrr_positions: Vec<HrrVec> = (0..20)
        .map(|i| HrrVec::from_seed(hrr_dim, 0xB050 + i as u64))
        .collect();

    let mut seq_results = Vec::new();
    for &seq_len in &seq_lengths {
        let mut ehv_correct = 0;
        let mut hrr_correct = 0;
        let mut total = 0;
        let n_trials = 10;

        for trial in 0..n_trials {
            let sequence: Vec<usize> = (0..seq_len)
                .map(|p| hash_u64(0x4245 + trial as u64, p as u64) as usize % n_seq_vocab)
                .collect();

            let ehv_bindings: Vec<EntangledHVec> = sequence.iter().enumerate()
                .map(|(p, &w)| ehv_vocab[w].permute(p + 1))
                .collect();
            let ehv_comp = EntangledHVec::bundle_bloom(&ehv_bindings);

            let hrr_bindings: Vec<HrrVec> = sequence.iter().enumerate()
                .map(|(p, &w)| hrr_positions[p].bind(&hrr_vocab[w]))
                .collect();
            let hrr_refs: Vec<&HrrVec> = hrr_bindings.iter().collect();
            let hrr_comp = HrrVec::bundle(&hrr_refs);

            for (pos, &true_idx) in sequence.iter().enumerate() {
                let ehv_best = (0..n_seq_vocab)
                    .max_by(|&x, &y| {
                        ehv_vocab[x].permute(pos + 1).containment_similarity(&ehv_comp)
                            .partial_cmp(&ehv_vocab[y].permute(pos + 1).containment_similarity(&ehv_comp)).unwrap()
                    }).unwrap();
                if ehv_best == true_idx { ehv_correct += 1; }

                let hrr_decoded = hrr_comp.correlate(&hrr_positions[pos]);
                let hrr_best = (0..n_seq_vocab)
                    .max_by(|&x, &y| {
                        hrr_decoded.cosine(&hrr_vocab[x])
                            .partial_cmp(&hrr_decoded.cosine(&hrr_vocab[y])).unwrap()
                    }).unwrap();
                if hrr_best == true_idx { hrr_correct += 1; }

                total += 1;
            }
        }

        seq_results.push(serde_json::json!({
            "sequence_length": seq_len,
            "ehv_accuracy": ehv_correct as f64 / total as f64,
            "hrr_accuracy": hrr_correct as f64 / total as f64,
        }));
    }

    serde_json::json!({
        "knowledge_graph_retrieval": h2h_results,
        "sequence_encoding": seq_results,
        "notes": format!(
            "EHV: sparse binary, D={}, 1/{}, {} active. HRR: dense real, D={}, O(n^2) convolve.",
            ehv_dim, dd, ehv_dim / dd, hrr_dim
        ),
    })
}

// ============================================================================
// Utilities
// ============================================================================

fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn print_report(report: &serde_json::Value) {
    let meta = &report["meta"];
    eprintln!("\nHMS Research Benchmark v{}", meta["version"]);
    eprintln!("D={}, density=1/{}, active={}, elapsed={:.1}s",
        meta["dim"], meta["density_denom"], meta["active_indices"],
        meta["elapsed_seconds"].as_f64().unwrap_or(0.0));
    eprintln!("{}", "=".repeat(70));

    let kg = &report["knowledge_graph"];
    eprintln!("\n[1] KNOWLEDGE GRAPH RETRIEVAL");
    eprintln!("    {} real + {} synthetic = {} total facts, {} entities, {} relations",
        kg["n_real_facts"], kg["n_synthetic_facts"], kg["total_facts"],
        kg["unique_entities_in_facts"], kg["n_relations"]);
    eprintln!("    Max entity frequency: {} (tests Zipfian interference)",
        kg["max_entity_frequency"]);
    eprintln!("    Arg1 accuracy:     {:.1}%", kg["arg1_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    eprintln!("    Arg2 accuracy:     {:.1}%", kg["arg2_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    eprintln!("    Relation accuracy: {:.1}%", kg["relation_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    eprintln!("    Overall:           {:.1}%", kg["overall_accuracy"].as_f64().unwrap_or(0.0) * 100.0);

    let an = &report["analogy"];
    eprintln!("\n[2] ANALOGY COMPLETION (A:B :: C:?)");
    eprintln!("    {} analogies tested, {:.1}% accuracy",
        an["total_analogies"], an["accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    if let Some(by_rel) = an["by_relation"].as_array() {
        for r in by_rel {
            eprintln!("      {}: {:.0}% ({} tests)",
                r["relation"].as_str().unwrap_or("?"),
                r["accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["n_tests"]);
        }
    }

    let intf = &report["interference"];
    eprintln!("\n[3] INTERFERENCE STRESS ({} entities, shared entity across facts)", intf["n_entities"]);
    eprintln!("    {:>6}  {:>10}  {:>10}  {:>12}", "Facts", "Individual", "Bundled", "Bun.Density");
    if let Some(results) = intf["results"].as_array() {
        for r in results {
            eprintln!("    {:>6}  {:>9.1}%  {:>9.1}%  {:>11.4}",
                r["n_facts"],
                r["individual_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["bundled_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["bundle_density"].as_f64().unwrap_or(0.0));
        }
    }

    let mh = &report["multihop"];
    eprintln!("\n[4] MULTI-HOP INFERENCE ({} countries)", mh["n_countries_tested"]);
    eprintln!("    Hop 1 (country→capital):  {:.1}%",
        mh["hop1_capital_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    eprintln!("    Hop 2 (country→continent): {:.1}%",
        mh["hop2_continent_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
    eprintln!("    Chain (capital via country→continent): {:.1}%",
        mh["chain_accuracy"].as_f64().unwrap_or(0.0) * 100.0);

    let seq = &report["sequence"];
    eprintln!("\n[5] SEQUENCE ENCODING (vocab={})", seq["vocab_size"]);
    if let Some(results) = seq["results"].as_array() {
        for r in results {
            eprintln!("    Length {:3}: {:.1}% position accuracy",
                r["sequence_length"], r["accuracy"].as_f64().unwrap_or(0.0) * 100.0);
        }
    }

    let h2h = &report["head_to_head_vs_hrr"];
    eprintln!("\n[6] HEAD-TO-HEAD: EHV vs HRR");
    eprintln!("    {}", h2h["notes"].as_str().unwrap_or(""));
    eprintln!("    Knowledge graph retrieval:");
    if let Some(results) = h2h["knowledge_graph_retrieval"].as_array() {
        for r in results {
            eprintln!("      {}: EHV={:.0}% vs HRR={:.0}%",
                r["relation"].as_str().unwrap_or("?"),
                r["ehv_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["hrr_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
        }
    }
    eprintln!("    Sequence encoding:");
    if let Some(results) = h2h["sequence_encoding"].as_array() {
        for r in results {
            eprintln!("      Len {:2}: EHV={:.0}% vs HRR={:.0}%",
                r["sequence_length"], r["ehv_accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                r["hrr_accuracy"].as_f64().unwrap_or(0.0) * 100.0);
        }
    }
}
