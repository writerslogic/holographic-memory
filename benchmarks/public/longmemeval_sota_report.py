# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Reproduce the frozen-dev evidence study without network access or model calls."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiktoken

import longmemeval_retrieval as R
import longmemeval_retrieval_limits as L
import longmemeval_sota as S
import longmemeval_sota_closure as F
import longmemeval_sota_context as C
import longmemeval_sota_expand as E
import longmemeval_sota_hybrid as H
import longmemeval_sota_receipts as N
import longmemeval_sota_replay as Q


COMPETITORS = [
    {"system": "Agentmemory V4", "advertised": "96.2%, 481/500", "metric": "micro end-to-end QA accuracy",
     "reader": "Claude Opus 4.6, temperature 0", "judge": "GPT-4o, temperature 0, seed 42; exact snapshot unspecified",
     "source_protocol": "README explicitly describes a 500-case oracle dataset; runner defaults to longmemeval_oracle.json. USE_DIRECT_CONTEXT=False only disables direct injection; it does not restore distractor sessions removed from the input.",
     "routing": "Runner receives question_type and uses it for answer hints, output limits and retrieval policies.",
     "budget": "Per-category budgets; unmatched prompts and generation limits.",
     "comparability": "Not established as a full-distractor LongMemEval-S result. Actual published-run dataset override/hash requires audit; do not equate oracle-filtered retrieval with S retrieval.",
     "sources": ["https://github.com/JordanMcCann/agentmemory", "https://raw.githubusercontent.com/JordanMcCann/agentmemory/main/run_longmemeval_full.py"]},
    {"system": "Agent Zero Memory", "advertised": "95.6%", "metric": "end-to-end QA accuracy on 500 questions",
     "reader": "gpt-5.5 best backbone; eight-backbone study", "judge": "Official judge claimed; exact snapshot not specified in inspected experimental setup",
     "source_protocol": "Three provenance-aware memories; parallel agentic hybrid searches; PostgreSQL/pgvector.",
     "budget": "Variable agentic search and prompt cost; see paper tables 5-6 rather than a fixed ten-round allowance.",
     "comparability": "Different backbone, search orchestration and budgets. Paper itself calls cross-system comparisons uncontrolled.",
     "sources": ["https://arxiv.org/html/2608.29606v1"]},
    {"system": "Mastra Observational Memory", "advertised": "94.87%", "metric": "unweighted mean of six category accuracies",
     "reader": "gpt-5-mini actor; gemini-2.5-flash observer and reflector", "judge": "gpt-4o; exact snapshot unspecified on research page",
     "source_protocol": "LongMemEval-S, chronological observation/reflection. Published category numerators sum to 468/500 (93.6% micro); 94.87% is the category macro score.",
     "budget": "Approximately 30k average context tokens; model-dependent ingest compression.",
     "comparability": "Macro category accuracy differs from 481/500 micro accuracy; model, prompts, ingest computation and context are unmatched.",
     "sources": ["https://mastra.ai/research/observational-memory"]},
    {"system": "Hindsight", "advertised": "94.6%", "metric": "AMB end-to-end QA accuracy; v0.4.19 single-query result",
     "reader": "Exact model/config for the 94.6% run not established by inspected summary", "judge": "Current AMB README describes Gemini generation and Gemini judging; do not substitute the older paper's GPT-OSS judge or assume a fixed run model.",
     "source_protocol": "AMB single-query mode, not the older 91.4% paper condition. Summary and live leaderboard are insufficient to pin the complete run configuration.",
     "budget": "Exact retained/retrieved tokens for this run not verified.",
     "comparability": "Requires pinned AMB configuration, prompts and model versions before matched comparison.",
     "sources": ["https://hindsight.vectorize.io/blog/2026/03/23/agent-memory-benchmark", "https://raw.githubusercontent.com/vectorize-io/agent-memory-benchmark/main/README.md", "https://agentmemorybenchmark.ai/"]},
    {"system": "M3 Memory", "advertised": "99.2% session hit@8 and @10; 92.0% QA in current v3 README",
     "metric": "Binary any-evidence-session hit per question, not all-evidence recall",
     "reader": "Opus 4.6 for reported QA", "judge": "GPT-4o for reported QA",
     "source_protocol": "500-question S; production search for SHR. QA harness separately enables session expansion and recall-surface routing from question text.",
     "budget": "Retrieved rows/turns with session-hit attribution, not ten complete evidence rounds; expanded QA context differs.",
     "comparability": "Retrieval-only hit is not answer accuracy or complete turn coverage. Raw-file cache served an older 89% category-routed report while current HTML/README described v3; revisions must be pinned.",
     "sources": ["https://github.com/skynetcmd/m3-memory", "https://github.com/skynetcmd/m3-memory/blob/main/benchmarks/longmemeval/LME-S_Benchmarking_Report.md"]},
    {"system": "ContextCore / MemOS", "advertised": "99.4% hit@20; 98.3% evidence coverage@20; 96.4% all-evidence@20",
     "metric": "Session-level deterministic retrieval, three distinct aggregations",
     "reader": None, "judge": None,
     "source_protocol": "500-question S; one entire haystack session per result; contextual nomic-embed-text 768d via Ollama, keyword/semantic weights 0.5/0.5.",
     "budget": "20 entire sessions; raw rounds, source turns and reader tokens are not specified by the headline.",
     "comparability": "No QA score. Equal k does not match session and round context costs. The blog's claim that adding candidates can only help recall is not generally valid for fixed-k ranking.",
     "sources": ["https://context-core.dev/blogs/beating-supermemory-longmemeval/"]},
    {"system": "MemEval (evaluation infrastructure)", "advertised": None, "metric": "Harness rather than a competing score",
     "comparability": "Potential independent evaluation route; no score attributed to this framework.",
     "sources": ["https://github.com/ProsusAI/MemEval"]},
]


def paired(before, after, metric, samples=10000):
    b = {row["qid"]: row["metrics"][metric] for row in before}
    a = {row["qid"]: row["metrics"][metric] for row in after}
    R.require(set(b) == set(a), "paired membership differs")
    values = np.array([a[qid] - b[qid] for qid in sorted(b) if b[qid] is not None and a[qid] is not None])
    rng = np.random.default_rng(20261022)
    estimates = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return {"n": len(values), "mean_delta": float(values.mean()),
            "paired_bootstrap_95_percent_interval": [float(v) for v in np.quantile(estimates, [.025, .975])],
            "resamples": samples, "seed": 20261022,
            "interpretation": "Dev-sample uncertainty only; not held-out significance or LLM run variance."}


def packing(entries, receipts, encoder):
    before, after, identities = [], [], []
    encode_seconds = 0.
    for qid, entry in sorted(entries.items()):
        evidence = receipts[qid]["replies"]["selected"]["evidence"]
        original = C.canonical(evidence["sources"])
        started = time.perf_counter()
        context = C.pack(original)
        encode_seconds += time.perf_counter() - started
        restored = C.unpack(context)
        R.require(restored == original, "packing changes an original source")
        groups, originals, gs, gt = R.rounds_for(entry)
        m = S.metrics(R.native_groups(evidence, originals, groups), gs, gt, encoder)
        for target, payload in ((before, original), (after, context)):
            metrics = {**m, "context_bytes": len(C.wire(payload)),
                       "context_tokens": len(encoder.encode(C.wire(payload).decode(), disallowed_special=()))}
            target.append({"qid": qid, "qtype": entry["question_type"], "metrics": metrics})
        identities.append({"qid": qid, "sources_sha256": context["sourcesSha256"],
                           "raw_rounds": m["returned_rounds"], "original_sources": len(original),
                           "derived_context_packets": 1, "exact_round_trip": True,
                           "unresolved_premise_references": sum(type(p) is str for turn in context["turns"] for p in turn[6])})
    return {"before": {"summary": R.summarize(before), "rows": before},
            "after": {"summary": R.summarize(after), "rows": after},
            "identity_checks": identities, "paired_tokens": paired(before, after, "context_tokens"),
            "python_encode_seconds_100_questions": encode_seconds,
            "production_latency": "Rust compilation/tests via CI; native encoding latency not measured.",
            "reader_effect": "No new QA calls. Equal reconstructed sources establish encoding fidelity, not equal or improved reader accuracy.",
            "cost_definition": "o200k_base tokens for the complete context payload including digest, all source IDs, shared session identities and premise references. Full source receipt before; compact source payload after. Anchors/drops stay in the separate evidence receipt. Official QA history formatting differs; this is a context-format comparison, not an official-reader token saving.",
            "count_completeness": "Unknown. One packet retains the same raw rounds and sources; packet count is never reported as retrieval k."}


def bottlenecks(entries, receipts, reference):
    rows, candidate_rows = [], []
    categories = Counter()
    for qid, entry in sorted(entries.items()):
        groups, originals, gs, gt = R.rounds_for(entry)
        sources, closures, _, membership = S.inputs(groups)
        pos = {source["sourceId"]: i for i, source in enumerate(sources)}
        broad = receipts[qid]["replies"]["candidate_ceiling"]["evidence"]
        selected = receipts[qid]["replies"]["selected"]["evidence"]
        base = receipts[qid]["replies"]["baseline"]["evidence"]
        pool = {pos[item["sourceId"]] for item in [*base["anchors"], *base["dropped"]]}
        got = {source["sourceId"] for source in selected["sources"]}
        pool_sources = {sources[j]["sourceId"] for i in pool for j in closures[i]}
        scores = np.zeros(len(sources))
        native_position = {}
        for anchor in broad["anchors"]:
            i = pos[anchor["sourceId"]]
            if i in pool:
                weight = .96 if sources[i]["role"] == "assistant" and not L.assistant_intent(entry["question"]) else 1.
                scores[i] = anchor["score"] * weight
                native_position[i] = (sources[i]["role"] == "assistant", round(1 / anchor["score"] - 61))
        R.require(all(scores[i] > 0 for i in pool), "baseline candidate missing from broad receipt")
        ranking = sorted(pool, key=lambda i: (-scores[i], native_position[i]))
        replay = S.select(ranking, sources, closures, membership)
        replay_sources = {source["sourceId"] for group in replay for source in group["sources"]}
        R.require(replay_sources == got, "native selected source replay differs")
        first_ten = {sources[j]["sourceId"] for i in ranking[:10] for j in closures[i]}
        if not gt:
            stage = "H_abstention_unscored_for_retrieval"
        elif not gt <= pool_sources:
            stage = "C_standard_candidate_pool_incomplete"
        elif gt <= got:
            stage = "annotated_complete_reader_unmeasured"
        elif gt <= first_ten:
            stage = "E_premise_round_or_byte_budget"
        else:
            stage = "D_ranking_within_available_pool"
        categories[stage] += 1
        rows.append({"qid": qid, "qtype": entry["question_type"], "stage": stage,
                     "native_annotated_complete": bool(gt <= got) if gt else None,
                     "partial_multi_source_evidence": bool(gt & got and not gt <= got) if gt else None,
                     "pool32_gold_coverage": len(gt & pool_sources) / len(gt) if gt else None,
                     "required_gold_closure_raw_rounds": len({membership[j] for i, source in enumerate(sources) if source["sourceId"] in gt for j in closures[i]}) if gt else None,
                     "exhaustive_count_verified": False})
        native_by_depth = {}
        for depth in (5, 10, 32, 128, 1000):
            candidate = {pos[anchor["sourceId"]] for anchor in broad["anchors"] if anchor["score"] >= 1 / (60 + depth) - 1e-15}
            members = {sources[j]["sourceId"] for i in candidate for j in closures[i]}
            direct = {sources[i]["sourceId"] for i in candidate}
            native_by_depth[str(depth)] = {"anchor_gold_coverage": len(direct & gt) / len(gt) if gt else None,
                                           "closure_gold_coverage": len(members & gt) / len(gt) if gt else None,
                                           "closure_all": float(gt <= members) if gt else None,
                                           "admitted_anchors": len(candidate)}
        lookup = {group["id"]: group for group in groups}
        ranking_ids = next(row["round_ids"] for row in reference["hybrid_base"]["rows"] if row["qid"] == qid)
        ref_depths = S.metrics([lookup[i] for i in ranking_ids], gs, gt, tiktoken.get_encoding("o200k_base"))
        candidate_rows.append({"qid": qid, "qtype": entry["question_type"], "native_per_role_depth": native_by_depth,
                               "reference_raw_round_depth": {metric: value for metric, value in ref_depths.items() if any(f"@{k}" == metric[-len(str(k))-1:] for k in S.DEPTHS)}})
    complete = {row["qid"] for row in reference["hybrid_base"]["rows"] if row["metrics"]["evidence_all@10"] == 1}
    reader_runs = []
    for name in ("longmemeval_official_dev_gpt4o_k10.json", *(f"longmemeval_official_dev_gpt4o_k10_r{i}.json" for i in range(1, 4))):
        path = R.ROOT / "benchmarks/results" / name
        data = json.loads(path.read_text())
        R.require({row["qid"] for row in data["rows"]} == set(entries) and not data["heldout_evaluated"], "reader artifact split differs")
        wrong = [row for row in data["rows"] if not row["correct"]]
        reader_runs.append({"artifact": name, "sha256": R.file_digest(path),
                            "correct": sum(bool(row["correct"]) for row in data["rows"]),
                            "wrong_despite_complete_annotated_context": [row["qid"] for row in wrong if row["qid"] in complete],
                            "wrong_with_annotation_gap_or_abstention": [row["qid"] for row in wrong if row["qid"] not in complete],
                            "max_reader_prompt_tokens": max(row["reader_prompt_tokens"] for row in data["rows"])})
    return {"native_stage_counts": dict(categories), "rows": rows, "candidate_depth_rows": candidate_rows,
            "candidate_depth_definition": "Native depth is per-role original rank among anchors admitted by the wide receipt (k=1000/role, 4096 sources, 2 MiB). Missing/dropped unscored anchors are not invented; deep curves are conservative admitted-pool measurements, not a claim of exhaustive history search. Reference depth counts uncompressed original user/assistant rounds, not sessions.",
            "reader_runs": reader_runs,
            "stage_limits": {"A": "No annotated source absent from the admitted wide pool; unrelated unannotated omissions are not measured.",
                             "B": "Representation adequacy cannot be separated causally from lexical mismatch using annotations alone.",
                             "F": "Partial multi-source failures are flagged separately; annotations do not establish event identity or exhaustive counts.",
                             "G": "Cached official-model reader/judge failures with all annotated turns are reader-or-judge failure candidates, not proof all semantically needed evidence reached the reader.",
                             "H": "Six dev abstention questions excluded from positive retrieval metrics. No learned abstention accuracy claimed."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=R.ROOT / "benchmarks/results/longmemeval_sota_dev_v1.json")
    parser.add_argument("--ci-run", type=int)
    parser.add_argument("--ci-revision")
    parser.add_argument("--ci-conclusion", default="not independently recorded")
    parser.add_argument("--python-gate-log", type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    canonical_path = R.ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json"
    canonical = json.loads(canonical_path.read_text())
    R.require(R.digest(entries) == canonical["dev_sha256"], "frozen dev digest differs")
    scripts = {path.name: R.file_digest(path) for path in sorted(Path(__file__).parent.glob("longmemeval_sota*.py"))}
    inputs = {name: R.file_digest(R.RUN / name) for name in ("scores.json", "targets.json", "rerank.json", "queries.json")}
    R.require(all(inputs[name] == canonical["artifacts"][name] for name in inputs), "cached model inputs differ")
    key = R.digest([scripts, inputs, R.file_digest(N.DEFAULT), canonical["dev_sha256"]])
    cache = S.CACHE / f"measurements-{key}.json"
    receipts = N.load(entries)
    if cache.exists() and not args.fresh:
        measurements = json.loads(cache.read_text())
    else:
        encoder = tiktoken.get_encoding("o200k_base")
        print("lexical architecture controls", flush=True)
        lexical = S.experiment(entries, receipts, encoder)
        print("native candidate/selection controls", flush=True)
        replay = Q.replay(entries, receipts, canonical)
        print("cached hybrid representation/selection controls", flush=True)
        hybrid = H.study(entries)
        measurements = {"lexical": lexical, "replay": replay, "hybrid": hybrid,
                        "packing": packing(entries, receipts, encoder),
                        "enumeration_expansion": E.study(entries, receipts),
                        "general_expansion": E.study(entries, receipts, aggregation_only=False),
                        "same_round_completion": F.study(entries, receipts),
                        "bottlenecks": bottlenecks(entries, receipts, hybrid)}
        S.CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(measurements))
    result = {"schema_version": 1, "as_of": "2026-10-22", "split": "frozen dev100", "dev_only": True,
              "n_questions": 100, "answerable_questions": 94, "heldout_records_decoded": 0, "llm_api_calls": 0,
              "paid_compute_calls": 0, "new_model_inference_calls": 0,
              "dev_sha256": canonical["dev_sha256"], "source_sha256": canonical["source_sha256"],
              "split_sha256": canonical["split_sha256"], "baseline_artifact_sha256": R.file_digest(canonical_path),
              "portable_native_receipts_sha256": R.file_digest(N.DEFAULT), "cached_reference_inputs": inputs,
              "analysis_scripts": scripts, "measurement_cache_key": key,
              "measurement_classes": {"native": "Previously validated native lexical receipts, replayed without a new build; every selected source set is checked against its receipt.",
                                      "lexical_prototypes": "Python full-history representation/selection experiments, not new native Rust retrieval measurements.",
                                      "reference": "Previously cached Qwen embeddings, fact/query transformations and reranker probabilities; no new inference.",
                                      "reader": "Prior official gpt-4o reader/judge dev artifacts only. No new QA score.",
                                      "oracle": "Gold dependency-cost analysis is evaluation-only and never enters ranking."},
              "retained": {"lossless_context": measurements["packing"],
                           "enumeration_expansion": {**measurements["enumeration_expansion"],
                             "hypothesis": "Use lossless metadata savings to append complete original-source closures for text-inferred enumeration queries without removing any baseline evidence.",
                             "baseline": "native selected: ten raw rounds, unchanged assistant weight 0.96",
                             "intervention": "Monotone append from admitted native candidates within role depth 128, capped at 64 original sources and 50 raw rounds; compact payload bytes <= the per-question baseline full source receipt bytes (or compact seed size if larger). Non-aggregation queries retain the seed only.",
                             "decision": "retain behind explicit API/CLI options; no default promotion before matched QA",
                             "limitations": "Variable raw-round budget. No hit@10 or complete-evidence@10 claim. Native candidate receipts are reused, not new model inference. Actual Rust acquisition with k=128/candidateLimit=1000 and latency remain unmeasured; candidate ranks at wider chunk depths may differ. Event identity, semantic sufficiency and exhaustive counts remain unverified.",
                             "paired_complete_evidence": paired(measurements["packing"]["before"]["rows"], measurements["enumeration_expansion"]["rows"], "evidence_all_selected"),
                             "paired_tokens": paired(measurements["packing"]["before"]["rows"], measurements["enumeration_expansion"]["rows"], "context_tokens")}},
              "experiment_ranking": [], "bottlenecks": measurements["bottlenecks"],
              "competitor_comparison": {"audit_date": "2026-10-22", "independent_reproduction_performed": False,
                                        "matrix": COMPETITORS, "comparable_competitor_win_established": False},
              "validation": {"new_local_cargo_invocations": 0, "new_remote_release_build_attempts": 0,
                             "old_pr_required_ci_run": 39821173020, "old_pr_revision": "2b2acad2fd8d35d71c1a1da8ffb8ae03ce3f7ba4",
                             "new_ci_run": args.ci_run, "new_ci_revision": args.ci_revision,
                             "new_ci_conclusion": args.ci_conclusion,
                             "python_gate_log_sha256": R.file_digest(args.python_gate_log) if args.python_gate_log else None,
                             "python_gate_log_tail": args.python_gate_log.read_text().splitlines()[-8:] if args.python_gate_log else None,
                             "rust_sources_sha256": {name: R.file_digest(R.ROOT / name) for name in (
                                 "src/core/memory_api/conversation.rs", "src/core/memory_api/conversation/context.rs",
                                 "src/bin/conversation-eval.rs", "tests/conversation_context.rs",
                                 "tests/fixtures/conversation_context_v1.json")}},
              "blocked": ["New matched reader/judge evaluation requires funded credits; no quota probes made.",
                          "Held-out 400 and final full-500 feedback are unauthorized.",
                          "New native model-stage retrieval and native context timing have not been measured; no fresh local cargo or remote release build was run."],
              "reproduction": {"command": "uv run --script benchmarks/public/longmemeval_sota_report.py --fresh",
                               "native_receipts": str(N.DEFAULT.relative_to(R.ROOT)),
                               "data": "Pinned public LongMemEval-S cleaned release; guarded binary-framed loader decodes exactly dev100.",
                               "reference_cache": "~/.cache/hms-bench/longmemeval_s/runs/s-dev-small-all2; all four file digests checked. Not newly inferred or distributed with this PR.",
                               "gate": "uv run --no-project --offline --python 3.13.9 --with numpy==2.2.6 --with tiktoken==0.14.0 --with openai==3.27.0 --with python-dotenv==1.2.4 --with tokenizers==0.23.3 python -m unittest discover -s benchmarks/public -p 'test_*longmemeval*.py'"},
              "highest_value_next_experiment": "Run a locally scored, event/entity-linked candidate expansion on the frozen dev count failures, preserving literal spans and unknown event identity; compare to existing matched native candidates under equal raw rounds and reader-format tokens before funding new QA."}
    result["retained"]["general_expansion"] = {
        **measurements["general_expansion"], "baseline": "native selected, ten raw rounds",
        "hypothesis": "Remove the aggregation-only gate to measure the evidence/context frontier across all query categories.",
        "intervention": "Same monotone source selection and byte budget as enumeration expansion; aggregationOnly=false.",
        "decision": "retain as an explicit option; existing retrieval defaults unchanged",
        "paired_complete_evidence": paired(measurements["packing"]["before"]["rows"], measurements["general_expansion"]["rows"], "evidence_all_selected"),
        "paired_tokens": paired(measurements["packing"]["before"]["rows"], measurements["general_expansion"]["rows"], "context_tokens"),
        "limitations": "Not ten-round recall. Additional source text can harm the reader. Token savings concern the defined source-context serialization including provenance; ordinary reader-history formatting has different costs. New native candidate acquisition, latency and QA remain unmeasured."
    }
    native_rows = measurements["packing"]["before"]["rows"]
    compact_rows = measurements["packing"]["after"]["rows"]
    completion_rows = measurements["same_round_completion"]["rows"]
    result["rejected"] = {"same_round_completion": {
        **measurements["same_round_completion"],
        "hypothesis": "Lossless metadata savings can recover missing evidence by filling sources inside the existing ten raw rounds without increasing source-payload bytes or tokens over the full native source receipt.",
        "baseline": "native selected, identical seed sources and raw-round membership",
        "intervention": "Greedily append original sources from seed rounds, prioritizing referenced premises then smaller payloads; cap at 64 sources and each question's baseline full source-payload bytes and o200k_base tokens. Selection never sees annotations.",
        "decision": "reject production promotion; retain this negative result",
        "paired_complete_evidence": paired(native_rows, completion_rows, "evidence_all_selected"),
        "paired_evidence_coverage": paired(native_rows, completion_rows, "evidence_coverage_selected"),
        "paired_tokens_vs_native": paired(native_rows, completion_rows, "context_tokens"),
        "paired_tokens_vs_compact": paired(compact_rows, completion_rows, "context_tokens"),
        "limitations": "A greedy Python receipt replay, not an exhaustive search or a native retrieval/QA measurement. A negative result does not prove that every equal-budget intervention must fail. No rank metrics are attributed to chronological source order."
    }}
    arms = {"native": measurements["packing"]["before"], "compact": measurements["packing"]["after"],
            "enumeration_expansion": measurements["enumeration_expansion"],
            "general_expansion": measurements["general_expansion"],
            "same_round_completion": measurements["same_round_completion"]}
    selected_metrics = ("evidence_all_selected", "evidence_coverage_selected", "session_coverage_selected")
    result["category_comparison"] = {}
    for category in arms["native"]["summary"]:
        rows = [row for row in native_rows if category == "overall" or row["qtype"] == category]
        result["category_comparison"][category] = {
            "n_questions": len(rows),
            "n_answerable": sum(row["metrics"]["evidence_all_selected"] is not None for row in rows),
            "arms": {name: {metric: arm["summary"][category][metric]
                            for metric in (*selected_metrics, "returned_rounds", "returned_turns", "context_bytes", "context_tokens")}
                     for name, arm in arms.items()}}
    result["coverage_regressions"] = {}
    baseline_by_id = {row["qid"]: row["metrics"] for row in native_rows}
    for name, arm in arms.items():
        regressions = {metric: [row["qid"] for row in arm["rows"]
                               if row["metrics"][metric] is not None
                               and row["metrics"][metric] < baseline_by_id[row["qid"]][metric]]
                       for metric in selected_metrics}
        R.require(not any(regressions.values()), f"{name} regresses selected evidence or session coverage")
        result["coverage_regressions"][name] = regressions
    completion = arms["same_round_completion"]["summary"]["overall"]
    native = arms["native"]["summary"]["overall"]
    expanded = arms["general_expansion"]["summary"]["overall"]
    result["tradeoff_assessment"] = {
        "serialization_redundancy": "Lossless packing reduces source-payload tokens with identical original sources and category coverage; reader accuracy and latency remain unmeasured.",
        "general_expansion_minus_native": {metric: expanded[metric] - native[metric]
                                           for metric in (*selected_metrics, "returned_rounds", "context_tokens")},
        "same_round_completion_minus_native": {metric: completion[metric] - native[metric]
                                              for metric in (*selected_metrics, "returned_rounds", "context_tokens")},
        "additional_raw_round_cost_eliminated": False,
        "fixed_seed_round_ceiling": measurements["same_round_completion"]["fixed_round_ceiling"],
        "interpretation": "All-query expansion improves annotated evidence coverage at a larger raw-round budget while remaining cheaper than the full native source receipt in this serialization. Completing only existing rounds recovers no additional annotated evidence on frozen dev100; evaluation of all sources in those fixed rounds finds no missing annotation there even without context caps. This control has not eliminated the evidence/raw-round tradeoff; selecting a different ten rounds or other interventions may still do so. Neither result establishes a matched QA improvement."
    }
    result["bottlenecks"]["contexts_with_unresolved_premise_references"] = sum(row["unresolved_premise_references"] > 0 for row in measurements["packing"]["identity_checks"])
    result["bottlenecks"]["ranked_interventions"] = [
        {"stage": stage, "questions": count,
         "intervention": {"C_standard_candidate_pool_incomplete": "Union contextual/semantic candidates before ranking; inspect direct-anchor versus premise coverage.",
                          "D_ranking_within_available_pool": "Optimize evidence sets and event identity rather than global role weights.",
                          "E_premise_round_or_byte_budget": "Bounded lossless packing/monotone expansion, followed by an audit of unresolved premise links."}.get(stage, "No retrieval intervention justified by this stage alone.")}
        for stage, count in sorted(result["bottlenecks"]["native_stage_counts"].items(), key=lambda item: -item[1])]
    result["development_selection"] = "Frozen dev used for architecture selection/rejection. No fitted model or new role-weight sweep. Paired bootstrap intervals characterize dev examples, not held-out generalization. No claim of significance for a small number of recovered questions."
    result["continuation"] = {
        "objective": "Improve evidence sufficiency toward matched full-500 end-to-end SOTA; no comparable win established.",
        "worktree": "/Volumes/A/.sol-wt-mem-sota", "branch": "feat/memory-sota-evidence-assembly",
        "draft_pr": "https://github.com/writerslogic/holographic-memory/pull/203", "base_pr": 202,
        "next": ["Inspect remaining ranking misses, unresolved premise links and event identities using these pinned dev-only receipts.",
                 "Measure the new native acquisition/assembly path only when an additional build/run is authorized; CI proves compilation/tests, not native dev quality.",
                 "Use a funded, matched reader/judge protocol before promoting either expansion option or making accuracy/context claims."],
        "constraints": "No paid calls or quota probes; no held-out decoding; no merges; preserve other worktrees. Old local cargo and both remote release-build attempts remain consumed."
    }
    for family in ("lexical", "replay", "hybrid"):
        for name, arm in measurements[family].items():
            if name in ("gold_dependency_cost", "native_selected", "native_candidate_ceiling", "reference_baseline", "reference_selected", "hybrid_base"):
                continue
            baseline = measurements["hybrid"]["hybrid_base"]["rows"] if family == "hybrid" else native_rows
            result["experiment_ranking"].append({"name": name, "family": family,
                "hypothesis": {"lexical": "Context, dependencies or literal event postings improve evidence selection beyond isolated lexical hits.",
                               "replay": "Shared round support or marginal dependency cost improves selection from the same native candidate evidence.",
                               "hybrid": "Pair representations or deeper original-question reranking recover evidence lost by the narrow reference head."}[family],
                "baseline": "cached hybrid_base" if family == "hybrid" else "native selected",
                "controlled_intervention": name, "decision": "retain as baseline control" if name in ("raw_full_turn", "native_wide_replay") else "reject production promotion",
                "reason": "No tested retrieval alternative dominates complete-evidence recall and per-category coverage; raising session coverage alone is insufficient.",
                "paired_complete_evidence": paired(baseline, arm["rows"], "evidence_all@10"),
                "summary": arm["summary"], "rows": arm["rows"]})
    result["experiment_ranking"].sort(key=lambda arm: (-arm["summary"]["overall"]["evidence_all@10"], -arm["summary"]["overall"]["session_recall@10"]))
    keep = {"session_recall@10", "session_all@10", "evidence_any@10", "evidence_all@10", "round_turn_recall@10",
            "evidence_all_selected", "evidence_coverage_selected", "session_coverage_selected", "returned_rounds", "returned_turns",
            "context_bytes", "context_tokens", "source_text_bytes", "source_text_tokens", "evidence_mrr", "source_ndcg@20"}
    for experiment in result["experiment_ranking"]:
        experiment["rows"] = [{"qid": row["qid"], "qtype": row["qtype"],
                               "metrics": {metric: value for metric, value in row["metrics"].items() if metric in keep}}
                              for row in experiment["rows"]]
    for name in ("before", "after"):
        for row in result["retained"]["lossless_context"][name]["rows"]:
            row["metrics"] = {metric: value for metric, value in row["metrics"].items() if metric in keep}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    p = result["retained"]["lossless_context"]
    print(json.dumps({"output": str(args.output), "native_stage_counts": result["bottlenecks"]["native_stage_counts"],
                      "before": p["before"]["summary"]["overall"], "after": p["after"]["summary"]["overall"]}), flush=True)


if __name__ == "__main__":
    main()
