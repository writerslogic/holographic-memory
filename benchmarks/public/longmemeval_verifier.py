# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Freeze an independent, source-only Sol reader for every dev question."""

import argparse
import ast
import copy
import fcntl
import json
import math
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import tiktoken

import longmemeval_reader_api as reader
import longmemeval_role_reader as role
import longmemeval_user_first as user_first
import longmemeval_verifier_sources as source_core
from check_longmemeval_answers import artifact_path, recompute_scores, require, same, wilson
from longmemeval_answer_evidence import reader_payload
from longmemeval_dev_conversations import load_dev_conversations
from longmemeval_role_retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks/results"
DEFAULT_WORK = RESULTS / "longmemeval_verifier_v1"
SYSTEM = "sol61_reader"
ARM = "citation_union"
READER_MODEL = "gpt-6.1-sol"
JUDGE_MODEL = "gpt-5.4-2026-03-05"
CONTEXT_CAP = 32768
FRAMING_RESERVE = 512
LOCAL_PROXY_CAP = 24000
NEW_SPEND_CAP_USD = 14.0
TRANSPORT_RETRY_POLICY = {
    "max_retries": 2, "http_statuses": [429], "http_status_range": [500, 599],
    "error_types": ["APIConnectionError", "APITimeoutError"], "requires_status": "error",
    "requires_response_none": True, "requires_usage_none": True, "backoff_seconds": [2, 8],
    "retry_call_id_suffix": ":transport-retry:{retry_number}", "completed_response_is_final": True,
    "semantic_retries": False,
}
SOURCE_POLICY = "uniform four-prior citation union, source premises, then question-only RRF counterevidence; complete original turns"
VERIFIER_RULES = (
    "\nIndependently answer using only the question and supplied original sources. Check whether the question's "
    "premise is established before accepting it. Search the supplied evidence for explicit corrections, later "
    "state changes, scope differences, and counterevidence to a plausible answer. Resolve these by source roles, "
    "entity identity, relation, effective time, and observation time, and give a qualified answer when warranted. "
    "The presence of a source does not endorse every assertion within it. Never infer a personal action from "
    "an assistant suggestion. Count only distinct identifiable events meeting the requested window; keep "
    "unnamed or unplaced mentions separate when they may describe the same event. Distinguish a habitual "
    "inventory or routine from a precisely stated usage frequency, and qualify ordinary supported routine use. "
    "Emit the required JSON directly, with original source IDs and exact quotations. If the requested fact is "
    "unsupported, answer 'I don't know.' with abstain true and claims []; do not mix an implied zero with "
    "an unknown quantity.\n"
)
PRIOR_SPECS = (
    ("v3_operand", user_first.BASELINE, "matched_gpt54", "operand"),
    ("full_history", RESULTS / "longmemeval_full_history_v1/answers.json", "matched_gpt54", "full_history"),
    ("role_retrieval", RESULTS / "longmemeval_role_retrieval_v1/answers.json", "matched_gpt54", "full_history"),
    ("complete_history", RESULTS / "longmemeval_complete_history_v1/answers.json", "matched_gpt54", "full_history"),
)
RATES = {
    "reader": {"input": 2.0, "cached_input": 0.1, "cache_write_30m": 2.5, "cache_write_12h": 5.0, "output": 10.0},
    "judge": {"input": 2.5, "cached_input": 0.25, "output": 15.0},
}


def literal_rule(path, name):
    tree = ast.parse(path.read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, "generic prompt rule is missing or ambiguous")
    value = ast.literal_eval(values[0])
    require(isinstance(value, str) and value, "invalid generic prompt rule")
    return value


def make_protocol(base):
    settings = base["systems"]["matched_gpt54"]
    for key, value in (("reader_model", JUDGE_MODEL), ("judge_model", JUDGE_MODEL), ("reasoning_effort", "medium"),
                       ("reader_max_output_tokens", 4096), ("judge_max_output_tokens", 2048)):
        same(settings[key], value, "original pinned settings changed")
    rule_files = [(Path(__file__).with_name("longmemeval_full_history.py"), "REASONING_RULES"),
                  (Path(__file__).with_name("longmemeval_complete_history.py"), "COMPLETE_REASONING_RULES")]
    rules = [(path, symbol, literal_rule(path, symbol)) for path, symbol in rule_files]
    return {
        "schema_version": 1, "dev_only": True, "heldout_run": False, "acceptance_correct": 100,
        "reader_prompt": base["reader_prompt"] + "".join(value for _, _, value in rules) + VERIFIER_RULES,
        "additional_reasoning_rules": VERIFIER_RULES,
        "prompt_rule_sources": [{"path": str(path.relative_to(ROOT)), "sha256": reader.sha(path),
                                 "symbol": symbol, "rule_sha256": reader.digest(value)} for path, symbol, value in rules],
        "systems": {SYSTEM: {"reader_model": READER_MODEL, "judge_model": JUDGE_MODEL,
                             "reader_reasoning_effort": "high", "judge_reasoning_effort": "medium",
                             "reader_max_output_tokens": 4096, "judge_max_output_tokens": 2048}},
        "judge_source": copy.deepcopy(base["judge_source"]), "token_cap": CONTEXT_CAP,
        "exact_preflight_cap": CONTEXT_CAP - FRAMING_RESERVE, "framing_reserve_tokens": FRAMING_RESERVE,
        "local_proxy_cap": LOCAL_PROXY_CAP, "source_policy": SOURCE_POLICY,
        "source_gate": {"metadata_ids_verified": 500, "dev_records_decoded": 100, "heldout_records_decoded": 0},
        "source_selection": {"prior_names": [spec[0] for spec in PRIOR_SPECS], "prior_answers_visible": False,
                             "question_only_counterevidence": "reserved user prior then equal-weight RRF60 fused ranking",
                             "whole_turns_only": True, "cap_overflow": "selection_failure; no source clipping"},
        "input_count": {"endpoint": "/v1/responses/input_tokens", "request_fields": ["model", "input"],
                        "settings_framing_reserve": FRAMING_RESERVE},
        "service_tier": "default", "reader_prompt_cache_options": {"mode": "explicit"},
        "rates_usd_per_million": copy.deepcopy(RATES), "new_spend_cap_usd": NEW_SPEND_CAP_USD,
        "automatic_transport_retries": 2, "transport_retry_policy": copy.deepcopy(TRANSPORT_RETRY_POLICY),
        "reader_judge_concurrency": 1,
        "limitations": ["Repeated-use dev only; no held-out claim.",
                        "The reader API publishes an alias, not an immutable dated snapshot; record raw response.model.",
                        "Local o200k counts are selection proxies; the model endpoint supplies the frozen preflight count.",
                        "The 512-token reserve covers generation settings absent from the count request; actual usage is checked.",
                        "Previous answer texts, judgments, scores, references, and question types do not enter source selection.",
                        "All 100 questions receive the new reader; no result-dependent fallback or resampling.",
                        "Explicit cache mode has no breakpoints; unexpected cache writes are charged and fail the call."],
    }


def request(protocol, prompt, stage):
    require(stage in ("reader", "judge"), "invalid stage")
    settings = protocol["systems"][SYSTEM]
    value = {"model": settings[f"{stage}_model"], "input": prompt,
             "reasoning": {"effort": settings[f"{stage}_reasoning_effort"]},
             "max_output_tokens": settings[f"{stage}_max_output_tokens"],
             "store": False, "truncation": "disabled", "service_tier": "default"}
    if stage == "reader":
        value["prompt_cache_options"] = {"mode": "explicit"}
    return value


def transport_request(flat):
    value = copy.deepcopy(flat)
    if "prompt_cache_options" in value:
        value["extra_body"] = {"prompt_cache_options": value.pop("prompt_cache_options")}
    return value


def serialize(question, groups, protocol):
    return protocol["reader_prompt"] + reader.canonical(reader_payload(question, {"source_groups": groups}))


def stripped_prior(report, system, arm_name, dev):
    require(report["dev_only"] is True and report["heldout_run"] is False, "prior is not dev-only")
    rows = user_first.membership(report["rows"], dev, "prior readers")
    stripped = []
    for qid in sorted(dev):
        row = rows[qid]
        arm = row["systems"][system]["arms"][arm_name]
        status = arm["reader"]["status"]
        require(status in ("ok", "error", "selection_failure"), "finish all 100 primary readers before preparing")
        parsed = arm["reader"].get("parsed")
        citations = list(dict.fromkeys(citation["source_id"] for claim in (parsed or {}).get("claims", [])
                                       for citation in claim["citations"]))
        require(all(isinstance(value, str) and value for value in citations), "invalid prior citation identity")
        stripped.append({"qid": qid, "question": row["question"], "question_date": row["question_date"],
                         "reader_status": status, "source_groups": copy.deepcopy(arm["source_groups"]),
                         "cited_source_ids": citations})
    return stripped


def prior_reader_proof(report, system, arm_name, dev):
    rows = user_first.membership(report["rows"], dev, "prior reader proof")
    return [{"qid": qid, "reader": {key: copy.deepcopy(value) for key, value in
             rows[qid]["systems"][system]["arms"][arm_name]["reader"].items()
             if key in ("status", "raw_text", "parsed", "call")}} for qid in sorted(dev)]


def freeze_priors(work, dev):
    manifest_path = work / "priors/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        same([entry["name"] for entry in manifest], [spec[0] for spec in PRIOR_SPECS], "prior snapshot names changed")
        snapshots = []
        for entry in manifest:
            path = work / "priors" / (entry["name"] + ".json")
            require(reader.sha(path) == entry["sha256"], "frozen prior snapshot changed")
            require(reader.sha(artifact_path(entry["reader_proof_path"])) == entry["reader_proof_sha256"],
                    "frozen prior reader proof changed")
            snapshot = json.loads(path.read_text())
            user_first.membership(snapshot["rows"], dev, "frozen prior readers")
            snapshots.append(snapshot)
        return snapshots, manifest
    snapshots, proofs = [], []
    for name, path, system, arm in PRIOR_SPECS:
        data = path.read_bytes()
        report = json.loads(data)
        rows = stripped_prior(report, system, arm, dev)
        snapshots.append({"schema_version": 1, "dev_only": True, "heldout_run": False, "name": name,
                          "source_path": str(path.relative_to(ROOT)), "source_sha256_at_snapshot": reader.digest(data.decode()),
                          "system": system, "arm": arm, "rows": rows})
        proofs.append({"schema_version": 1, "dev_only": True, "heldout_run": False, "name": name,
                       "source_path": str(path.relative_to(ROOT)), "source_sha256_at_snapshot": reader.digest(data.decode()),
                       "system": system, "arm": arm, "rows": prior_reader_proof(report, system, arm, dev)})
    manifest = []
    for snapshot, proof in zip(snapshots, proofs, strict=True):
        path = work / "priors" / (snapshot["name"] + ".json")
        proof_path = work / "priors" / (snapshot["name"] + "_reader_proof.json")
        reader.write(path, snapshot)
        reader.write(proof_path, proof)
        manifest.append({"name": snapshot["name"], "path": str(path.relative_to(ROOT)), "sha256": reader.sha(path),
                         "reader_proof_path": str(proof_path.relative_to(ROOT)), "reader_proof_sha256": reader.sha(proof_path)})
    reader.write(manifest_path, manifest)
    return snapshots, manifest


def exact_input_count(api, prompt, work):
    req = {"model": READER_MODEL, "input": prompt}
    key = reader.digest(reader.canonical({"namespace": "sol61-input-count-v1", "request": req}))
    path = work / "input_counts" / (key + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        same(result["request"], req, "cached input-count request changed")
        return result
    result = {"cache_key": key, "request": req, "request_sha256": reader.digest(reader.canonical(req))}
    try:
        response = api.responses.input_tokens.count(**req).model_dump(mode="json")
        require(type(response.get("input_tokens")) is int and response["input_tokens"] >= 0, "invalid endpoint token count")
        result.update(status="ok", response=response)
    except Exception as error:
        result.update(status="error", response=None, error_type=type(error).__name__,
                      http_status=getattr(error, "status_code", None))
    reader.write(path, result)
    return result


def prepare_arm(question, original_groups, seeds, counter_ids, protocol, local_count, exact_count):
    selected = source_core.select_verifier_sources(original_groups, seeds, counter_ids,
                                                  lambda groups: serialize(question, groups, protocol),
                                                  local_count, cap=protocol["local_proxy_cap"])
    groups = selected["source_groups"]
    prompt = serialize(question, groups, protocol)
    check = exact_count(prompt)
    require(check["request"] == {"model": READER_MODEL, "input": prompt}, "input-count request differs from reader")
    tokens = check["response"]["input_tokens"] if check["status"] == "ok" else 0
    require(type(tokens) is int and tokens >= 0, "invalid exact token count")
    failure = selected["failure_reason"] if selected["selection_failure"] else None
    if check["status"] != "ok":
        failure = "exact_input_count_failed"
    elif tokens > protocol["exact_preflight_cap"]:
        failure = "exact_input_count_exceeds_cap"
    arm = {"source_groups": groups, "retained_sources": reader.retained(groups), "reader_input": prompt,
           "reader_input_sha256": reader.digest(prompt), "input_tokens": tokens,
           "local_proxy_input_tokens": selected["input_tokens"], "framing_checks": [check],
           "source_selection": {key: value for key, value in selected.items() if key != "source_groups"},
           "reader_cap": {"cap": protocol["token_cap"], "exact_preflight_cap": protocol["exact_preflight_cap"],
                          "input_tokens": tokens, "selection_failure": failure is not None,
                          "failure_reason": failure, "source_policy": "citation_union"},
           "reader": {"status": "selection_failure" if failure else "pending", "raw_text": "", "parsed": None},
           "judge": {"status": "pending", "raw_text": "", "score": False}}
    arm["scores"] = recompute_scores(arm, allow_incomplete=True)
    return arm


def call_id(work, stage, qid):
    return reader.digest(f"sol61-verifier:{work.name}:{stage}:{qid}")


def retry_call_id(base_id, retry_number):
    require(type(retry_number) is int and 0 <= retry_number <= TRANSPORT_RETRY_POLICY["max_retries"],
            "invalid transport retry number")
    return base_id if retry_number == 0 else base_id + f":transport-retry:{retry_number}"


def cache_key(req, identity, protocol_sha256):
    return reader.digest(reader.canonical({"namespace": "sol61-verifier-call-v1", "request": req,
                                           "call_id": identity, "protocol_sha256": protocol_sha256}))


def nonnegative_int(value, label):
    require(type(value) is int and value >= 0, "invalid usage " + label)
    return value


def usage_cost(usage, stage):
    require(stage in RATES and isinstance(usage, dict), "invalid cost stage or usage")
    incoming = nonnegative_int(usage.get("input_tokens"), "input_tokens")
    outgoing = nonnegative_int(usage.get("output_tokens"), "output_tokens")
    if "total_tokens" in usage:
        require(nonnegative_int(usage["total_tokens"], "total_tokens") == incoming + outgoing,
                "usage total differs from input plus output")
    output_details = usage.get("output_tokens_details", {})
    require(isinstance(output_details, dict), "invalid output token details")
    require(nonnegative_int(output_details.get("reasoning_tokens", 0), "reasoning_tokens") <= outgoing,
            "reasoning tokens exceed output")
    details = usage.get("input_tokens_details")
    require(isinstance(details, dict), "invalid input token details")
    cached = nonnegative_int(details.get("cached_tokens"), "cached_tokens")
    writes = nonnegative_int(details.get("cache_write_tokens"), "cache_write_tokens")
    require(cached + writes <= incoming, "cache reads and writes exceed total input")
    rates = RATES[stage]
    conservative = False
    if stage == "reader":
        classes = details.get("cache_write_tokens_details")
        if classes is None:
            short, long = 0, writes
            conservative = writes > 0
        else:
            require(isinstance(classes, dict), "invalid cache write classes")
            require(set(classes) <= {"ttl_30m_tokens", "ttl_12h_tokens"}, "unknown cache write classes")
            short = nonnegative_int(classes.get("ttl_30m_tokens", 0), "ttl_30m_tokens")
            long = nonnegative_int(classes.get("ttl_12h_tokens", 0), "ttl_12h_tokens")
            require(short + long == writes, "cache write classes do not sum to writes")
        cost = ((incoming - cached - writes) * rates["input"] + cached * rates["cached_input"]
                + short * rates["cache_write_30m"] + long * rates["cache_write_12h"] + outgoing * rates["output"]) / 1e6
    else:
        require(writes == 0, "unexpected judge cache writes")
        cost = ((incoming - cached) * rates["input"] + cached * rates["cached_input"] + outgoing * rates["output"]) / 1e6
    return {"cost_usd": cost, "unexpected_cache_writes": writes > 0, "cost_is_conservative": conservative}


def reservation_usd(input_tokens, output_tokens, stage):
    nonnegative_int(input_tokens, "reservation input")
    require(type(output_tokens) is int and output_tokens > 0 and stage in RATES, "invalid reservation output or stage")
    rates = RATES[stage]
    input_rate = rates.get("cache_write_12h", rates["input"])
    return ((input_tokens + FRAMING_RESERVE) * input_rate + output_tokens * rates["output"]) / 1e6


def ledger_totals(entries, caps):
    categories = {category: sum(entry["cost_usd"] for entry in entries if entry["category"] == category)
                  for category in caps if category != "total"}
    return {"schema_version": 1, "caps_usd": copy.deepcopy(caps), "entries": entries,
            "total_usd": sum(entry["cost_usd"] for entry in entries), "by_category_usd": categories}


def merge_ledgers(ledgers, caps, own_call_ids, terminal_proofs):
    require(caps["reader_judge_dev"] == 120 and caps["total"] == 250 and caps["heldout400"] == 0,
            "global spend caps changed")
    grouped = {}
    for ledger in ledgers:
        same(ledger["caps_usd"], caps, "ledger spend caps differ")
        for entry in ledger["entries"]:
            cost = entry["cost_usd"]
            require(isinstance(entry["cache_key"], str) and entry["cache_key"] and isinstance(cost, (int, float))
                    and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0, "invalid ledger entry")
            require(entry["category"] in caps and entry["category"] != "total", "invalid spend category")
            grouped.setdefault(entry["cache_key"], []).append(entry)
    merged = []
    for key, entries in grouped.items():
        own = any(entry.get("call_id") in own_call_ids for entry in entries)
        if own:
            anchor = entries[0]
            for entry in entries:
                for field in ("call_id", "category", "reserved_usd"):
                    same(entry[field], anchor[field], "conflicting own reservation identity")
            proof = terminal_proofs.get(key)
            terminals = [entry for entry in entries if entry["status"] != "reserved"]
            if proof is not None:
                same(proof["call_id"], anchor["call_id"], "terminal cache call identity differs")
                same(proof["reserved_usd"], anchor["reserved_usd"], "terminal cache reservation differs")
                for entry in terminals:
                    for field in ("status", "cost_usd", "usage"):
                        same(entry.get(field), proof.get(field), "conflicting terminal ledger records")
                chosen = {**copy.deepcopy(anchor), **{field: copy.deepcopy(proof.get(field))
                          for field in ("status", "cost_usd", "usage")}}
            else:
                require(not terminals, "own terminal ledger lacks durable cache proof")
                chosen = max(entries, key=lambda entry: entry["cost_usd"])
        else:
            chosen = max(entries, key=lambda entry: entry["cost_usd"])
        merged.append(copy.deepcopy(chosen))
    return ledger_totals(merged, caps)


def refresh_ledger(work, own_call_ids, protocol_sha256):
    paths = sorted({RESULTS / "spend_2026-10.json", *RESULTS.glob("longmemeval_dev_answers*_spend.json"),
                    *RESULTS.glob("longmemeval_*/spend.json"), work / "spend.json"})
    paths = [path for path in paths if path.exists()]
    primary = json.loads((RESULTS / "spend_2026-10.json").read_text())
    proofs = {}
    for path in (work / "cache").glob("*.json"):
        output = json.loads(path.read_text())
        require(output["cache_key"] == path.stem and output["call_id"] in own_call_ids
                and output["protocol_sha256"] == protocol_sha256, "invalid own terminal cache provenance")
        require(cache_key(output["request"], output["call_id"], protocol_sha256) == path.stem
                and reader.digest(reader.canonical(output["request"])) == output["request_sha256"], "invalid cached request")
        require(output["status"] in ("ok", "error", "incomplete", "validation_failure"), "invalid terminal cache status")
        if output["cost_is_conservative_reservation"]:
            same(output["cost_usd"], output["reserved_usd"], "conservative cache charge changed")
        else:
            stage = "reader" if output["request"]["model"] == READER_MODEL else "judge"
            same(output["cost_usd"], usage_cost(output["usage"], stage)["cost_usd"], "cached usage charge changed")
        proofs[path.stem] = output
    result = merge_ledgers([json.loads(path.read_text()) for path in paths], primary["caps_usd"], own_call_ids, proofs)
    result["observed_ledger_paths"] = [str(path.relative_to(ROOT)) for path in paths]
    reader.write(work / "spend.json", result)
    return result


def variant_spend(ledger, identities):
    return sum(entry["cost_usd"] for entry in ledger["entries"] if entry.get("call_id") in identities)


def budget_rejection(ledger, identities, reserve):
    require(isinstance(reserve, (int, float)) and not isinstance(reserve, bool)
            and math.isfinite(reserve) and reserve >= 0, "invalid reservation")
    if variant_spend(ledger, identities) + reserve > NEW_SPEND_CAP_USD:
        return "new_variant_spend_cap"
    if ledger["total_usd"] + reserve > ledger["caps_usd"]["total"]:
        return "global_total_spend_cap"
    if ledger["by_category_usd"]["reader_judge_dev"] + reserve > ledger["caps_usd"]["reader_judge_dev"]:
        return "reader_judge_dev_spend_cap"
    return None


@contextmanager
def ledger_lock(work):
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".ledger.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def api_call(api, protocol, req, identity, stage, work, identities, input_tokens):
    protocol_sha256 = reader.sha(work / "protocol.json")
    key = cache_key(req, identity, protocol_sha256)
    path = work / "cache" / (key + ".json")
    reserve = reservation_usd(input_tokens, req["max_output_tokens"], stage)
    with ledger_lock(work):
        ledger = refresh_ledger(work, identities, protocol_sha256)
        if path.exists():
            return json.loads(path.read_text())
        require(not any(entry["cache_key"] == key for entry in ledger["entries"]),
                "unsettled call already reserved; reconcile before resuming, never resample")
        reason = budget_rejection(ledger, identities, reserve)
        if reason:
            return {"status": "budget_exhausted", "raw_text": "", "response": None, "usage": None, "cost_usd": 0,
                    "request": req, "request_sha256": reader.digest(reader.canonical(req)), "cache_key": key,
                    "call_id": identity, "protocol_sha256": protocol_sha256, "budget_scope": reason}
        ledger["entries"].append({"cache_key": key, "call_id": identity, "category": "reader_judge_dev",
                                  "status": "reserved", "reserved_usd": reserve, "cost_usd": reserve,
                                  "unix_time": time.time(), "usage": None})
        reader.write(work / "spend.json", ledger_totals(ledger["entries"], ledger["caps_usd"]))
    output = {"status": "error", "raw_text": "", "response": None, "usage": None, "cost_usd": reserve,
              "cost_is_conservative_reservation": True}
    try:
        response = api.responses.create(**transport_request(req))
        output.update(raw_text=response.output_text, response=response.model_dump(mode="json"),
                      usage=response.usage.model_dump(mode="json") if response.usage else None)
        priced = usage_cost(output["usage"], stage)
        output.update(priced)
        output["cost_is_conservative_reservation"] = False
        problems = []
        if response.model != req["model"]:
            problems.append("response_model_differs_from_requested_alias")
        if stage == "reader" and output["usage"]["input_tokens"] > protocol["token_cap"]:
            problems.append("actual_input_exceeds_cap")
        if priced["unexpected_cache_writes"]:
            problems.append("unexpected_explicit_mode_cache_writes")
        output["validation_failures"] = problems
        output["status"] = "validation_failure" if problems else ("ok" if response.status == "completed" else "incomplete")
    except Exception as error:
        output.update(error_type=type(error).__name__, http_status=getattr(error, "status_code", None))
    output.update(request=copy.deepcopy(req), request_sha256=reader.digest(reader.canonical(req)), cache_key=key,
                  call_id=identity, protocol_sha256=protocol_sha256, reserved_usd=reserve)
    with ledger_lock(work):
        reader.write(path, output)
        refresh_ledger(work, identities, protocol_sha256)
    return output


def retryable_transport_failure(output):
    if output.get("status") != "error" or output.get("response") is not None or output.get("usage") is not None:
        return False
    status = output.get("http_status")
    return ((type(status) is int and (status == 429 or 500 <= status <= 599))
            or output.get("error_type") in TRANSPORT_RETRY_POLICY["error_types"])


def call_with_transport_retries(api, protocol, req, base_id, stage, work, identities, input_tokens,
                                sleep_callback=None):
    same(protocol["automatic_transport_retries"], 2, "transport retry count changed")
    same(protocol["transport_retry_policy"], TRANSPORT_RETRY_POLICY, "transport retry policy changed")
    previous = []
    for attempt in range(TRANSPORT_RETRY_POLICY["max_retries"] + 1):
        if attempt:
            (sleep_callback or time.sleep)(TRANSPORT_RETRY_POLICY["backoff_seconds"][attempt - 1])
        output = api_call(api, protocol, req, retry_call_id(base_id, attempt), stage, work, identities, input_tokens)
        if not retryable_transport_failure(output) or attempt == TRANSPORT_RETRY_POLICY["max_retries"]:
            return output, previous
        previous.append(output)
    raise RuntimeError("transport retry loop did not terminate")


def input_manifest(rows):
    return [{"qid": row["qid"], "reader_input_sha256": row["systems"][SYSTEM]["arms"][ARM]["reader_input_sha256"],
             "input_tokens": row["systems"][SYSTEM]["arms"][ARM]["input_tokens"],
             "input_count_cache_key": row["systems"][SYSTEM]["arms"][ARM]["framing_checks"][0]["cache_key"]} for row in rows]


def validate_inputs(rows, dev, protocol):
    user_first.membership(rows, dev, "prepared verifier answers")
    same(input_manifest(rows), protocol["prepared_inputs"], "frozen verifier inputs changed")
    for row in rows:
        arm = row["systems"][SYSTEM]["arms"][ARM]
        prompt = serialize(row, arm["source_groups"], protocol)
        require(prompt == arm["reader_input"] and reader.digest(prompt) == arm["reader_input_sha256"], "prepared verifier prompt changed")
        same(reader.retained(arm["source_groups"]), arm["retained_sources"], "retained verifier sources changed")
        check = arm["framing_checks"][0]
        same(check["request"], {"model": READER_MODEL, "input": prompt}, "exact preflight input changed")
        require(reader.digest(reader.canonical(check["request"])) == check["request_sha256"], "preflight request hash changed")
        if not arm["reader_cap"]["selection_failure"]:
            require(check["status"] == "ok" and type(check["response"]["input_tokens"]) is int
                    and check["response"]["input_tokens"] == arm["input_tokens"] <= protocol["exact_preflight_cap"],
                    "reader exceeds exact preflight cap")


def summarize(report):
    rows = report["rows"]
    require(len(rows) == 100 and len({row["qid"] for row in rows}) == 100, "accuracy requires all 100 dev questions")
    arms = [row["systems"][SYSTEM]["arms"][ARM] for row in rows]
    correct = sum(arm["scores"]["correct"] for arm in arms)
    terminal = all(arm[stage]["status"] != "pending" for arm in arms for stage in ("reader", "judge"))
    tokens = [arm["input_tokens"] for arm in arms]
    return {"n": 100, "correct": correct, "answer_accuracy": correct / 100, "answer_accuracy_ci95": wilson(correct, 100),
            "acceptance_correct": 100, "acceptance_met": correct == 100 and report["complete"] and terminal,
            "reader_status": dict(Counter(arm["reader"]["status"] for arm in arms)),
            "judge_status": dict(Counter(arm["judge"]["status"] for arm in arms)),
            "input_tokens": {"median": statistics.median(tokens), "max": max(tokens)},
            "abstentions": sum(arm["scores"]["abstain"] for arm in arms),
            "exact_quote_claims": sum(arm["scores"]["citation_valid_claims"] for arm in arms),
            "total_claims": sum(arm["scores"]["citation_total_claims"] for arm in arms),
            "user_occurrences": sum(source["role"] == "user" for arm in arms for source in arm["retained_sources"]),
            "assistant_occurrences": sum(source["role"] == "assistant" for arm in arms for source in arm["retained_sources"])}


def all_call_ids(work, rows):
    return {retry_call_id(call_id(work, stage, row["qid"]), attempt) for row in rows
            for stage in ("reader", "judge") for attempt in range(TRANSPORT_RETRY_POLICY["max_retries"] + 1)}


def save(report, work):
    report["summary"] = summarize(report)
    ledger = json.loads((work / "spend.json").read_text())
    report["spend_ledger_sha256"] = reader.sha(work / "spend.json")
    report["incremental_spend_usd"] = variant_spend(ledger, all_call_ids(work, report["rows"]))
    report["global_recorded_spend_usd"] = ledger["total_usd"]
    reader.write(work / "answers.json", report)


def prepare(work):
    require(not (work / "answers.json").exists(), "prepared experiment exists; resume instead of overwriting")
    dev = set(json.loads(user_first.SPLIT.read_text())["dev"])
    require(len(dev) == 100, "verifier requires exactly 100 dev questions")
    snapshots, prior_manifest = freeze_priors(work, dev)
    prior_rows = [user_first.membership(snapshot["rows"], dev, "frozen prior") for snapshot in snapshots]
    conversations = load_dev_conversations(role.SOURCE, user_first.SPLIT)
    history = user_first.membership(json.loads(user_first.HISTORICAL.read_text())["rows"], dev, "historical ranking")
    protocol = make_protocol(user_first.load_protocol())
    tokenizer = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(tokenizer.encode(text))
    api = reader.client()
    rows = []
    for qid in sorted(dev):
        raw = conversations[qid]
        question = {"question": raw["question"], "question_date": raw["question_date"]}
        seeds = []
        for previous in prior_rows:
            row = previous[qid]
            same({key: row[key] for key in question}, question, "prior question differs from original dev question")
            seeds.append({key: row[key] for key in ("source_groups", "cited_source_ids")})
        pool, by_alias, orphans = role.groups_for_question(raw, set(history[qid]["ids"]))
        user_prior = [by_alias[alias] for alias in dict.fromkeys(history[qid]["baseline_ranking"])]
        ranked = retrieve(question["question"], pool, user_prior, limit=64)
        order = list(dict.fromkeys(ranked["reserved_user_prior"] + ranked["fused_ranking"]))
        counter_ids = [pool[index]["group_id"] for index in order]
        arm = prepare_arm(question, pool, seeds, counter_ids, protocol, count,
                          lambda prompt: exact_input_count(api, prompt, work))
        rows.append({"qid": qid, "qtype": raw["question_type"], **question,
                     "systems": {SYSTEM: {"arms": {ARM: arm}}}, "conversation_groups": len(pool),
                     "orphan_assistant_groups": orphans, "counterevidence_group_ids": counter_ids})
        if len(rows) % 10 == 0:
            print(json.dumps({"prepared_dev_questions": len(rows)}), flush=True)
    protocol["prepared_inputs"] = input_manifest(rows)
    protocol["prior_snapshots"] = prior_manifest
    protocol["global_spend_caps_usd"] = json.loads((ROOT / reader.PREREGISTRATION).read_text())["spend_caps_usd"]
    validate_inputs(rows, dev, protocol)
    reader.write(work / "protocol.json", protocol)
    identities = all_call_ids(work, rows)
    with ledger_lock(work):
        ledger = refresh_ledger(work, identities, reader.sha(work / "protocol.json"))
    files = [Path(__file__), Path(source_core.__file__), Path(role.__file__), Path(reader.__file__), Path(user_first.__file__),
             Path(__file__).with_name("longmemeval_role_retrieval.py"), Path(__file__).with_name("longmemeval_dev_conversations.py"),
             Path(__file__).with_name("longmemeval_answer_evidence.py"), Path(__file__).with_name("longmemeval_dev_sources.py"),
             Path(__file__).with_name("longmemeval_reader_budget.py"), Path(__file__).with_name("check_longmemeval_answers.py"),
             Path(__file__).parent / "proxies/beir_proxies.py", role.SOURCE, user_first.HISTORICAL, user_first.SPLIT,
             ROOT / reader.PROTOCOL, ROOT / reader.PREREGISTRATION, user_first.BASELINE, user_first.EXPORT,
             artifact_path(protocol["judge_source"]["path"]), work / "priors/manifest.json",
             *[artifact_path(entry["path"]) for entry in prior_manifest],
             *[artifact_path(entry["reader_proof_path"]) for entry in prior_manifest],
             *[artifact_path(entry["path"]) for entry in protocol["prompt_rule_sources"]],
             *[(work / "input_counts" / (entry["input_count_cache_key"] + ".json")) for entry in protocol["prepared_inputs"]]]
    files = list(dict.fromkeys(files))
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [SYSTEM],
              "variant": work.name, "complete": False, "rows": rows,
              "protocol_path": str((work / "protocol.json").relative_to(ROOT)), "protocol_sha256": reader.sha(work / "protocol.json"),
              "starting_spend_usd": ledger["total_usd"], "prior_snapshots": prior_manifest,
              "artifacts": [{"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                             "sha256": reader.sha(path)} for path in files]}
    save(report, work)
    print(json.dumps(report["summary"]), flush=True)


def run_stage(report, work, stage):
    require(stage in ("reader", "judge") and report["dev_only"] is True and report["heldout_run"] is False,
            "only reader/judge dev stages are supported")
    for artifact in report["artifacts"]:
        require(reader.sha(artifact_path(artifact["path"])) == artifact["sha256"], "frozen experiment artifact changed")
    require(reader.sha(work / "protocol.json") == report["protocol_sha256"], "frozen protocol changed")
    protocol = json.loads((work / "protocol.json").read_text())
    require(protocol["token_cap"] == CONTEXT_CAP and protocol["exact_preflight_cap"] == CONTEXT_CAP - FRAMING_RESERVE
            and protocol["new_spend_cap_usd"] == NEW_SPEND_CAP_USD and protocol["source_policy"] == SOURCE_POLICY,
            "verifier protocol settings changed")
    dev = set(json.loads(user_first.SPLIT.read_text())["dev"])
    validate_inputs(report["rows"], dev, protocol)
    require(stage != "judge" or all(row["systems"][SYSTEM]["arms"][ARM]["reader"]["status"] != "pending"
                                     for row in report["rows"]), "finish all readers before judging")
    references = user_first.reference_answers(json.loads(user_first.BASELINE.read_text()), dev) if stage == "judge" else None
    tasks = [row for row in report["rows"] if row["systems"][SYSTEM]["arms"][ARM][stage]["status"] == "pending"]
    api = reader.client() if tasks else None
    identities = all_call_ids(work, report["rows"])
    tokenizer = tiktoken.get_encoding("o200k_base")
    for done, row in enumerate(tasks, 1):
        arm = row["systems"][SYSTEM]["arms"][ARM]
        if stage == "judge" and arm["reader"]["status"] != "ok":
            arm["judge"] = {"status": "reader_failure", "raw_text": "", "score": False}
        else:
            prompt = arm["reader_input"] if stage == "reader" else reader.judge_prompt(
                protocol, row, arm["reader"]["parsed"]["answer"], references[row["qid"]])
            tokens = arm["input_tokens"] if stage == "reader" else len(tokenizer.encode(prompt))
            output, attempts = call_with_transport_retries(
                api, protocol, request(protocol, prompt, stage), call_id(work, stage, row["qid"]),
                stage, work, identities, tokens)
            if stage == "reader":
                parsed = reader.parse_answer(output["raw_text"]) if output["status"] == "ok" else None
                arm["reader"] = {"status": "ok" if parsed is not None else "error", "raw_text": output["raw_text"],
                                 "parsed": parsed, "call": output, "transport_attempts": attempts}
            else:
                text = output["raw_text"].strip().lower()
                arm["judge"] = {"status": "ok" if output["status"] == "ok" and text in ("yes", "no") else "error",
                                "raw_text": output["raw_text"], "score": output["status"] == "ok" and text == "yes",
                                "call": output, "transport_attempts": attempts}
        arm["scores"] = recompute_scores(arm, allow_incomplete=True)
        save(report, work)
        print(json.dumps({"stage": stage, "done": done, "total": len(tasks), "status": arm[stage]["status"],
                          "correct": report["summary"]["correct"], "incremental_spend_usd": report["incremental_spend_usd"]}), flush=True)
    report["complete"] = all(row["systems"][SYSTEM]["arms"][ARM][part]["status"] != "pending"
                             for row in report["rows"] for part in ("reader", "judge"))
    save(report, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "reader", "judge", "run"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "run from the repository root")
    work = args.work.resolve()
    require(work.is_relative_to(RESULTS), "experiment must live in benchmark results")
    if args.stage == "prepare":
        prepare(work)
    else:
        report = json.loads((work / "answers.json").read_text())
        for stage in (("reader", "judge") if args.stage == "run" else (args.stage,)):
            run_stage(report, work, stage)
        print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
