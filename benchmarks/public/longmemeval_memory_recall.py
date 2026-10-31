# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LongMemEval dev questions through the WARM memory API, retrieval only (no model calls).

One warm-server tenant per question holds every embedded (user) haystack turn as a fact `"{role}: {content}"` with
the session date as event time and the cached all-MiniLM-L6-v2 turn embedding. Recall uses the
cached question embedding. Reported: answer-session recall@k (any / all answer sessions reached by
at least one of the top-k facts), per question type and overall. Dev questions only.

Usage:
  uv run --no-project --with openai==1.109.1 --with nltk==3.9.1 --with tiktoken==0.12.0 --with jinja2==3.1.6 \
    python benchmarks/public/longmemeval_memory_recall.py --server-bin target/release/warm-server \
    --cache-dir ~/.cache/warm-bench --out OUT.json [--order cosine]
"""

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'locomo'))
from locomo_bench import Server, cap_tokens, pack_by_similarity

KS = (5, 10, 30)
CUTS = (5, 10, 20)


class OnePerFact:
    """Token counter for the packing functions that makes every fact cost 1, so a budget is a fact count."""

    @staticmethod
    def encode(_text):
        return []


def event_ms(date):
    return int(datetime.strptime(date, '%Y/%m/%d (%a) %H:%M').replace(tzinfo=timezone.utc).timestamp() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server-bin', required=True)
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--recall-options', default='{}')
    ap.add_argument('--order', choices=('rank', 'cosine'), default='rank',
                    help='keep recall order, or reorder the recalled facts by query cosine before the cut')
    ap.add_argument('--split', action='store_true',
                    help='two tenants per question by session parity, packed to fact-count cuts (see --pack)')
    ap.add_argument('--pack', choices=('round-robin', 'zipper'), default='round-robin')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    root = Path(args.cache_dir).expanduser()
    dev = json.loads((root / 'longmemeval_sota_dev_v1/dev.json').read_text())
    lme = root / 'longmemeval_s'
    meta = json.loads((lme / 'meta.json').read_text())
    dim = meta['dim']
    order = [json.loads(line)['id'] for line in (lme / 'questions.jsonl').read_text().splitlines()]
    queries = np.fromfile(lme / 'queries.f32', dtype=np.float32).reshape(-1, dim)
    turns = np.fromfile(lme / 'turn.f32', dtype=np.float32).reshape(-1, dim)
    wanted = {order.index(qid) for qid in dev}
    turn_rows = {}
    with open(lme / 'turn.jsonl') as f:
        for line in f:
            t = json.loads(line)
            if t['q'] in wanted:
                turn_rows[(t['q'], t['id'])] = t
    server = Server(args.server_bin, dim, None)
    options = json.loads(args.recall_options)

    def run(qid):
        item, q = dev[qid], order.index(qid)
        tenant = f'lme-{qid}'
        facts, missing = [], 0
        sessions = zip(item['haystack_session_ids'], item['haystack_dates'], item['haystack_sessions'])
        # A haystack can repeat a session id with different content, so ids carry the position.
        for n, (sid, date, session) in enumerate(sessions):
            for j, turn in enumerate(session):
                row = turn_rows.get((q, f'{sid}_{j + 1}'))
                fact = {'id': f'{n}#{sid}#{j}', 'text': f"{turn['role']}: {turn['content']}", 'eventTime': event_ms(date)}
                # The cache embeds user turns only; mixing unembedded facts into hybrid recall would
                # skew it, so only embedded turns are stored.
                if row is not None and turn['content'].startswith(row['text'][:40]):
                    fact['embedding'] = turns[row['e']].tolist()
                    facts.append(fact)
                else:
                    missing += 1
        if args.split:
            # Two tenants by session-index parity: the two-list analog of LoCoMo's per-speaker memories.
            parts = {side: [f for f in facts if int(f['id'].split('#')[0]) % 2 == side] for side in (0, 1)}
            body = {'query': item['question'], 'k': 30, 'embedding': queries[q].tolist(), 'explain': True, **options}
            lists, sims = {}, {}
            for side, part in parts.items():
                for s in range(0, len(part), 64):
                    server.memory(f'{tenant}-{side}', 'remember', {'facts': part[s:s + 64]})
                results = server.memory(f'{tenant}-{side}', 'recall', body)['results'] if part else []
                lists[side] = [h['id'].split('#')[1] for h in results]
                sims[side] = [h['explanation'].get('semanticScore') or 0.0 for h in results]
            gold = set(item['answer_session_ids'])
            row = {'id': qid, 'type': item['question_type'], 'facts': len(facts), 'skipped': missing}
            for cut in CUTS:
                if args.pack == 'zipper':
                    kept = pack_by_similarity(lists, sims, OnePerFact, cut, True)
                    got = {lists[side][i] for side in kept for i in kept[side]}
                else:
                    got = {sid for side_ids in cap_tokens(lists, OnePerFact, cut).values() for sid in side_ids}
                row[f'any@{cut}'] = 1.0 if got & gold else 0.0
                row[f'all@{cut}'] = 1.0 if gold <= got else 0.0
            return row
        for s in range(0, len(facts), 64):
            server.memory(tenant, 'remember', {'facts': facts[s:s + 64]})
        body = {'query': item['question'], 'k': max(KS), 'embedding': queries[q].tolist(), **options}
        if args.order == 'cosine':
            body['explain'] = True
        results = server.memory(tenant, 'recall', body)['results']
        if args.order == 'cosine':
            # The LoCoMo adapter's similarity packing on a single list: the top max(KS) recalled facts
            # reordered by query cosine before the cut at k.
            results = sorted(results, key=lambda h: -(h['explanation'].get('semanticScore') or 0.0))
        hits = [h['id'].split('#')[1] for h in results]
        gold = set(item['answer_session_ids'])
        row = {'id': qid, 'type': item['question_type'], 'facts': len(facts), 'skipped': missing}
        for k in KS:
            got = set(hits[:k]) & gold
            row[f'any@{k}'] = 1.0 if got else 0.0
            row[f'all@{k}'] = 1.0 if got == gold else 0.0
        return row

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(run, sorted(dev)))
    finally:
        server.close()

    def block(rs):
        cuts = CUTS if args.split else KS
        return {'n': len(rs), **{key: round(statistics.fmean(r[key] for r in rs), 4)
                                 for k in cuts for key in (f'any@{k}', f'all@{k}')}}

    summary = {'questions': 'longmemeval_s dev (sota_dev_v1/dev.json)', 'embedding_model': meta['model'],
               'recall_options': options, 'order': args.order, 'split': args.split, 'pack': args.pack if args.split else None, 'skipped_unembedded_turns': sum(r['skipped'] for r in rows),
               'facts': sum(r['facts'] for r in rows), 'overall': block(rows),
               'by_type': {t: block([r for r in rows if r['type'] == t]) for t in sorted({r['type'] for r in rows})}}
    Path(args.out).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['overall']), 'skipped', summary['skipped_unembedded_turns'], 'kept', summary['facts'])


if __name__ == '__main__':
    main()
