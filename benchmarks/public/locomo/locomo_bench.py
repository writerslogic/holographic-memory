# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LoCoMo through the HMS memory API, scored with Mem0's published protocol.

Ingest: one hms-server tenant per (conversation, speaker); every turn the speaker said is one fact
`"{speaker}: {text}"` with the session date as event time and the original date string as
metadata. Retrieval: one recall per speaker with k=30, rendered `"{timestamp}: {text}"` as Mem0's
harness renders its memories. Reader, judge, F1 and BLEU-1 follow mem0ai/mem0 evaluation/ at
393a4fd5 (see mem0_prompts.py). Only the question text reaches recall and the reader.

Usage:
  uv run --no-project --with openai==1.109.1 --with nltk==3.9.1 --with tiktoken==0.12.0 --with jinja2==3.1.6 \
    python benchmarks/public/locomo/locomo_bench.py --server-bin target/release/hms-server \
    --data ~/.cache/hms-bench/locomo/locomo10.json --convs conv-26,conv-30,conv-41 --out OUT
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nltk
import tiktoken
from jinja2 import Template
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mem0_prompts import ACCURACY_PROMPT, ANSWER_PROMPT, FULL_CONTEXT_PROMPT  # noqa: E402

READER_MODEL = 'gpt-4o-mini-2024-07-18'
JUDGE_MODEL = 'gpt-4o-mini-2024-07-18'
EMBED_MODEL = 'text-embedding-3-small'
EMBED_DIM = 1536
# USD per 1M tokens (OpenAI list price): input, output.
PRICES = {READER_MODEL: (0.15, 0.60), EMBED_MODEL: (0.02, 0.0)}
SPLIT = {
    'tuning': ['conv-26', 'conv-30', 'conv-41'],
    'heldout': ['conv-42', 'conv-43', 'conv-44', 'conv-47', 'conv-48', 'conv-49', 'conv-50'],
}
DIA = re.compile(r'D\d+:\d+')


class Cache:
    """Prompt-hash keyed store, so reruns and resumed runs pay once."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)')
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            row = self.db.execute('SELECT v FROM kv WHERE k=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO kv VALUES (?,?)', (key, json.dumps(value)))
            self.db.commit()


class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = {}

    def add(self, model, prompt, completion):
        with self.lock:
            p, c = self.tokens.get(model, (0, 0))
            self.tokens[model] = (p + prompt, c + completion)

    def usd(self):
        return round(sum(t[0] * PRICES[m][0] / 1e6 + t[1] * PRICES[m][1] / 1e6 for m, t in self.tokens.items()), 4)


def digest(*parts):
    return hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()


class Llm:
    def __init__(self, cache, usage):
        self.client = OpenAI(max_retries=8, timeout=120)
        self.cache, self.usage = cache, usage

    def chat(self, model, messages, tag, **extra):
        key = digest('chat', model, tag, json.dumps(messages, sort_keys=True), json.dumps(extra, sort_keys=True))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        r = self.client.chat.completions.create(model=model, messages=messages, temperature=0.0, **extra)
        self.usage.add(model, r.usage.prompt_tokens, r.usage.completion_tokens)
        text = r.choices[0].message.content
        self.cache.put(key, text)
        return text

    def embed(self, texts):
        out, todo = [None] * len(texts), []
        for i, t in enumerate(texts):
            hit = self.cache.get(digest('embed', EMBED_MODEL, t))
            if hit is None:
                todo.append(i)
            else:
                out[i] = hit
        for s in range(0, len(todo), 256):
            idx = todo[s:s + 256]
            r = self.client.embeddings.create(model=EMBED_MODEL, input=[texts[i] for i in idx])
            self.usage.add(EMBED_MODEL, r.usage.prompt_tokens, 0)
            for i, d in zip(idx, r.data):
                out[i] = d.embedding
                self.cache.put(digest('embed', EMBED_MODEL, texts[i]), d.embedding)
        return out


class Server:
    def __init__(self, binary, dim, config):
        self.dir = tempfile.mkdtemp(prefix='locomo-hms-')
        self.log = open(os.path.join(self.dir, 'server.log'), 'wb')
        cmd = [binary, '--no-auth', '--bind', '127.0.0.1:0', '--data-dir', os.path.join(self.dir, 'data'),
               '--input-dim', str(dim)]
        if config:
            cmd += ['--memory-config', config]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log)
        line = self.proc.stdout.readline().decode()
        m = re.search(r'listening on (\S+)', line)
        if not m:
            raise RuntimeError(f'hms-server did not start: {line!r}')
        self.base = f'http://{m.group(1)}'

    def memory(self, tenant, op, body):
        req = urllib.request.Request(f'{self.base}/api/v1/memory/{op}', data=json.dumps(body).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'X-Tenant-ID': tenant})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:300]
                if e.code not in (429, 503) or attempt == 7:
                    raise RuntimeError(f'{op} {tenant}: HTTP {e.code} {detail!r}') from e
                time.sleep(0.2 * 2 ** attempt)

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)
        self.log.close()


def session_turns(conv):
    """(session date string, epoch ms, turn) for every turn, in conversation order."""
    n = 1
    while f'session_{n}' in conv:
        date = conv[f'session_{n}_date_time']
        ms = int(datetime.strptime(date, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp() * 1000)
        for turn in conv[f'session_{n}']:
            yield date, ms, turn
        n += 1


def tenant(sample_id, side):
    return f'locomo-{sample_id}-{side}'


def ingest(server, llm, sample, use_embed):
    conv = sample['conversation']
    sides = {conv['speaker_a']: 'a', conv['speaker_b']: 'b'}
    facts = {'a': [], 'b': []}
    for date, ms, turn in session_turns(conv):
        facts[sides[turn['speaker']]].append({
            'id': turn['dia_id'], 'text': f"{turn['speaker']}: {turn['text']}", 'eventTime': ms,
            'metadata': {'timestamp': date, 'speaker': turn['speaker']}})
    for side, items in facts.items():
        if use_embed:
            for f, e in zip(items, llm.embed([f['text'] for f in items])):
                f['embedding'] = e
        for s in range(0, len(items), 64):
            server.memory(tenant(sample['sample_id'], side), 'remember', {'facts': items[s:s + 64]})
    return {side: {f['id']: f['metadata']['timestamp'] for f in items} for side, items in facts.items()}


def simple_tokenize(text):
    return str(text).lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()


def f1_score(prediction, reference):
    p, r = set(simple_tokenize(str(prediction).strip())), set(simple_tokenize(str(reference).strip()))
    if not p or not r:
        return 0.0
    common = p & r
    precision, recall = len(common) / len(p), len(common) / len(r)
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def bleu1(prediction, reference):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    pred = nltk.word_tokenize(str(prediction).lower())
    ref = [nltk.word_tokenize(str(reference).lower())]
    try:
        return sentence_bleu(ref, pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0


def cap_tokens(lists, enc, budget):
    """Keep memories round-robin by rank across both speakers while their tokens fit the budget."""
    kept = {side: [] for side in lists}
    used, rank, full = 0, 0, set()
    while len(full) < len(lists):
        for side, items in lists.items():
            if side in full or rank >= len(items):
                full.add(side)
                continue
            cost = len(enc.encode(json.dumps(items[rank]))) + 1
            if used + cost > budget:
                full.add(side)
                continue
            kept[side].append(items[rank])
            used += cost
        rank += 1
    return kept


FULL_CONTEXT_SYSTEM = ('You are a helpful assistant that can answer questions based on the provided context.'
                       'If the question involves timing, use the conversation date for reference.'
                       'Provide the shortest possible answer.'
                       'Use words directly from the conversation when possible.'
                       'Avoid using subjects in your answer.')


def answer_full_context(llm, sample, qa, args, enc):
    """Mem0's full-context baseline (evaluation/src/rag.py, chunk_size -1): no memory system."""
    context = ''.join(f"{date} | {turn['speaker']}: {turn['text']}\n"
                      for date, _, turn in session_turns(sample['conversation']))
    prompt = Template(FULL_CONTEXT_PROMPT).render(CONTEXT=context, QUESTION=qa['question'])
    messages = [{'role': 'system', 'content': FULL_CONTEXT_SYSTEM}, {'role': 'user', 'content': prompt}]
    responses = [llm.chat(READER_MODEL, messages, f'full-context-{run}') for run in range(args.runs)]
    return {'retrieved_a': [], 'retrieved_b': [], 'responses': responses,
            'context_tokens': len(enc.encode(context)), 'prompt_tokens_local': len(enc.encode(prompt))}


def answer(server, llm, sample, dates, qa, args, enc):
    conv = sample['conversation']
    question = qa['question']
    use_embed = args.embed
    body = {'query': question, 'k': args.k, **json.loads(args.recall_options)}
    if use_embed:
        body['embedding'] = llm.embed([question])[0]
    lists, ids = {}, {}
    for side in ('a', 'b'):
        hits = server.memory(tenant(sample['sample_id'], side), 'recall', body)['results']
        ids[side] = [h['id'] for h in hits]
        lists[side] = [f"{dates[side][h['id']]}: {h['text']}" for h in hits]
    if args.token_budget:
        lists = cap_tokens(lists, enc, args.token_budget)
        ids = {side: ids[side][:len(lists[side])] for side in ids}
    mem_a, mem_b = json.dumps(lists['a'], indent=4), json.dumps(lists['b'], indent=4)
    prompt = Template(ANSWER_PROMPT).render(
        speaker_1_user_id=conv['speaker_a'], speaker_2_user_id=conv['speaker_b'],
        speaker_1_memories=mem_a, speaker_2_memories=mem_b, question=question)
    responses = [] if args.retrieval_only else [
        llm.chat(READER_MODEL, [{'role': 'system', 'content': prompt}], 'reader' if run == 0 else f'reader-{run}')
        for run in range(args.runs)]
    return {'retrieved_a': ids['a'], 'retrieved_b': ids['b'], 'responses': responses,
            'context_tokens': len(enc.encode(mem_a)) + len(enc.encode(mem_b)),
            'prompt_tokens_local': len(enc.encode(prompt))}


def judge(llm, question, gold, generated, run):
    content = ACCURACY_PROMPT.format(question=question, gold_answer=gold, generated_answer=generated)
    text = llm.chat(JUDGE_MODEL, [{'role': 'user', 'content': content}], f'judge-{run}',
                    response_format={'type': 'json_object'})
    try:
        return 1 if json.loads(text)['label'] == 'CORRECT' else 0
    except (ValueError, KeyError, TypeError):
        return 0


def summarize(rows, runs):
    def block(rs):
        if not rs:
            return None
        def runwise(key):
            return [statistics.fmean(r[key][i] for r in rs) for i in range(runs)]
        out = {'n': len(rs), 'evidence_recall': round(statistics.fmean(r['evidence_recall'] for r in rs), 4),
               'context_tokens_mean': round(statistics.fmean(r['context_tokens'] for r in rs), 1)}
        if not runs:
            return out
        j = runwise('judge')
        return {**out, 'j': round(statistics.fmean(j), 4), 'j_std': round(statistics.stdev(j), 4) if runs > 1 else 0.0,
                'j_runs': [round(x, 4) for x in j],
                'f1': round(statistics.fmean(runwise('f1')), 4), 'bleu1': round(statistics.fmean(runwise('bleu1')), 4)}
    out = {'overall': block(rows)}
    for c in sorted({r['category'] for r in rows}):
        out[f'category_{c}'] = block([r for r in rows if r['category'] == c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server-bin', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--convs', required=True, help="comma-separated sample ids, or 'all'")
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default=os.path.expanduser('~/.cache/hms-bench/locomo/llm-cache.sqlite'))
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--embed', action='store_true', help=f'hybrid recall with {EMBED_MODEL} embeddings')
    ap.add_argument('--memory-config', default=None)
    ap.add_argument('--full-context', action='store_true', help="Mem0's full-context baseline (no memory system)")
    ap.add_argument('--runs', type=int, default=1, help='independent reader+judge passes (Mem0 reports 10)')
    ap.add_argument('--retrieval-only', action='store_true', help='skip reader and judge; report evidence recall')
    ap.add_argument('--recall-options', default='{}', help='extra RecallOptions JSON, e.g. {"fusion": "rrf"}')
    ap.add_argument('--token-budget', type=int, default=0,
                    help='cap the rendered memories of both speakers at this many o200k tokens (0: k only)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--hms-sha', default='')
    args = ap.parse_args()

    nltk_dir = os.path.expanduser('~/.cache/hms-bench/nltk')
    nltk.data.path.insert(0, nltk_dir)
    for pkg in ('punkt', 'punkt_tab'):
        nltk.download(pkg, download_dir=nltk_dir, quiet=True)

    data = json.load(open(args.data))
    wanted = [s['sample_id'] for s in data] if args.convs == 'all' else args.convs.split(',')
    samples = [s for s in data if s['sample_id'] in wanted]
    assert len(samples) == len(wanted), 'unknown sample id'
    usage, cache = Usage(), Cache(args.cache)
    llm, enc = Llm(cache, usage), tiktoken.get_encoding('o200k_base')
    server = None if args.full_context else Server(args.server_bin, EMBED_DIM, args.memory_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    try:
        for sample in samples:
            dates = None if args.full_context else ingest(server, llm, sample, args.embed)
            qas = [(i, qa) for i, qa in enumerate(sample['qa']) if int(qa['category']) != 5]
            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(lambda iq: answer_full_context(llm, sample, iq[1], args, enc) if args.full_context
                                        else answer(server, llm, sample, dates, iq[1], args, enc),
                                        qas))
            for (i, qa), res in zip(qas, results):
                gold = str(qa['answer'])
                evidence = set(DIA.findall(' '.join(map(str, qa.get('evidence', [])))))
                got = set(res['retrieved_a']) | set(res['retrieved_b'])
                rows.append({'sample_id': sample['sample_id'], 'qa_index': i, 'category': int(qa['category']),
                             'question': qa['question'], 'answer': gold, **res,
                             'evidence': sorted(evidence),
                             'evidence_recall': 1.0 if args.full_context or not evidence else len(evidence & got) / len(evidence),
                             'f1': [f1_score(x, gold) for x in res['responses']],
                             'bleu1': [bleu1(x, gold) for x in res['responses']]})
            print(f"{sample['sample_id']}: {len(qas)} questions, {time.time() - t0:.0f}s", file=sys.stderr)
    finally:
        if server:
            server.close()
    with ThreadPoolExecutor(args.workers) as pool:
        labels = list(pool.map(lambda r: [judge(llm, r['question'], r['answer'], x, run)
                                          for run, x in enumerate(r['responses'])], rows))
    for r, lab in zip(rows, labels):
        r['judge'] = lab
    with open(out / 'predictions.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    ran = {r['sample_id'] for r in rows}
    groups = {'run': rows}
    for name, ids in SPLIT.items():
        if set(ids) <= ran:
            groups[name] = [r for r in rows if r['sample_id'] in ids]
    summary = {
        'benchmark': 'LoCoMo (locomo10.json), categories 1-4', 'protocol': 'mem0ai/mem0 evaluation@393a4fd5',
        'reader_model': READER_MODEL, 'judge_model': JUDGE_MODEL,
        'arm': 'full-context' if args.full_context else 'hms-memory-api',
        'embedding_model': EMBED_MODEL if args.embed else None, 'k_per_speaker': args.k,
        'recall_options': json.loads(args.recall_options), 'token_budget': args.token_budget or None,
        'memory_config': json.load(open(args.memory_config)) if args.memory_config else None,
        'runs': 0 if args.retrieval_only else args.runs, 'conversations': sorted(ran), 'split': SPLIT, 'hms_sha': args.hms_sha,
        'new_api_tokens': {m: list(t) for m, t in usage.tokens.items()}, 'new_api_spend_usd': usage.usd(),
        'wall_seconds': round(time.time() - t0, 1),
        'results': {name: summarize(rs, 0 if args.retrieval_only else args.runs) for name, rs in groups.items()},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, res in summary['results'].items():
        print(name, json.dumps(res['overall']))
        for c in (1, 2, 3, 4):
            b = res.get(f'category_{c}')
            if b:
                print(f"  cat{c} n={b['n']} J={b.get('j')} F1={b.get('f1')} B1={b.get('bleu1')} ev={b['evidence_recall']}")
    print('spend_usd', usage.usd())


if __name__ == '__main__':
    main()
