"""Usage: uv run python -I benchmarks/public/batching_inputs.py <longmemeval_s_cleaned.json> <out dir>

Builds batching-parity inputs from LongMemEval_S: the first 64 distinct haystack sessions
(first-seen order) as user-turn lists, user turns for embedding, and (question, turn) pairs."""
import json, sys
src, out = sys.argv[1], sys.argv[2]
data = json.load(open(src))
seen, sessions, ids = set(), [], []
for q in data:
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        if sid in seen:
            continue
        seen.add(sid)
        turns = [t["content"] for t in sess if t["role"] == "user"]
        if not turns:
            continue
        sessions.append(turns); ids.append(sid)
        if len(sessions) == 64:
            break
    if len(sessions) == 64:
        break
json.dump(sessions, open(f"{out}/facts64.json", "w"))
json.dump(ids, open(f"{out}/facts64_ids.json", "w"))
turns = [t for s in sessions for t in s]
# 96 user turns spread over the sessions, varied lengths.
step = max(1, len(turns) // 96)
docs = turns[::step][:96]
json.dump(docs, open(f"{out}/embed_docs.json", "w"))
qs = [q["question"] for q in data[:96]]
json.dump(qs, open(f"{out}/embed_queries.json", "w"))
pairs = [[qs[i % len(qs)], docs[i]] for i in range(len(docs))]
json.dump(pairs, open(f"{out}/rerank_pairs.json", "w"))
lens = sorted(len(" ".join(s).split()) for s in sessions)
print(len(sessions), "sessions; words min/med/max", lens[0], lens[len(lens)//2], lens[-1])
dl = sorted(len(d.split()) for d in docs)
print(len(docs), "docs; words min/med/max", dl[0], dl[len(dl)//2], dl[-1])
