"""Usage: uv run python -I benchmarks/public/batching_compare.py facts|embed|rerank <a> <b> [dim]

Compares two lme-model outputs. facts: raw LLM strings (identical text, parsed fact lists);
embed: little-endian f32 rows (cosine); rerank: p(yes) lists (abs diff)."""
import json
import re
import struct
import sys
import math

def parse(raw):
    s = raw.strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except Exception:
        return None
    f = v.get("facts") if isinstance(v, dict) else None
    if not isinstance(f, list):
        return None
    return [x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in f]

def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()

if __name__ == "__main__":
    kind, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    if kind == "facts":
        A, B = json.load(open(a)), json.load(open(b))
        assert len(A) == len(B)
        n = len(A)
        ident = [i for i in range(n) if A[i] == B[i]]
        diff = [i for i in range(n) if A[i] != B[i]]
        same_facts = [i for i in diff if parse(A[i]) is not None and parse(A[i]) == parse(B[i])]
        same_norm = [i for i in diff if parse(A[i]) is not None and parse(B[i]) is not None
                     and {norm(x) for x in parse(A[i])} == {norm(x) for x in parse(B[i])}]
        print(json.dumps({"items": n, "identical_text": len(ident),
            "different_text": diff,
            "different_text_same_parsed_facts": len(same_facts),
            "different_text_same_fact_set_normalized": len(same_norm),
            "different_parsed_facts": [i for i in diff if i not in same_facts],
            "parse_failures": [sum(parse(x) is None for x in A), sum(parse(x) is None for x in B)],
            "facts_total": [sum(len(parse(x) or []) for x in A), sum(len(parse(x) or []) for x in B)]}))
    elif kind == "embed":
        dim = int(sys.argv[4])
        def rows(p):
            d = open(p, "rb").read()
            v = struct.unpack(f"<{len(d)//4}f", d)
            return [v[i:i+dim] for i in range(0, len(v), dim)]
        A, B = rows(a), rows(b)
        assert len(A) == len(B)
        cos = []
        for x, y in zip(A, B):
            dot = sum(p*q for p, q in zip(x, y))
            nx = math.sqrt(sum(p*p for p in x))
            ny = math.sqrt(sum(q*q for q in y))
            cos.append(dot / (nx*ny))
        print(json.dumps({"items": len(A), "min_cosine": min(cos), "max_one_minus_cosine": 1 - min(cos),
            "max_abs_diff": max(abs(p-q) for x, y in zip(A, B) for p, q in zip(x, y)),
            "nan": sum(any(math.isnan(p) for p in x) for x in B)}))
    else:
        A, B = json.load(open(a)), json.load(open(b))
        d = [abs(x-y) for x, y in zip(A, B)]
        print(json.dumps({"items": len(A), "max_abs_diff_p_yes": max(d), "mean_abs_diff_p_yes": sum(d)/len(d)}))
