# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Write the frozen LongMemEval dev / held-out split (benchmarks/public/longmemeval_split.json).

100 dev and 400 held-out question ids, stratified by (question_type, abstention) with
largest-remainder allocation and a fixed seed. S and M share the same 500 question ids, so one
split serves both. Tuning uses dev only; held-out is scored once, on the final M run.
"""

import json
import random
import sys
from pathlib import Path

SEED = 20261005
N_DEV = 100
OUT = Path(__file__).with_name("longmemeval_split.json")


def main(src: str) -> None:
    data = json.loads(Path(src).read_text())
    strata: dict[tuple[str, bool], list[str]] = {}
    for e in data:
        strata.setdefault((e["question_type"], "_abs" in e["question_id"]), []).append(e["question_id"])
    keys = sorted(strata)
    quota = {k: N_DEV * len(strata[k]) / len(data) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[: N_DEV - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(SEED)
    dev = []
    for k in keys:
        dev += rng.sample(sorted(strata[k]), alloc[k])
    dev_set = set(dev)
    split = {
        "seed": SEED,
        "stratified_by": ["question_type", "abstention (_abs in question_id)"],
        "strata": {f"{t}{' (abs)' if a else ''}": {"total": len(strata[(t, a)]), "dev": alloc[(t, a)]} for t, a in keys},
        "dev": sorted(dev),
        "heldout": sorted(e["question_id"] for e in data if e["question_id"] not in dev_set),
    }
    OUT.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {OUT}: dev={len(split['dev'])} heldout={len(split['heldout'])}")


if __name__ == "__main__":
    main(sys.argv[1])
