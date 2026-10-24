import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

records = defaultdict(lambda: defaultdict(list))
recalls = {}
encodings = []
text = Path(sys.argv[1]).read_text()
for line in text.splitlines():
    fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
    if line.startswith("encode "):
        encodings.append(fields)
    elif line.startswith("recall "):
        key = (fields["mode"], int(fields["ef"]), fields["deleted"], int(fields["threads"]))
        recalls.setdefault(key, {})[fields["arm"]] = int(fields["hits"])
    elif line.startswith("timing "):
        key = (fields["mode"], int(fields["ef"]), fields["deleted"], int(fields["threads"]))
        records[key][(int(fields["pair"]), fields["candidate"])].append(int(fields["ns"]))
rows = []
for key, arms in records.items():
    mode, ef, deleted, threads = key
    ratios = []
    complete = len(arms) == 6 and all(len(v) == 5 for v in arms.values())
    for pair in range(3):
        base, candidate = arms.get((pair, "false")), arms.get((pair, "true"))
        if base and candidate:
            ratios.append(statistics.median(candidate) / statistics.median(base))
    recall = recalls.get(key, {})
    base_width = ef // 2 if mode == "Binary" else ef
    row = dict(mode=mode, ef=ef, deleted=deleted, threads=threads, complete=complete,
               recall_hits=recall, pair_ratios=ratios,
               median_ratio=statistics.median(ratios) if ratios else None)
    if "candidate" in recall and f"base{base_width}" in recall:
        row["recall_delta"] = (recall["candidate"] - recall[f"base{base_width}"]) / 1280
    rows.append(row)
json.dump(dict(log=Path(sys.argv[1]).name, complete="EXIT_STATUS 0" in text,
               encodings=encodings, rows=rows), sys.stdout, indent=2)
print()
