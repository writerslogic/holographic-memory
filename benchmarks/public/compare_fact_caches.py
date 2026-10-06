"""Compare two cached LongMemEval fact-extraction outputs (Python/vLLM vs HMS local-models) key by key.

  uvx modal volume get hms-lme cache/facts/<model>@<rev12> py
  uvx modal volume get hms-lme cache/facts/<model>@<rev12>+hms hms
  uv run python benchmarks/public/compare_fact_caches.py <dir holding py/ and hms/> <out.json>
"""

import json
import re
import sys
from pathlib import Path


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*/*.json")):
        out.update(json.loads(f.read_text()))
    return out


def facts(v) -> list[str] | None:
    """The fact list of one parsed LLM output, or None if the output did not parse."""
    f = v.get("facts") if isinstance(v, dict) else None
    if not isinstance(f, list):
        return None
    return [x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in f]


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def tokens(s: str) -> set[str]:
    return set(norm(s).split())


def best_jaccard(a: str, others: list[str]) -> float:
    ta = tokens(a)
    return max((len(ta & tokens(b)) / max(1, len(ta | tokens(b))) for b in others), default=0.0)


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    py, hm = load(root / "py"), load(root / "hms")
    keys = sorted(k for k in hm if k in py)
    both = [k for k in keys if facts(py[k]) is not None and facts(hm[k]) is not None]
    rec = [best_jaccard(x, facts(hm[k])) for k in both for x in facts(py[k])]
    pre = [best_jaccard(x, facts(py[k])) for k in both for x in facts(hm[k])]
    res = {
        "sessions_compared": len(keys),
        "parse_failures": {"python": sum(facts(py[k]) is None for k in keys),
                           "hms": sum(facts(hm[k]) is None for k in keys)},
        "identical_output": sum(facts(py[k]) == facts(hm[k]) for k in both),
        "same_fact_set_after_normalization": sum(
            {norm(x) for x in facts(py[k])} == {norm(x) for x in facts(hm[k])} for k in both),
        "agree_on_empty_vs_nonempty": sum((not facts(py[k])) == (not facts(hm[k])) for k in both),
        "facts_total": {"python": sum(len(facts(py[k])) for k in both),
                        "hms": sum(len(facts(hm[k])) for k in both)},
    }
    for name, vals in (("python_facts_with_hms_match", rec), ("hms_facts_with_python_match", pre)):
        for t in (0.8, 0.5):
            res[f"{name}_jaccard_ge_{t}"] = sum(v >= t for v in vals) / max(1, len(vals))
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
