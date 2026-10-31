"""Scan every registry package/version in a uv lock, including inactive forks.

Usage: uv run --no-project --python 3.11 scan_lock.py PATH/uv.lock OUTPUT.json
Uses the free public OSV API; exits nonzero on incomplete responses.
"""
import datetime
import hashlib
import json
import pathlib
import sys
import tomllib
import urllib.request

lock = pathlib.Path(sys.argv[1]).read_bytes()
packages = tomllib.loads(lock.decode())["package"]
queries = [{"package": {"name": p["name"], "ecosystem": "PyPI"}, "version": p["version"]}
           for p in packages if "registry" in p["source"]]
results = []
for start in range(0, len(queries), 100):
    batch = queries[start:start + 100]
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
        data=json.dumps({"queries": batch}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = json.load(response)["results"]
    if len(page) != len(batch):
        raise RuntimeError("OSV returned an incomplete batch")
    for query, result in zip(batch, page):
        vulns = list(result.get("vulns", []))
        seen_tokens = set()
        while result.get("next_page_token"):
            token = result["next_page_token"]
            if token in seen_tokens:
                raise RuntimeError("OSV repeated a pagination token")
            seen_tokens.add(token)
            request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
                data=json.dumps({"queries": [{**query, "page_token": token}]}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)["results"][0]
            vulns.extend(result.get("vulns", []))
        results.append({"query": query, "vulns": vulns})
report = {"source": "https://api.osv.dev/v1/querybatch",
          "queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "lock_sha256": hashlib.sha256(lock).hexdigest(), "results": results}
pathlib.Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
print(f"Queried {len(results)} package/version pairs")
for result in results:
    if result["vulns"]:
        query = result["query"]
        print(query["package"]["name"], query["version"], len(result["vulns"]))
