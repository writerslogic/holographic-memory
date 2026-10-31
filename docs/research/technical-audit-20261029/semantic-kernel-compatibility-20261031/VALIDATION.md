# Semantic Kernel runtime compatibility blocker

At main `08776a83780e670056ec3881acbf98e1a95dd560`, Python 3.10 with the
declared minimum semantic-kernel 1.39.4 cannot instantiate WarmMemoryStore.
All 12 MemoryStoreBase abstract methods remain unimplemented: the adapter uses
obsolete `_async` suffixes. The retained script/log use the installed framework;
no server, model, credentials or paid service is involved.

Reproduce from the repository root:

```sh
PYTHONPATH=bindings/python/warm-sdk uv run --no-project --python 3.10 \
  --with 'semantic-kernel==1.39.4' --with 'sniffio>=1.3' python \
  docs/research/technical-audit-20261029/semantic-kernel-compatibility-20261031/reproduce.py
```

Renaming methods alone is unsafe: existing collection lifecycle methods are
stubs, raw record IDs collide across collection names, and nearest queries omit
collection filtering. Those are source observations, not reproduced runtime
cross-collection disclosures: ordinary instantiation is blocked first.

Acceptance for restoration: official MemoryStoreBase and SemanticTextMemory
workflows instantiate and pass against minimum/current supported versions;
collection identities and lifecycle survive restart; same record key in two
collections cannot overwrite/read/delete each other; query filters apply before
top-k; missing records and unsupported embedding options are explicit; async I/O
does not block the event loop; owned clients close correctly. Full record retrieval
and durable collection semantics must be designed before enabling old stubs.
No extra or Python version is removed by this finding.
