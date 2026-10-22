# Bemula forgetting benchmark

[Bemula](https://github.com/Nboxff/bemula-benchmark) (arXiv 2610.18052) measures whether a memory
system forgets on request: 50 multi-session dialogues with 135 in-dialogue forget requests of six
types (F1 factual, F2 compositional, F3 semantic category, F4 inferential, F5 time period, F6
replacement) and 1,742 probes in five suites (retention, efficacy, preservation, relearning,
attack). Answers are graded by an LLM judge. The suite has no license, so nothing from it is
copied here; run it from a checkout.

- `hms_adapter.py`: the `MemoryAdapter`. Each turn is one memory-API fact in hms-server (text,
  turn time, speaker, a `text-embedding-3-small` embedding); a probe recalls the top 5 facts by
  hybrid lexical + embedding search and the agent model answers from them alone. A user turn that
  asks to forget is classified by the agent model, every stored fact is reviewed against it,
  matches are deleted with `forget` + `purge`, and the parts of a matched fact that the request
  does not cover are stored again. The request turn and the reply after it are not stored; a
  replacement value is.
- `run_hms.py`: registers the adapter as baseline `hms` and runs Bemula's `scripts.run_eval`.
  Through OpenRouter it turns reasoning off and pins DeepSeek's own deployment, which Bemula's
  `thinking` switch does not do there; `--baseline rag` runs Bemula's Naive RAG the same way.
- `summarize.py`: writes `benchmarks/results/bemula_v1.json`.

```sh
cargo build --release --features server --bin hms-server
target/release/hms-server --no-auth --bind 127.0.0.1:8787 --input-dim 1536 --data-dir /tmp/hms-bemula
git clone https://github.com/Nboxff/bemula-benchmark && cd bemula-benchmark && git checkout 862aff6
# chat calls go to OPENAI_BASE_URL, embeddings to EMBED_PROXY_BASE_URL
export BEMULA_AGENT_MODEL=deepseek/deepseek-v4-flash-0731 BEMULA_JUDGE_MODEL=deepseek/deepseek-v4-flash-0731 \
       BEMULA_THINKING=disabled BEMULA_EMBED_BACKEND=api BEMULA_EMBED_MODEL=text-embedding-3-small
PYTHONPATH=<this directory>:. HMS_QUERY_LOG=/tmp/hms-bemula.jsonl uv run python -m run_hms \
  --baseline hms --dialogues data/full/dialogues.jsonl --questions data/full/questions.jsonl \
  --out /tmp/hms_full --workers 8 --resume
```

## What the HMS guarantee covers here

Bemula grades behavior through an LLM; it never compares stored bytes. The engine guarantee
(`docs/memory-api/README.md`, "Forgetting") is that a purged fact is gone: no search mode returns
it and the files match a store that never held it. The adapter checks the first half on every
probe and records any purged id that recall returns. Everything else a Bemula item depends on is
outside that guarantee and is reported as measured:

- which stored facts a request covers (adapter policy, decided by the agent model);
- F4 and multi-hop attacks, where the answer can be inferred from facts that were never covered;
- the agent model's own guesses (enumeration and jailbreak attacks answer yes/no questions).

`summarize.py` attributes each failed forget-variant efficacy or attack item to `engine` (recall
returned a purged id), `policy` (the target value was still in the retrieved context) or `model`
(it was not retrieved).
