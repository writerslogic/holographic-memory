"""Run the Bemula evaluation with HMS registered as baseline `hms`.

Run from a checkout of https://github.com/Nboxff/bemula-benchmark (pinned commit in
README.md) with this directory on PYTHONPATH; every argument after the script name is
passed to Bemula's `scripts.run_eval`. HMS_SERVER_URL names a running hms-server
(`--no-auth --input-dim 1536`); HMS_QUERY_LOG, if set, receives one JSON line per query
and per forget request.
"""

from __future__ import annotations

import os

import bemula.llm
import scripts.run_eval as run_eval

_upstream = run_eval._build_baseline
_upstream_extra_body = bemula.llm.thinking_extra_body


def _extra_body(model):
    # OpenRouter ignores Bemula's DeepSeek `thinking` switch; it takes `reasoning`, and the
    # provider pin keeps every call on DeepSeek's own deployment.
    if "openrouter.ai" in os.environ.get("OPENAI_BASE_URL", "") and "deepseek" in (model or ""):
        return {
            "reasoning": {"enabled": False},
            "provider": {"order": ["DeepSeek"], "allow_fallbacks": False},
        }
    return _upstream_extra_body(model)


bemula.llm.thinking_extra_body = _extra_body


def _build_baseline(name, llm, embedder_provider):
    if name != "hms":
        return _upstream(name, llm, embedder_provider)
    from bemula.baselines.rag import get_embedder

    from hms_adapter import HmsAdapter

    return HmsAdapter(
        llm,
        get_embedder(),
        os.environ.get("HMS_SERVER_URL", "http://127.0.0.1:8787"),
        top_k=int(os.environ.get("HMS_TOP_K", "5")),
        log_path=os.environ.get("HMS_QUERY_LOG"),
    )


run_eval._build_baseline = _build_baseline

if __name__ == "__main__":
    run_eval.main()
