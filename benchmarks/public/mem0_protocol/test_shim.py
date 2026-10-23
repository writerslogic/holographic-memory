# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""uv run --no-project --with pytest --with aiohttp==3.13.2 pytest benchmarks/public/mem0_protocol/test_shim.py"""

import re
from datetime import datetime, timezone

import shim


def test_tenant_ids_are_valid_and_injective_on_harness_ids():
    for uid in ("locomo_0_4c92d691", "longmemeval_gpt4_2655b836_abs_1a2b3c4d", "beam_10M_3_deadbeef"):
        assert shim.tenant_for(uid) == uid
    for uid in ("a/b", "x" * 65, "q.1", "ü"):
        t = shim.tenant_for(uid)
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", t)
    assert shim.tenant_for("a/b") != shim.tenant_for("a.b")


def test_split_fact_boundaries():
    word = "abcdefghi"  # 9 bytes + 1 separator
    at_limit = " ".join([word] * shim.MAX_FACT_WORDS)
    assert len(at_limit.encode()) <= shim.MAX_FACT_BYTES
    assert shim.split_fact(at_limit) == [at_limit]
    assert shim.split_fact(" ".join([word] * (shim.MAX_FACT_WORDS - 1))) == [" ".join([word] * (shim.MAX_FACT_WORDS - 1))]
    over = " ".join([word] * (shim.MAX_FACT_WORDS + 1))
    parts = shim.split_fact(over)
    assert len(parts) == 2 and " ".join(parts) == over
    big = "y" * 70_000
    parts = shim.split_fact(big)
    assert len(parts) > 1 and all(len(p.encode()) <= shim.MAX_FACT_BYTES for p in parts)
    assert "".join(p.replace(" ", "") for p in parts) == big
    long_bytes = " ".join(["z" * 999] * 70)  # 70,000 bytes, 70 words
    parts = shim.split_fact(long_bytes)
    assert all(len(p.encode()) <= shim.MAX_FACT_BYTES for p in parts) and " ".join(parts) == long_bytes


def test_fact_text_keeps_harness_strings_verbatim():
    msgs = [{"role": "user", "content": "Caroline: hi [Sharing image - query: x. The image shows: y]"},
            {"role": "assistant", "content": "ok"}]
    assert shim.fact_text(msgs) == "user: Caroline: hi [Sharing image - query: x. The image shows: y]\nassistant: ok"


def test_created_at_parses_with_the_harness_date_rule():
    # memory-benchmarks _to_human_date: strptime(iso[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S") first.
    epoch = int(datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc).timestamp())
    iso = shim.iso_from_ms(epoch * 1000)
    assert iso == "2023-05-08T13:56:00"
    assert datetime.strptime(iso[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S").strftime("%A, %B %d, %Y") == "Monday, May 08, 2023"
    assert shim.iso_from_ms(None) is None
    # lexicographic order of the strings equals time order (the harness sorts them as strings)
    assert shim.iso_from_ms(1_000) < shim.iso_from_ms(86_400_000) < shim.iso_from_ms(10**12)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"max_k": 200, "recall": {"lexicalWeight": 1}}')
    assert shim.load_config(str(p))["max_k"] == 200
    p.write_text('{"max_k": 200, "answers": {}}')
    try:
        shim.load_config(str(p))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown key accepted")
