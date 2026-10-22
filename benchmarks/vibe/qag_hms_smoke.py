# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real production VIBE HMS wrapper and native patience transitions."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def array_digest(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def answers(index, queries, rows, count=32):
    results = []
    for query in queries:
        ids = np.asarray(index.query(query, count))
        if (ids.shape != (count,) or ids.dtype.kind not in "iu"
                or len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= rows)):
            raise ValueError("native smoke query returned invalid or duplicate IDs")
        results.append(ids)
    return np.asarray(results, dtype="<i8")


def wrapper_state(index):
    return index.ef, index.rerank, index.patience


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibe", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    vibe, extension, wrapper = (path.expanduser().resolve(strict=True) for path in
                                (args.vibe, args.extension, args.wrapper))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(vibe))
    from vibe.algorithms.qag_pair.module import _load_extension

    native = _load_extension(extension, "_hms_production_smoke_candidate")
    sys.modules["holographic_memory"] = native
    package = types.ModuleType("vibe.algorithms.hms")
    package.__path__ = [str(wrapper.parent)]
    sys.modules[package.__name__] = package
    name = "vibe.algorithms.hms.module"
    spec = importlib.util.spec_from_file_location(name, wrapper)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HMS wrapper {wrapper}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    package.module = module

    rng = np.random.default_rng(24301)
    base = rng.normal(size=(256, 16)).astype(np.float32)
    learn = rng.normal(size=(32, 16)).astype(np.float32)
    queries = rng.normal(size=(8, 16)).astype(np.float32)
    input_hashes = {"base": array_digest(base), "learn": array_digest(learn),
                    "evaluation": array_digest(queries)}
    neighbors = np.argsort(-(learn.astype(np.float64) @ base.astype(np.float64).T),
                          axis=1, kind="stable")[:, :100].astype(np.uint32)
    trained = module.HMS("ip", 16, 64, code_bits=8, query_aware=True)
    trained.fit_ood(base, learn, neighbors)
    trained.set_query_arguments(64, 32)
    zero = answers(trained, queries, len(base))
    positive = None
    for vector in ((64, 32, 8), [64, 32, 8]):
        trained.set_query_arguments(vector)
        if wrapper_state(trained) != (64, 32, 8):
            raise ValueError("vector query arguments did not set patience")
        current = answers(trained, queries, len(base))
        if positive is not None:
            np.testing.assert_array_equal(current, positive)
        positive = current
        trained.set_query_arguments([64, 32])
        if wrapper_state(trained) != (64, 32, 0):
            raise ValueError("two-value query arguments did not reset patience")
        np.testing.assert_array_equal(answers(trained, queries, len(base)), zero)
    trained.set_query_arguments(64, 32, 8)
    np.testing.assert_array_equal(answers(trained, queries, len(base)), positive)
    if wrapper_state(trained) != (64, 32, 8):
        raise ValueError("three separate arguments did not set patience")
    before = wrapper_state(trained)
    try:
        trained.set_query_arguments(0, 32, 8)
    except (ValueError, OverflowError) as error:
        rejection = type(error).__name__
    else:
        raise ValueError("native ef=0 was accepted")
    if wrapper_state(trained) != before:
        raise ValueError("rejected native arguments changed wrapper fields")
    np.testing.assert_array_equal(answers(trained, queries, len(base)), positive)

    legacy = module.HMS("ip", 16, 64)
    legacy.fit(base)
    legacy.set_query_arguments([64, 32])
    legacy_answers = answers(legacy, queries, len(base))
    default = native.VectorIndex("ip", degree=16, build_ef=64, threads=1)
    default.fit(base)
    default.set_query_arguments(64, 32)
    np.testing.assert_array_equal(answers(default, queries, len(base)), legacy_answers)
    if legacy.index.index_bytes != default.index_bytes:
        raise ValueError("legacy wrapper allocation differs from native defaults")
    baseline_info = None
    if args.baseline is not None:
        baseline_path = args.baseline.expanduser().resolve(strict=True)
        baseline_native = _load_extension(baseline_path, "_hms_production_smoke_baseline")
        baseline = baseline_native.VectorIndex("ip", degree=16, build_ef=64, threads=1)
        baseline.fit(base)
        baseline.set_query_arguments(64, 32)
        np.testing.assert_array_equal(answers(baseline, queries, len(base)), legacy_answers)
        if baseline.index_bytes != legacy.index.index_bytes:
            raise ValueError("legacy wrapper allocation differs from baseline")
        baseline_info = {"path": str(baseline_path), "sha256": file_digest(baseline_path),
                         "index_bytes": int(baseline.index_bytes)}
    if input_hashes != {"base": array_digest(base), "learn": array_digest(learn),
                        "evaluation": array_digest(queries)}:
        raise ValueError("native integration changed caller matrices")
    print(json.dumps({
        "schema_version": 1, "scope": "synthetic real-native integration; no timing measurement",
        "seed": 24301, "base_shape": list(base.shape), "learn_shape": list(learn.shape),
        "evaluation_shape": list(queries.shape), "vibe": str(vibe),
        "extension": {"path": str(extension), "sha256": file_digest(extension)},
        "wrapper": {"path": str(wrapper), "sha256": file_digest(wrapper)},
        "smoke_sha256": file_digest(Path(__file__)), "baseline": baseline_info,
        "input_sha256": input_hashes,
        "result_sha256": {"zero_patience": array_digest(zero),
                          "positive_patience": array_digest(positive),
                          "legacy4": array_digest(legacy_answers)},
        "checks": {"tuple_and_list_positive_patience": True, "two_value_reset": True,
                   "three_separate_arguments": True, "native_rejection": rejection,
                   "rejection_preserves_wrapper_and_native_state": True,
                   "legacy_matches_native_defaults": True,
                   "legacy_matches_optional_baseline": True if baseline_info is not None else None,
                   "caller_matrices_unchanged": True},
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
