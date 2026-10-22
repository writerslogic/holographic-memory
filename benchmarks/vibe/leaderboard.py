# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb==2.0.0"]
# ///
"""Reproduce VIBE's CPU overview from a saved copy of its three published inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


THRESHOLDS = (0.12, 0.73, 0.77, 0.90, 0.94, 0.95, 0.99, 1.0)


def summarize(directory, k=100):
    connection = duckdb.connect()
    for table, filename, reader in (
        ("summary", "summary.parquet", "read_parquet"),
        ("basics", "dataset_basics.csv", "read_csv_auto"),
        ("algorithms", "algorithms_basics.csv", "read_csv_auto"),
    ):
        connection.execute(f"create table {table} as select * from {reader}(?)", [str(directory / filename)])
    if not connection.execute("select count(*) from summary where k=?", [k]).fetchone()[0]:
        raise ValueError(f"published inputs have no k={k} results")
    connection.execute("""create table selected_datasets as select distinct dataset, type
        from basics join summary using(dataset) where not ends_with(type, '-additional')""")
    result = {"schema_version": 1, "k": k, "scope": "CPU, published default datasets; missing targets contribute zero",
              "metric": "mean of per-dataset QPS divided by that dataset's fastest eligible CPU QPS",
              "datasets": connection.execute("select dataset,type from selected_datasets order by dataset").fetchall(),
              "inputs": {}, "thresholds": {}, "per_dataset": {}}
    for filename in ("summary.parquet", "dataset_basics.csv", "algorithms_basics.csv"):
        payload = (directory / filename).read_bytes()
        remote = "results/summary.parquet" if filename.endswith("parquet") else filename
        result["inputs"][filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                                     "source": "https://vector-index-bench.github.io/" + remote}
    for threshold in THRESHOLDS:
        rows = connection.execute("""
            with cpu_algorithms as (
                select distinct algorithm from summary join algorithms using(algorithm) where not is_gpu
            ), best as (
                select dataset,algorithm,max(qps) qps from summary
                join selected_datasets using(dataset) join cpu_algorithms using(algorithm)
                where k=? and recall>=? group by all
            ), scaled as (
                select *,qps/max(qps) over(partition by dataset) score from best
            )
            select algorithm,avg(coalesce(score,0) order by dataset) mean_normalized_qps,
                   count(score) datasets_reached,count(*) datasets
            from selected_datasets cross join cpu_algorithms
            left join scaled using(dataset,algorithm)
            group by algorithm order by mean_normalized_qps desc,algorithm
        """, [k, threshold]).fetchall()
        result["thresholds"][str(threshold)] = [
            dict(zip(("algorithm", "mean_normalized_qps", "datasets_reached", "datasets"), row)) for row in rows
        ]
        detailed = connection.execute("""
            with best as (
                select dataset,algorithm,max(qps) qps from summary
                join selected_datasets using(dataset) join algorithms using(algorithm)
                where not is_gpu and k=? and recall>=? group by all
            )
            select dataset,algorithm,qps,qps/max(qps) over(partition by dataset) normalized_qps
            from best order by dataset,normalized_qps desc,algorithm
        """, [k, threshold]).fetchall()
        result["per_dataset"][str(threshold)] = [
            dict(zip(("dataset", "algorithm", "qps", "normalized_qps"), row)) for row in detailed
        ]
    connection.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.inputs, args.k), indent=2) + "\n")
