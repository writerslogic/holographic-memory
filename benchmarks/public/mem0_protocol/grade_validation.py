# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared validation of the fixed binary top-200 grading protocol."""


def complete_grade(data: dict, hit_count: int) -> dict:
    """Return complete grades without selecting for correctness or nonempty answers."""
    grades = data.get("cutoff_results") if isinstance(data, dict) else None
    if not isinstance(grades, dict) or not isinstance(grades.get("top_200"), dict):
        raise ValueError("cannot reuse an incomplete top-200 grade")
    cutoff = grades["top_200"]
    score = cutoff.get("score")
    if (
        type(hit_count) is not int
        or not 0 <= hit_count <= 200
        or type(score) not in (int, float)
        or score not in (0, 1)
        or not isinstance(cutoff.get("generated_answer"), str)
        or type(cutoff.get("memories_evaluated")) is not int
        or cutoff["memories_evaluated"] != hit_count
    ):
        raise ValueError("cannot reuse an incomplete top-200 grade")
    return grades
