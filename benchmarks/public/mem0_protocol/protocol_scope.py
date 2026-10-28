# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve expected question IDs using the pinned harness's own selectors."""

import importlib
import json
import sys
from pathlib import Path


def expected_ids(clone: Path, command: list[str]) -> list[str]:
    modules = {"benchmarks.locomo.run", "benchmarks.longmemeval.run", "benchmarks.beam.run"}
    if command[:1] != ["-m"] or command[1] not in modules:
        raise ValueError("unsupported harness module")
    sys.path.insert(0, str(clone))
    module = importlib.import_module(command[1])
    sys.argv = [command[1], *command[2:]]
    args = module.parse_args()
    if command[1] == "benchmarks.locomo.run":
        path = Path(args.dataset_path) if args.dataset_path else clone / module.DEFAULT_DATASET_DIR / module.DEFAULT_DATASET_FILE
        dataset = module.load_dataset(str(path))
        items = module.expected_locomo_question_items(
            dataset, [int(value) for value in args.conversations.split(",")],
            [int(value) for value in args.categories.split(",")], args.max_questions,
        )
        return [item[0] for item in items]
    if command[1] == "benchmarks.longmemeval.run":
        path = Path(args.dataset_path) if args.dataset_path else clone / module.DEFAULT_DATASET_DIR / module.DEFAULT_DATASET_FILE
        dataset = module.load_dataset(str(path))
        if not args.all_questions or args.question_types:
            raise ValueError("LongMemEval protocol requires all questions in the supplied dataset")
        return [item["question_id"] for item in dataset]
    indices = module._parse_conversation_indices(args.conversations)
    selected_types = set(args.question_types.split(",")) if args.question_types else None
    cache = Path(args.dataset_cache_dir) if args.dataset_cache_dir else clone / module.DEFAULT_DATASET_DIR
    ids = []
    for size in args.chat_sizes.split(","):
        dataset = json.loads((cache / f"beam_{size}.json").read_text())
        for index in indices:
            if index >= len(dataset):
                continue
            questions = module.extract_probing_questions(dataset[index])
            if selected_types:
                questions = [question for question in questions if question.get("question_type") in selected_types]
            ids.extend(f"{size}_{index}_q{number}_{question.get('question_type', 'unknown')}"
                       for number, question in enumerate(questions))
    return ids


if __name__ == "__main__":
    clone, command = Path(sys.argv[1]), json.loads(sys.argv[2])
    print(json.dumps(expected_ids(clone, command)))
