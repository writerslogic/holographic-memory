# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
result = {}
for path in sorted(root.rglob("*")):
    if path.is_file():
        with path.open("rb") as stream:
            result[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
print(json.dumps({"files_sha256": result}, sort_keys=True))
