# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import importlib
import json
from pathlib import Path
import sys
import types


repo = Path(__file__).resolve().parents[3]
# Load the pure importer mapper without importing the unrelated HTTP SDK dependency.
package = types.ModuleType("hms_audit_importers")
package.__path__ = [
    str(repo / "bindings/python/holographic-sdk/holographic_sdk/importers")
]
sys.modules[package.__name__] = package
mapping = importlib.import_module(package.__name__ + ".mapping")
model = importlib.import_module(package.__name__ + ".model")
options = importlib.import_module(package.__name__ + ".options")
report = importlib.import_module(package.__name__ + ".report")

opts = options.ImportOptions()
left = mapping.hms_id("a", "b::c", opts)
right = mapping.hms_id("a::b", "c", opts)
assert left == right

mapper = mapping.Mapper(
    "audit",
    options.ImportOptions(vectors="reencode", max_text_words=1, max_text_bytes=1024),
    mapping.Target(server_embeds=True),
    report.Stats(),
)
parts = mapper.map(model.Record(id="doc", text="alpha beta"))
literal = mapper.map(model.Record(id="doc#part-0", text="gamma"))
assert len(parts) == 2
assert parts[0].id == literal[0].id

document_bytes = ctypes.c_int32(120 * (1 << 29)).value
document_id_bytes = ctypes.c_int32(16 * (1 << 28)).value
assert document_bytes == document_id_bytes == 0

print(
    json.dumps(
        {
            "importer_namespace_collision": [left, right],
            "importer_part_collision": [parts[0].id, literal[0].id],
            "dotnet_x64_document_struct_bytes": 120,
            "dotnet_unchecked_int32_document_allocation_bytes": document_bytes,
            "dotnet_unchecked_int32_id_allocation_bytes": document_id_bytes,
            "unsafe_native_writes_executed": False,
        },
        indent=2,
    )
)
