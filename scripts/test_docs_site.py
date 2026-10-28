# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the docs-site generators. Run: python -m unittest scripts/test_docs_site.py

Each test pins a rule the site depends on: a number is never dropped or rounded in HMS's favour,
a generated reference fails loudly on input it does not understand, and canonical URLs never
point at a page that does not exist.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docs_charts  # noqa: E402
import docs_cheader  # noqa: E402
import docs_finalize  # noqa: E402
import docs_openapi  # noqa: E402
import docs_pyo3_stub  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class BrandingTests(unittest.TestCase):
    def test_brand_assets_resolve_within_each_site_version(self):
        import docs_site

        for url in ("https://docs.example/", "https://docs.example/dev/", "http://127.0.0.1:8123/latest/"):
            config = {"docs_dir": str(ROOT / "docs"), "site_url": url, "extra": {}, "theme": {}}
            docs_site.on_config(config)
            for key in ("logo", "favicon"):
                asset = config["theme"][key]
                self.assertTrue(asset.startswith(url))
                self.assertTrue((ROOT / "docs" / asset.removeprefix(url)).is_file())


class VectorChartTests(unittest.TestCase):
    def chart(self, table, competitors=(), hms=()):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "r.json").write_text(json.dumps({
                "dataset": {"name": "d"}, "environment": [{"cpu": "cpu"}], "max_load": 3.0,
                "qps_at_recall": table, "competitors": list(competitors), "hms": list(hms),
            }))
            spec = {"file": "r.json", "title": "t", "recall_targets": ["0.90", "0.95"], "hms_prefix": "HMS",
                    "annotations": {"HMS slow": "unconfirmed"}, "link": "x.md"}
            return docs_charts.vector_chart(Path(tmp), spec)

    def test_a_system_that_misses_a_target_stays_on_the_chart(self):
        out = self.chart({
            "HMS fast": {"0.90": {"qps": 900.0}, "0.95": {"qps": None, "method": "not reached"}},
            "Other": {"0.90": {"qps": 500.0}, "0.95": {"qps": 100.0}},
        })
        self.assertIn("not reached", out["svg"])
        self.assertIn("Other", out["svg"])
        self.assertTrue(any("no HMS configuration reached it" in line for line in out["summary"]))

    def test_summary_says_slower_when_a_competitor_wins(self):
        out = self.chart({
            "HMS fast": {"0.90": {"qps": 400.0}, "0.95": {"qps": None}},
            "Other": {"0.90": {"qps": 500.0}, "0.95": {"qps": 100.0}},
        })
        self.assertTrue(any("slower than" in line for line in out["summary"]))

    def test_annotated_rows_are_marked_and_excluded_from_the_headline_sentence(self):
        out = self.chart({
            "HMS slow": {"0.90": {"qps": 9000.0}, "0.95": {"qps": None}},
            "HMS fast": {"0.90": {"qps": 400.0}, "0.95": {"qps": None}},
            "Other": {"0.90": {"qps": 500.0}, "0.95": {"qps": 100.0}},
        })
        sentence = next(l for l in out["summary"] if l.startswith("At recall@10 of 0.90"))
        self.assertIn("400 QPS", sentence)
        self.assertNotIn("9,000", sentence)
        self.assertTrue(any(l.startswith("†") for l in out["summary"]))

    def test_load_gate_counts_every_configuration(self):
        out = self.chart(
            {"HMS fast": {"0.90": {"qps": 1.0}, "0.95": {"qps": 1.0}}},
            competitors=[{"load_gate_met": True}, {"load_gate_met": False}],
            hms=[{"sweep": [{"load_gate_met": False}]}],
        )
        self.assertEqual((out["gate_met"], out["gate_total"]), (1, 3))


class CHeaderTests(unittest.TestCase):
    HEADER = """#define HMS_H
// The ABI major version.
#define HMS_ABI_MAJOR 1

// An open store. Opaque to C.
typedef struct hms_store_t hms_store_t;

// Options.
typedef struct {
  // Size of this struct.
  uint32_t struct_size;
} hms_open_options_t;

// Open a store.
// # Safety
// `path` must be valid.
hms_status_t hms1_open(const hms_open_options_t *options, hms_store_t **out);
"""

    def test_declarations_keep_their_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp, "h.h")
            header.write_text(self.HEADER)
            text = docs_cheader.render(header)
        self.assertIn("### `hms1_open`", text)
        self.assertIn("**Safety**", text)
        self.assertIn("### `hms_open_options_t`", text)
        self.assertIn("struct_size;", text)
        self.assertNotIn("### `HMS_H`", text, "an include guard is not API")


class OpenApiTests(unittest.TestCase):
    SPEC = {
        "openapi": "3.0.3",
        "info": {"title": "hms-server", "version": "1"},
        "paths": {"/query": {"post": {
            "summary": "Query", "tags": ["Search"],
            "requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Q"}}}},
            "responses": {"200": {"description": "ok"}, "401": {"description": "no key"}},
        }}},
        "components": {"schemas": {"Q": {"type": "object", "required": ["vector"], "properties": {
            "vector": {"type": "array", "items": {"type": "number"}}, "top_k": {"type": "integer"}}}}},
    }

    def test_operations_and_schemas_render(self):
        text = docs_openapi.render(self.SPEC)
        self.assertIn("#### `POST /query`", text)
        self.assertIn("| `401` | no key |", text)
        self.assertIn("| `vector` | array of `number` | required |", text)
        self.assertIn("[`Q`](#schema-q)", text)


    def test_a_boolean_property_schema_means_any_value(self):
        spec = {"info": {"title": "t", "version": "1"}, "paths": {},
                "components": {"schemas": {"Free": {"type": "object", "properties": {"anything": True}}}}}
        self.assertIn("| `anything` |", docs_openapi.render(spec))


class Pyo3StubTests(unittest.TestCase):
    def test_current_and_historical_layouts_generate_the_same_api(self):
        source = (ROOT / "src" / "bindings" / "python" / "mod.rs").read_text()
        expected = docs_pyo3_stub.generate(ROOT / "src" / "bindings" / "python" / "mod.rs")
        for relative in ("src/bindings/python/mod.rs", "src/python.rs"):
            with self.subTest(layout=relative), tempfile.TemporaryDirectory() as temporary:
                tree = Path(temporary)
                path = tree / relative
                path.parent.mkdir(parents=True)
                path.write_text(source)
                hook = tree / "scripts" / "docs_site.py"
                hook.parent.mkdir()
                hook.write_bytes((ROOT / "scripts" / "docs_site.py").read_bytes())
                result = subprocess.run(
                    [sys.executable, "-I", "-c",
                     "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2])",
                     str(ROOT / "scripts"), str(hook)],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                stub = tree / ".docs-gen" / "pyi" / "holographic_memory" / "__init__.py"
                self.assertEqual(stub.read_text(), expected)

    def test_the_real_module_is_described(self):
        stub = docs_pyo3_stub.generate(ROOT / "src" / "bindings" / "python" / "mod.rs")
        self.assertIn("class PhaseHVec:", stub)
        self.assertIn("def resonator_factorize(", stub)
        self.assertIn("max_iter: int = 50", stub)
        compile(stub, "stub.pyi", "exec")

    def test_an_unknown_type_fails_instead_of_guessing(self):
        with self.assertRaises(docs_pyo3_stub.StubError):
            docs_pyo3_stub.py_type("HashMap<String, u8>", {}, "X")


class FinalizeTests(unittest.TestCase):
    def page(self, canonical):
        return f'<html><head><link rel="canonical" href="{canonical}"><meta property="og:url" content="{canonical}"></head></html>'

    def test_canonical_points_at_self_when_latest_lacks_the_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            for rel in ("latest/index.html", "dev/index.html", "dev/new/index.html"):
                (site / rel).parent.mkdir(parents=True, exist_ok=True)
                (site / rel).write_text(self.page("https://x/latest/"))
            fixed = docs_finalize.repair_canonicals(site, "https://x")
            self.assertEqual(fixed, 1)
            self.assertIn("https://x/dev/new/", (site / "dev/new/index.html").read_text())
            self.assertIn("https://x/latest/", (site / "dev/index.html").read_text())


if __name__ == "__main__":
    unittest.main()
