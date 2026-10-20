# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""mkdocs-gen-files script: the API reference, generated from the source of the tree being built.

  Rust     rustdoc (scripts/build-rustdoc.sh) is embedded under api/rust/doc/ by scripts/docs_site.py
  Python   mkdocstrings reads the SDK source, and a stub written from src/python.rs for the PyO3 module
  Node     typedoc reads every `types` entry in package.json "exports" (index.d.ts, semantic.d.ts, ...)
  HTTP     the OpenAPI document the server publishes, rendered as pages (when the tree has one)
  C        the shipped header, rendered as pages (when the tree has one)

A surface the tree does not have yet gets no page. Set HMS_DOCS_REQUIRE_API=1 (CI does) to make a
missing generator input, such as the typedoc install or the rustdoc output, fail the build instead
of silently dropping a page.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mkdocs_gen_files as gen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_cheader  # noqa: E402
import docs_openapi  # noqa: E402
import docs_pyo3_stub  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQUIRE = os.environ.get("HMS_DOCS_REQUIRE_API") == "1"
TOOLS = Path(os.environ.get("HMS_DOCS_API_TOOLS") or Path(__file__).resolve().parent / "docs-api")


def write(path: str, text: str, edit: str | None = None) -> None:
    with gen.open(path, "w") as handle:
        handle.write(text)
    if edit:
        gen.set_edit_path(path, "../" + edit)


def front(description: str, boost: float = 1.0) -> str:
    """Front matter for a generated page; `boost` weights it in site search."""
    return "---\ndescription: " + json.dumps(description) + f"\nsearch:\n  boost: {boost}\n---\n\n"


def missing(what: str) -> None:
    if REQUIRE:
        raise SystemExit(f"docs_api: {what} is required (HMS_DOCS_REQUIRE_API=1) and is not available")
    print(f"docs_api: skipping {what}")


# ---- Python -----------------------------------------------------------------------------------

def python_pages() -> list[str]:
    pages = []
    sdk = ROOT / "bindings" / "python" / "holographic-sdk" / "holographic_sdk"
    intro = {
        "client": ("HTTP clients", "`HolographicClient` and `AsyncHolographicClient` talk to `hms-server`, which stores vectors you supply and answers queries by exact cosine."),
        "memory": ("Memory clients", "The memory API over HTTP: record, recall, supersede and explain."),
        "evidence": ("Evidence packets", "Encode, decode and complete the evidence packets that carry sources with an answer."),
    }
    for module, (title, text) in intro.items():
        if (sdk / f"{module}.py").is_file():
            name = f"api/python/{module}.md"
            write(name, front(text) + f"# {title}\n\n{text}\n\n::: holographic_sdk.{module}\n",
                  f"bindings/python/holographic-sdk/holographic_sdk/{module}.py")
            pages.append(name)
    if (ROOT / "src" / "python.rs").is_file():
        stub_dir = ROOT / ".docs-gen" / "pyi"
        docs_pyo3_stub.write(ROOT, stub_dir)
        text = ("The `holographic-vsa` wheel is a different package from the SDK: the deterministic "
                "quantized-phase vector substrate, imported as `holographic_memory`. It does not include "
                "document search.")
        write("api/python/vsa.md", front(text) + f"# Phase vectors\n\n{text}\n\n::: holographic_memory\n", "src/python.rs")
        pages.append("api/python/vsa.md")
    return pages


# ---- Node -------------------------------------------------------------------------------------

def node_entry_points() -> list[Path]:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    found = []
    for target in (package.get("exports") or {}).values():
        types = target.get("types") if isinstance(target, dict) else None
        if types and (ROOT / types).is_file() and (ROOT / types) not in found:
            found.append(ROOT / types)
    return found


TITLE = re.compile(r"^# (?:Class|Interface|Function|Type Alias|Enumeration|Variable): (.*)$")


def node_pages() -> list[str]:
    entries = node_entry_points()
    typedoc = TOOLS / "node_modules" / ".bin" / "typedoc"
    if not entries or not typedoc.is_file():
        missing("the typedoc output (run `npm ci` in scripts/docs-api)")
        return []
    with tempfile.TemporaryDirectory(prefix="hms-typedoc-") as tmp:
        out = Path(tmp) / "md"
        tsconfig = Path(tmp) / "tsconfig.json"
        tsconfig.write_text(json.dumps({
            "compilerOptions": {"target": "es2022", "module": "nodenext", "moduleResolution": "nodenext",
                                "skipLibCheck": True, "strict": True, "types": [], "noEmit": True},
            "include": [str(p) for p in entries],
        }), encoding="utf-8")
        command = [
            str(typedoc), "--plugin", "typedoc-plugin-markdown", "--tsconfig", str(tsconfig),
            "--entryPoints", *map(str, entries), "--out", str(out), "--readme", "none",
            "--hidePageHeader", "--hideBreadcrumbs", "--disableSources", "--entryFileName", "index",
            "--parametersFormat", "table", "--propertiesFormat", "table", "--interfacePropertiesFormat", "table",
            "--classPropertiesFormat", "table", "--typeDeclarationFormat", "table", "--enumMembersFormat", "table",
            "--useCodeBlocks", "--outputFileStrategy", "members", "--logLevel", "Warn",
        ]
        subprocess.run(command, cwd=TOOLS, check=True)
        pages, modules, seen = [], {}, {}
        for md in sorted(out.rglob("*.md")):
            rel = md.relative_to(out)
            if len(rel.parts) == 1:
                continue  # the list of modules; rebuilt below
            module = rel.parts[0]
            if rel.name == "index.md":
                modules[module] = md.read_text(encoding="utf-8")
                continue
            if rel.name in seen:
                raise SystemExit(f"docs_api: two Node symbols are both named {rel.name}: {seen[rel.name]} and {rel}")
            seen[rel.name] = rel
        label = {"index": "holographic-memory"}
        for md in sorted(out.rglob("*.md")):
            rel = md.relative_to(out)
            if len(rel.parts) < 3:
                continue
            text = md.read_text(encoding="utf-8")
            first, _, rest = text.partition("\n")
            kind = rel.parts[1]
            heading = TITLE.match(first)
            name = heading.group(1).rstrip("()") if heading else md.stem
            text = f"# {name}\n" + rest
            text = re.sub(r"\]\(((?:\.\./|\./)*)(?:[\w-]+/)*([\w.-]+\.md)", r"](\2", text)
            module_name = label.get(rel.parts[0], f"holographic-memory/{rel.parts[0]}")
            desc = f"{kind.replace('-', ' ').rstrip('s').capitalize()} `{name}` from `{module_name}`."
            target = f"api/node/{md.stem}.md"
            write(target, front(desc, 0.5) + text, None)
            pages.append(target)
        lines = [front("The Node and TypeScript API of the holographic-memory package, generated from its type declarations.").rstrip(), "",
                 "# Node and TypeScript API", "",
                 "Generated from the package's `.d.ts` files, one page per exported symbol. "
                 "The package is built from source with a Rust toolchain; the [quickstart](../../quickstart/index.md) shows how.", ""]
        for module, listing in modules.items():
            lines += [f"## `{label.get(module, 'holographic-memory/' + module)}`", ""]
            for line in listing.splitlines():
                if line.startswith("## "):
                    lines += ["", f"### {line[3:]}", ""]
                else:
                    lines.append(re.sub(r"\]\(((?:\.\./|\./)*)(?:[\w-]+/)*([\w.-]+\.md)", r"](\2", line) if line.startswith("- [") else "")
        write("api/node/index.md", "\n".join(lines).rstrip() + "\n", "package.json")
        pages.append("api/node/index.md")
    return pages


# ---- Rust, HTTP, C ----------------------------------------------------------------------------

def rust_pages() -> list[str]:
    docs = os.environ.get("HMS_RUSTDOC_DIR")
    version = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M).group(1)
    if not docs or not (Path(docs) / "holographic_memory" / "index.html").is_file():
        missing("the rustdoc output (HMS_RUSTDOC_DIR, built by scripts/build-rustdoc.sh)")
        return []
    text = front("The Rust API of the holographic-memory crate, generated by rustdoc from the source of this version.") + f"""# Rust API reference

The reference is the crate's own rustdoc, built from the source of this version of the docs with
the `provenance` feature (which includes `security`). It is a separate, searchable site.

<p><a class="hms-btn hms-btn--primary" href="doc/holographic_memory/index.html">Open the rustdoc for {version}</a></p>

Released versions are also on [docs.rs](https://docs.rs/holographic-memory). Experimental modules sit behind the
`experimental` feature and are not part of the published API, so they are not in this reference.
"""
    write("api/rust/index.md", text, "src/lib.rs")
    return ["api/rust/index.md"]


def http_pages() -> list[str]:
    path = docs_openapi.find_spec(ROOT)
    if not path:
        return []
    write("api/http.md", docs_openapi.render(docs_openapi.load(path)), str(path.relative_to(ROOT)))
    return ["api/http.md"]


def c_pages() -> list[str]:
    header = docs_cheader.find_header(ROOT)
    if not header:
        return []
    write("api/c.md", docs_cheader.render(header), str(header.relative_to(ROOT)))
    return ["api/c.md"]


def overview(pages: list[str]) -> None:
    rows = {
        "api/rust/index.md": ("Rust", "The `holographic-memory` crate: the engine itself."),
        "api/node/index.md": ("Node and TypeScript", "The `holographic-memory` npm package."),
        "api/http.md": ("HTTP", "The `hms-server` endpoints, from its OpenAPI document."),
        "api/c.md": ("C", "The stable C ABI that the Go, Java and .NET bindings sit on."),
    }
    lines = [front("The API reference for Holographic Memory in every language, generated from the source so it cannot drift").rstrip(), "",
             "# API reference", "",
             "Every page in this section is generated at build time from the source of the version you are reading: "
             "rustdoc for Rust, typedoc for Node and TypeScript, mkdocstrings for Python, the server's OpenAPI document for HTTP and the "
             "shipped header for C. Nothing here is written by hand, so a signature in the docs is a signature in the code.", "",
             "| Surface | What it covers |", "| --- | --- |"]
    for page, (name, text) in rows.items():
        if page in pages:
            lines.append(f"| [{name}]({page.removeprefix('api/')}) | {text} |")
    py = [p for p in pages if p.startswith("api/python/")]
    if py:
        lines.append(f"| [Python]({py[0].removeprefix('api/')}) | The SDK clients and the phase-vector module. |")
    write("api/index.md", "\n".join(lines) + "\n", None)


all_pages = [*rust_pages(), *python_pages(), *node_pages(), *http_pages(), *c_pages()]
overview(all_pages)
