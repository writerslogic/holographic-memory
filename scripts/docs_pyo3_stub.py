# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Writes a .pyi stub for the PyO3 module from src/python.rs, with its doc comments as docstrings.

mkdocstrings reads the stub statically, so the Python reference for the compiled module needs no
wheel and matches whichever source tree is being documented (a release tag included). Anything
the generator does not understand raises, so the build fails instead of documenting a guess.
scripts/test_docs_pyo3_stub.py compares the stub with the compiled module when it is installed.
"""

import re
import sys
from pathlib import Path

PRIMITIVES = {
    "usize": "int", "u8": "int", "u16": "int", "u32": "int", "u64": "int", "i32": "int", "i64": "int",
    "f32": "float", "f64": "float", "bool": "bool", "String": "str", "str": "str",
}


class StubError(Exception):
    pass


def py_type(rust: str, classes: dict, self_name: str) -> str:
    rust = rust.strip().lstrip("&").replace("mut ", "").strip()
    if rust in ("Self", "Bound<'_, Self>"):
        return self_name
    if rust in PRIMITIVES:
        return PRIMITIVES[rust]
    if rust in classes:
        return classes[rust]
    m = re.fullmatch(r"Vec<(.+)>", rust)
    if m:
        return f"list[{py_type(m.group(1), classes, self_name)}]"
    m = re.fullmatch(r"Option<(.+)>", rust)
    if m:
        return f"{py_type(m.group(1), classes, self_name)} | None"
    m = re.fullmatch(r"PyResult<(.+)>", rust)
    if m:
        return py_type(m.group(1), classes, self_name)
    if rust.startswith("(") and rust.endswith(")"):
        parts = _split(rust[1:-1])
        return "tuple[" + ", ".join(py_type(p, classes, self_name) for p in parts) + "]"
    raise StubError(f"unsupported Rust type in the Python surface: {rust!r}")


def _split(text: str):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _docs(lines):
    return "\n".join(l.strip()[3:].strip() for l in lines if l.strip().startswith("///")).strip()


def _quote(doc: str, indent: str) -> str:
    doc = doc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    lines = doc.splitlines()
    return indent + '"""' + ("\n" + indent).join(lines) + '"""'


def _fn_blocks(text: str):
    """Yield (doc_lines, attrs, signature_text, is_in_impl_of) for each fn with its leading attrs."""
    pattern = re.compile(
        r"(?P<lead>(?:[ \t]*(?:///[^\n]*|#\[[^\n]*\])\n)*)[ \t]*fn\s+(?P<name>\w+)\s*\((?P<args>.*?)\)\s*(?:->\s*(?P<ret>[^{]+?))?\s*\{",
        re.S,
    )
    for m in pattern.finditer(text):
        lead = m.group("lead").splitlines()
        yield (
            [l for l in lead if l.strip().startswith("///")],
            [l.strip() for l in lead if l.strip().startswith("#[")],
            m.group("name"), m.group("args"), (m.group("ret") or "").strip(),
        )


def generate(rs_path: Path) -> str:
    text = rs_path.read_text(encoding="utf-8")
    module_doc = re.search(r'"__doc__",\s*"([^"]+)"', text)
    classes = {}
    for m in re.finditer(r'#\[pyclass\(name\s*=\s*"(\w+)"[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*struct\s+(\w+)', text):
        classes[m.group(2)] = m.group(1)
    out = ['"""' + (module_doc.group(1) if module_doc else "") + '"""', "", "from typing import Final", ""]
    for rust_name, py_name in classes.items():
        decl = re.search(r"((?:[ \t]*///[^\n]*\n)*)[ \t]*#\[pyclass\(name\s*=\s*\"" + py_name, text)
        out.append(f"class {py_name}:")
        doc = _docs(decl.group(1).splitlines()) if decl else ""
        if doc:
            out.append(_quote(doc, "    "))
        impl = re.search(r"#\[pymethods\]\s*impl\s+" + rust_name + r"\s*\{(.*?)\n\}\n", text, re.S)
        if not impl:
            raise StubError(f"class {py_name} has no #[pymethods] block")
        for doc_lines, attrs, name, args, ret in _fn_blocks(impl.group(1)):
            static = any(a.startswith("#[staticmethod]") for a in attrs)
            getter = any(a.startswith("#[getter]") for a in attrs)
            params = []
            for arg in _split(args):
                arg = arg.strip()
                if not arg or arg in ("&self", "&mut self", "self"):
                    continue
                pname, _, ptype = arg.partition(":")
                params.append(f"{pname.strip()}: {py_type(ptype, classes, py_name)}")
            rtype = py_type(ret, classes, py_name) if ret else "None"
            doc = _docs(doc_lines)
            body = _quote(doc, "        ").lstrip() if doc else "..."
            if getter:
                out += ["    @property", f"    def {name}(self) -> {rtype}:", f"        {body}"]
            elif static:
                out += ["    @staticmethod", f"    def {name}({', '.join(params)}) -> {rtype}:", f"        {body}"]
            else:
                out += [f"    def {name}({', '.join(['self'] + params)}) -> {rtype}:", f"        {body}"]
        out.append("")
    for doc_lines, attrs, name, args, ret in _fn_blocks(re.sub(r"#\[pymethods\].*?\n\}\n", "", text, flags=re.S)):
        if not any(a.startswith("#[pyfunction]") for a in attrs):
            continue
        sig = next((a for a in attrs if a.startswith("#[pyo3(signature")), "")
        defaults = dict(re.findall(r"(\w+)\s*=\s*([^,)]+)", sig))
        params = []
        for arg in _split(args):
            pname, _, ptype = arg.strip().partition(":")
            pname = pname.strip()
            suffix = f" = {defaults[pname]}" if pname in defaults else ""
            params.append(f"{pname}: {py_type(ptype, classes, '')}{suffix}")
        doc = _docs(doc_lines)
        out += [f"def {name}({', '.join(params)}) -> {py_type(ret, classes, '')}:", _quote(doc, "    ") if doc else "    ...", ""]
    return "\n".join(out).rstrip() + "\n"


def write(root: Path, out_dir: Path) -> Path:
    stub = out_dir / "holographic_memory" / "__init__.pyi"
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text(generate(root / "src" / "python.rs"), encoding="utf-8")
    return stub


if __name__ == "__main__":
    print(generate(Path(sys.argv[1])))
