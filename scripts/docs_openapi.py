# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Renders an OpenAPI 3 document as Markdown pages, so the HTTP reference is searchable text
generated from the same file the server publishes and cannot drift from it."""

import json
import re
from pathlib import Path

METHODS = ("get", "put", "post", "delete", "patch", "head", "options")


def _anchor(name: str) -> str:
    return "schema-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def type_of(schema, spec) -> str:
    """A short, linked type expression for a schema."""
    if not isinstance(schema, dict):
        return "any"
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return f"[`{name}`](#{_anchor(name)})"
    for key in ("oneOf", "anyOf"):
        if key in schema:
            return " or ".join(type_of(s, spec) for s in schema[key])
    if "allOf" in schema:
        return " and ".join(type_of(s, spec) for s in schema["allOf"])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = " or ".join(kind)
    if kind == "array":
        return f"array of {type_of(schema.get('items', {}), spec)}"
    if "enum" in schema:
        return "one of " + ", ".join(f"`{v}`" for v in schema["enum"])
    fmt = f" ({schema['format']})" if schema.get("format") else ""
    return f"`{kind or 'object'}`{fmt}"


def _properties(schema, spec, required=()):
    rows = []
    for name, prop in (schema.get("properties") or {}).items():
        prop = prop if isinstance(prop, dict) else {}  # `true` is a schema that accepts anything
        need = "required" if name in required else ""
        # A property may be a boolean schema (`true`), which carries no description.
        description = prop.get("description") if isinstance(prop, dict) else None
        rows.append(f"| `{name}` | {type_of(prop, spec)} | {need} | {_cell(description)} |")
    return rows


def _table(rows):
    return ["| Field | Type | | Description |", "| --- | --- | --- | --- |", *rows, ""] if rows else []


def _body(spec, content):
    lines = []
    for media, detail in (content or {}).items():
        schema = detail.get("schema", {})
        lines.append(f"Content type `{media}`: {type_of(schema, spec)}")
        lines.append("")
    return lines


def render(spec: dict) -> str:
    info = spec.get("info", {})
    out = [
        "---",
        "description: " + json.dumps(_cell(info.get("description") or f"HTTP reference for {info.get('title', 'the server')}.")[:200]),
        "---",
        "",
        f"# {info.get('title', 'HTTP API')} reference",
        "",
    ]
    if info.get("description"):
        out += [info["description"].strip(), ""]
    out += [f"Version `{info.get('version', 'unknown')}` of the OpenAPI document the server publishes.", ""]
    for server in spec.get("servers", []):
        out.append(f"- Server: `{server.get('url')}` {server.get('description', '')}".rstrip())
    if spec.get("servers"):
        out.append("")
    out += ["## Operations", ""]
    tags = {}
    for path, item in spec.get("paths", {}).items():
        for method in METHODS:
            op = item.get(method)
            if op:
                tags.setdefault((op.get("tags") or ["Operations"])[0], []).append((path, method, op, item))
    for tag, ops in tags.items():
        out += [f"### {tag}", ""]
        for path, method, op, item in ops:
            out += [f"#### `{method.upper()} {path}`", ""]
            if op.get("summary"):
                out += [f"**{op['summary'].strip()}**", ""]
            if op.get("description"):
                out += [op["description"].strip(), ""]
            params = (item.get("parameters") or []) + (op.get("parameters") or [])
            if params:
                out += ["| Parameter | In | Type | | Description |", "| --- | --- | --- | --- | --- |"]
                for p in params:
                    out.append(
                        f"| `{p.get('name')}` | {p.get('in')} | {type_of(p.get('schema', {}), spec)} | "
                        f"{'required' if p.get('required') else ''} | {_cell(p.get('description'))} |"
                    )
                out.append("")
            body = op.get("requestBody")
            if body:
                out += ["Request body" + (" (required)" if body.get("required") else ""), ""]
                out += _body(spec, body.get("content"))
            if op.get("security") or spec.get("security"):
                schemes = [name for req in (op.get("security") or spec.get("security")) for name in req]
                out += ["Authentication: " + ", ".join(f"`{s}`" for s in schemes), ""]
            out += ["| Status | Description | Body |", "| --- | --- | --- |"]
            for code, resp in (op.get("responses") or {}).items():
                media = (resp.get("content") or {}).values()
                kinds = "; ".join(type_of(m.get("schema", {}), spec) for m in media)
                out.append(f"| `{code}` | {_cell(resp.get('description'))} | {kinds} |")
            out.append("")
    schemas = (spec.get("components") or {}).get("schemas") or {}
    if schemas:
        out += ["## Schemas", ""]
        for name, schema in schemas.items():
            out += [f"### {name} {{#{_anchor(name)}}}", ""]
            if schema.get("description"):
                out += [schema["description"].strip(), ""]
            rows = _properties(schema, spec, schema.get("required") or ())
            out += _table(rows) if rows else [f"Type: {type_of(schema, spec)}", ""]
    return "\n".join(out).rstrip() + "\n"


def find_spec(root: Path):
    for candidate in ("docs/server/openapi.json", "schemas/openapi.json", "docs/openapi.json", "openapi.json"):
        path = root / candidate
        if path.is_file():
            return path
    return None


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
