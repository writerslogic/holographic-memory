# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hook: data-driven landing page, social cards, canonical URLs and version banner.

Everything the landing page states about runtimes, features and measured results is read at
build time from docs/_data/site.yml and the committed files under benchmarks/results/, so the
copy cannot drift from the repository. A violated contract fails the strict build.

Environment, set by scripts/build-docs-site.sh:
  WARM_DOCS_REF         the git ref the site is built from (main for dev, vX.Y.Z for a tag)
  WARM_CANONICAL_BASE   site URL of the `latest` build; every version canonicalises to it
  WARM_IS_LATEST        "1" while building the `latest` site (no outdated-version banner)
"""

import html
import json
import os
import posixpath
import re
import shutil
import sys
from pathlib import Path

import yaml
from urllib.parse import quote_from_bytes, urlsplit
from mkdocs.exceptions import PluginError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docs_charts  # noqa: E402
import docs_cards  # noqa: E402

REPO = "https://github.com/writerslogic/warm"
SNIPPET = re.compile(r"^(?P<path>[^:]+):(?P<name>[A-Za-z0-9_-]+)$")
MARK = re.compile(r"^\s*(?://|#)\s*--8<--\s*\[(?P<kind>start|end):(?P<name>[A-Za-z0-9_-]+)\]\s*$")

_state = {"pages": []}


def _fail(message):
    raise PluginError(f"docs/_data/site.yml: {message}")


def _repo_root(config):
    return Path(config["docs_dir"]).resolve().parent


def _dotted(data, path):
    node = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            _fail(f"metric path '{path}' is missing at '{part}'")
        node = node[part]
    return node


def read_snippet(root, spec):
    match = SNIPPET.match(spec)
    if not match:
        _fail(f"snippet '{spec}' must look like path:section")
    path = root / match["path"]
    if not path.is_file():
        _fail(f"snippet file '{match['path']}' does not exist")
    lines, inside = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        mark = MARK.match(line)
        if mark and mark["name"] == match["name"]:
            inside = mark["kind"] == "start"
            continue
        if inside and not MARK.match(line):
            lines.append(line)
    if not lines:
        _fail(f"snippet '{spec}' selects no lines")
    indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    return "\n".join(l[indent:] for l in lines)


def highlight(code, lang):
    from markdown import Markdown

    md = Markdown(
        extensions=["pymdownx.highlight", "pymdownx.superfences"],
        extension_configs={"pymdownx.highlight": {"anchor_linenums": False}},
    )
    return md.convert(f"```{lang}\n{code}\n```")


LANGS = {"rust": "rust", "python": "python", "node": "js", "go": "go", "java": "java",
         "dotnet": "csharp", "swift": "swift", "kotlin": "kotlin", "wasm": "js"}


def build_runtimes(root, site):
    out = []
    for rt in site.get("runtimes", []):
        status = rt.get("status")
        if status not in ("available", "preview"):
            _fail(f"runtime '{rt.get('id')}' has status '{status}', expected available or preview")
        entry = dict(rt)
        if status == "available":
            for need in rt.get("requires", []):
                if not (root / need).exists():
                    _fail(f"runtime '{rt['id']}' is available but '{need}' does not exist")
            if not rt.get("install"):
                _fail(f"runtime '{rt['id']}' is available but has no install command")
            lang = LANGS.get(rt["id"], "text")
            entry["install_html"] = highlight(rt["install"], "sh")
            entry["code_html"] = highlight(
                "\n\n".join(read_snippet(root, s) for s in rt.get("snippets", [])), lang
            ) if rt.get("snippets") else ""
        out.append(entry)
    return out


def build_headline(root, site):
    spec = site["headline"]
    path = root / spec["file"]
    if not path.is_file():
        _fail(f"headline file '{spec['file']}' does not exist")
    data = json.loads(path.read_text(encoding="utf-8"))
    value = _dotted(data, spec["metric"])
    rows = [{"label": f"WARM hybrid (BM25 + exact cosine)", "value": value, "warm": True}]
    for item in spec.get("compare", []):
        rows.append({"label": item["label"], "value": _dotted(data, item["metric"]), "warm": True})
    for item in spec.get("published", []):
        rows.append({"label": item["label"], "value": item["value"], "warm": False,
                     "source": item.get("source")})
    rows.sort(key=lambda r: -r["value"])
    return {
        "title": spec["title"], "value": value, "metric_label": spec["metric_label"],
        "context": spec.get("context", ""), "caveat": spec.get("caveat", ""),
        "file": spec["file"], "link": spec.get("link"), "rows": rows,
        "machine": data.get("environment", {}).get("hms", {}).get("cpu")
        or data.get("environment", {}).get("cpu"),
        "svg": docs_charts.headline_bars(rows, spec["metric_label"]),
    }


def doc_url(docs_dir, rel):
    """Map a docs-relative Markdown path to the page URL, failing if the page does not exist."""
    if not rel:
        return None
    if not (docs_dir / rel).is_file():
        _fail(f"link target '{rel}' is not a page in docs/")
    stem = rel[:-3] if rel.endswith(".md") else rel
    for index in ("/index", "/README"):
        if stem.endswith(index):
            stem = stem[: -len(index)]
    return stem + "/"


def inline_css(docs_dir, site_url):
    """The site's own stylesheets, minified, for one <style> block: two fewer blocking requests."""
    from csscompressor import compress

    css = "\n".join((docs_dir / "stylesheets" / name).read_text(encoding="utf-8") for name in ("warm.css", "home.css"))
    base = urlsplit(site_url or "/").path.rstrip("/")
    return compress(css.replace("../assets/", base + "/assets/"))


def crate_version(root):
    match = re.search(r'^version\s*=\s*"([^"]+)"', (root / "Cargo.toml").read_text(encoding="utf-8"), re.M)
    return match.group(1) if match else None


def build_comparison(root, site):
    supplied = root / "docs" / "_data" / "comparison.json"
    if supplied.is_file():
        table = json.loads(supplied.read_text(encoding="utf-8"))
        for key in ("columns", "rows"):
            if key not in table:
                _fail(f"comparison.json lacks '{key}'")
        table["generated"] = True
        return table
    return docs_charts.fallback_comparison(root, site)


def on_config(config):
    for key in ("logo", "favicon"):
        config["theme"][key] = config["site_url"].rstrip("/") + "/assets/crystal-detailed-static.svg"
    root = _repo_root(config)
    site = yaml.safe_load((root / "docs" / "_data" / "site.yml").read_text(encoding="utf-8"))
    ref = os.environ.get("WARM_DOCS_REF", "main")
    docs_dir = Path(config["docs_dir"]).resolve()
    extra = config["extra"]
    features = [dict(f, href=doc_url(docs_dir, f["link"])) for f in site.get("features", [])]
    headline = build_headline(root, site)
    headline["link"] = doc_url(docs_dir, headline["link"])
    chart = docs_charts.vector_chart(root, site["chart"])
    chart["link"] = doc_url(docs_dir, chart["link"])
    extra["warm"] = {
        "technical_line": site["technical_line"],
        "tagline": site["tagline"], "lede": " ".join(site["lede"].split()),
        "license": site.get("license", ""), "announcement": site.get("announcement") or {},
        "runtimes": build_runtimes(root, site), "features": features,
        "headline": headline, "chart": chart, "inline_css": inline_css(docs_dir, config["site_url"]), "crate_version": crate_version(root),
        "comparison": build_comparison(root, site),
        "hero_image": "data:image/svg+xml," + quote_from_bytes((docs_dir / "assets/crystal-detailed-static.svg").read_bytes()),
        "ref": ref, "is_dev": ref == "main", "is_latest": os.environ.get("WARM_IS_LATEST") == "1",
        "repo": REPO, "repo_ref": ref,
    }
    return config


FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _source_title(path):
    """The first level-one heading of a page, or None when front matter sets a title."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    front = FRONT_MATTER.match(text)
    if front:
        if re.search(r"^title\s*:", front.group(0), re.M):
            return None
        text = text[front.end():]
    text = HTML_COMMENT.sub("", text)
    for line in text.splitlines():
        if line.strip():
            return line[2:].strip() if line.startswith("# ") else None
    return None


def on_nav(nav, config, files):
    """Every published page must be reachable from the navigation; only the landing page is not."""
    for page in nav.pages:
        title = _source_title(page.file.abs_src_path)
        if title:
            # MkDocs reads a title from the source only when the first line is the heading, so a
            # page that opens with a licence comment would show its file name in every sidebar
            # that is rendered before the page itself.
            page._title_from_render = title
    listed = {page.file.src_uri for page in nav.pages}
    stray = sorted(
        f.src_uri for f in files.documentation_pages()
        if f.src_uri not in listed and f.src_uri not in ("index.md", "404.md")
    )
    if stray:
        raise PluginError(
            "pages missing from the navigation (list them in docs/.nav.yml or the .nav.yml of their "
            "folder): " + ", ".join(stray)
        )
    return nav


def _description(html_text):
    match = re.search(r"<p>(.*?)</p>", html_text, re.S)
    if not match:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
    text = " ".join(text.split())
    if len(text) > 200:
        text = text[:197].rsplit(" ", 1)[0] + "..."
    return text


def on_page_markdown(markdown, page, config, files):
    """Resolve `next_steps` targets, written as Markdown paths, to page URLs."""
    steps = page.meta.get("next_steps")
    if not steps:
        return markdown
    here = posixpath.dirname(page.file.src_uri)
    resolved = []
    for step in steps:
        target = step["url"]
        found = None
        for candidate in (posixpath.normpath(posixpath.join(here, target)), posixpath.normpath(target)):
            found = files.get_file_from_path(candidate)
            if found is not None:
                break
        if found is None:
            raise PluginError(f"{page.file.src_uri}: next_steps target '{target}' is not a page")
        resolved.append(dict(step, url=found.url))
    page.meta["next_steps"] = resolved
    return markdown


def on_page_content(content, page, config, files):
    if not page.meta.get("description"):
        description = _description(content)
        if description:
            page.meta["description"] = description
    return content


def on_page_context(context, page, config, nav):
    base = os.environ.get("WARM_CANONICAL_BASE")
    if base:
        page.canonical_url = base.rstrip("/") + "/" + page.url
    slug = re.sub(r"[^a-z0-9]+", "-", page.file.src_uri.lower().removesuffix(".md")).strip("-") or "index"
    section = page.ancestors[-1].title if page.ancestors else ""
    page.warm_og = f"assets/og/{slug}.png"
    _state["pages"].append((slug, page.title or config["site_name"], section,
                            page.meta.get("description", config["site_description"])))
    site_url = config["site_url"] or ""
    context["warm_og_url"] = site_url + page.warm_og
    return context


def on_post_build(config):
    rustdoc = os.environ.get("WARM_RUSTDOC_DIR")
    if rustdoc and (Path(rustdoc) / "writerslogic_warm").is_dir():
        shutil.copytree(rustdoc, Path(config["site_dir"]) / "api" / "rust" / "doc", dirs_exist_ok=True)
    out = Path(config["site_dir"]) / "assets" / "og"
    out.mkdir(parents=True, exist_ok=True)
    root = _repo_root(config)
    fonts = root / "docs" / "assets" / "fonts"
    seen = set()
    for slug, title, section, description in _state["pages"]:
        if slug in seen:
            continue
        seen.add(slug)
        docs_cards.render(out / f"{slug}.png", fonts, title, section, description)
    _state["pages"].clear()

# mkdocstrings resolves its search paths when the plugin starts, before the generated pages exist, so
# the stub package for the PyO3 module is written as soon as this hook is loaded.
import docs_pyo3_stub  # noqa: E402

_root = Path(__file__).resolve().parents[1]
if any((_root / relative).is_file() for relative in ("src/bindings/python/mod.rs", "src/python.rs")):
    docs_pyo3_stub.write(_root, _root / ".docs-gen" / "pyi")
