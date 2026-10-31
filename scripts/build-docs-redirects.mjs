import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";

export const CONTEXT_SHA256 = "ec2dbc721f02dc1f06a7d03f589c5a7e43b5fa0ff8184d6a384516b774790b45";
const DESTINATION = "https://writerslogic.com/warm/docs";
const LEGACY_VERSIONS = ["v0.6.4", "v0.6.3", "v0.6.1"];

export function destination(pathname) {
  const legacy = LEGACY_VERSIONS.find((version) => pathname === `/${version}` || pathname.startsWith(`/${version}/`));
  return DESTINATION + (legacy ? `/latest${pathname.slice(legacy.length + 1) || "/"}` : pathname);
}

const escapeAttribute = (value) => value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;");
const scriptJson = (value) => JSON.stringify(value).replaceAll("<", "\\u003c");

export function redirectHtml(pathname, fallback = false) {
  const target = fallback ? `${DESTINATION}/latest/` : destination(pathname);
  const script = `(()=>{let p=location.pathname;const v=${scriptJson(LEGACY_VERSIONS)}.find(v=>p==="/"+v||p.startsWith("/"+v+"/"));if(v)p="/latest"+(p.slice(v.length+1)||"/");location.replace(${scriptJson(DESTINATION)}+p+location.search+location.hash)})()`;
  return `<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,follow"><title>Documentation moved — WARM</title><link rel="canonical" href="${escapeAttribute(target)}">${fallback ? "" : `<meta http-equiv="refresh" content="0;url=${escapeAttribute(target)}">`}<script>${script}</script></head><body><main><h1>Documentation moved to WARM</h1><p>WARM is WritersLogic Adaptive Retention &amp; Memory.</p><p><a href="${escapeAttribute(target)}">Continue to the WARM documentation</a>.</p><p>Retired version links open current documentation under <code>latest</code>; historical source remains available in the repository's version tags.</p></main></body></html>\n`;
}

export function buildDocsRedirects(sourceDirectory, outputDirectory) {
  const source = resolve(sourceDirectory);
  const output = resolve(outputDirectory);
  if (source === output || output.startsWith(`${source}${sep}`) || source.startsWith(`${output}${sep}`)) throw new Error("Source and output must be separate directories");
  if (!existsSync(join(source, "latest/index.html"))) throw new Error("Expected a complete versioned documentation build");
  const context = readFileSync(join(source, "ns/hms/v1.jsonld"));
  if (createHash("sha256").update(context).digest("hex") !== CONTEXT_SHA256) throw new Error("Immutable legacy context differs from its published bytes");
  const pages = [];
  function visit(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) visit(path);
      else if (entry.name.endsWith(".html")) pages.push(relative(source, path).split(sep).join("/"));
    }
  }
  visit(source);
  rmSync(output, { recursive: true, force: true });
  const routes = new Map();
  for (const page of pages) {
    routes.set(page, page === "index.html" ? "/" : `/${page.replace(/\/index\.html$/, "/")}`);
    if (page.startsWith("latest/")) {
      for (const version of LEGACY_VERSIONS) {
        const oldPage = `${version}/${page.slice("latest/".length)}`;
        routes.set(oldPage, `/${oldPage.replace(/\/index\.html$/, "/")}`);
      }
    }
  }
  const write = (name, contents) => {
    const file = join(output, name);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, contents);
  };
  for (const [page, pathname] of routes) write(page, redirectHtml(pathname));
  write("404.html", redirectHtml("/", true));
  write("ns/hms/v1.jsonld", context);
  write("CNAME", "docs.holographic-memory.com\n");
  write(".nojekyll", "");
  write("robots.txt", `User-agent: *\nAllow: /\nSitemap: ${DESTINATION}/latest/sitemap.xml\n`);
  console.log(`Built ${routes.size} redirect pages in ${output}; legacy context SHA-256 ${CONTEXT_SHA256}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 4) throw new Error("Usage: node scripts/build-docs-redirects.mjs <full_site> <redirect_output>");
  buildDocsRedirects(process.argv[2], process.argv[3]);
}
