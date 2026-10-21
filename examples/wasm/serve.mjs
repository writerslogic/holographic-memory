// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Static server for the demo: node serve.mjs [port]
// Serves this directory, and the built package (bindings/wasm/npm/dist) at /pkg/.
// localhost is a secure context, which OPFS, IndexedDB and service workers require.

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const roots = [
  ["/pkg/", resolve(here, "../../bindings/wasm/npm/dist")],
  ["/", here],
];
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
};

export function startServer(port = 0, extraRoots = []) {
  const server = createServer(async (req, res) => {
    try {
      const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
      for (const [prefix, root] of [...extraRoots, ...roots]) {
        if (!path.startsWith(prefix)) continue;
        let file = normalize(join(root, path.slice(prefix.length)));
        if (file !== root && !file.startsWith(root + sep)) break;
        if ((await stat(file).catch(() => null))?.isDirectory()) file = join(file, "index.html");
        const body = await readFile(file).catch(() => null);
        if (!body) continue;
        res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "cache-control": "no-cache" });
        return res.end(body);
      }
      res.writeHead(404).end("not found");
    } catch {
      res.writeHead(400).end("bad request");
    }
  });
  return new Promise((resolveStart) => server.listen(port, "127.0.0.1", () => resolveStart(server)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const server = await startServer(Number(process.argv[2] ?? 8080));
  console.log(`http://localhost:${server.address().port}/`);
}
