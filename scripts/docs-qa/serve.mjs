// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// A static file server for a built docs site, with the directory-index and 404 behaviour of
// GitHub Pages. Used by the screenshot, axe, Lighthouse and search checks.
import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { extname, join, normalize, resolve } from 'node:path'
import { gzipSync } from 'node:zlib'

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.xml': 'application/xml',
  '.txt': 'text/plain',
  '.gz': 'application/gzip',
}

export async function serve(root, port = 0, prefix = '') {
  const base = resolve(root)
  const server = createServer(async (req, res) => {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname)
    if (prefix && path.startsWith(prefix)) path = path.slice(prefix.length) || '/'
    const file = join(base, normalize(path))
    if (!file.startsWith(base)) {
      res.writeHead(403).end()
      return
    }
    let target = file
    try {
      if ((await stat(target)).isDirectory()) {
        if (!path.endsWith('/')) {
          res.writeHead(301, { location: req.url.split('?')[0] + '/' }).end()
          return
        }
        target = join(target, 'index.html')
      }
      const body = await readFile(target)
      const type = TYPES[extname(target)] ?? 'application/octet-stream'
      const headers = { 'content-type': type, 'cache-control': 'public, max-age=600' }
      // GitHub Pages compresses text responses; do the same so timings are comparable.
      const text = /^(text\/|application\/(json|xml)|image\/svg)/.test(type)
      if (text && /\bgzip\b/.test(req.headers['accept-encoding'] ?? '')) {
        headers['content-encoding'] = 'gzip'
        res.writeHead(200, headers)
        res.end(gzipSync(body))
      } else {
        res.writeHead(200, headers)
        res.end(body)
      }
    } catch {
      try {
        const body = await readFile(join(base, '404.html'))
        res.writeHead(404, { 'content-type': TYPES['.html'] }).end(body)
      } catch {
        res.writeHead(404).end('not found')
      }
    }
  })
  await new Promise(done => server.listen(port, '127.0.0.1', done))
  return { server, port: server.address().port, close: () => new Promise(done => server.close(done)) }
}
