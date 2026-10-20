// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Lighthouse on the landing page and one documentation page, mobile and desktop, with a floor
// on every category. Usage: node lighthouse.mjs <site-root> [path ...]
// Environment: THRESHOLD (default 95), REPORT_DIR (writes the JSON reports there).
import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import lighthouse, { desktopConfig } from 'lighthouse'
import { serve } from './serve.mjs'

const [root, ...extra] = process.argv.slice(2)
const port = Number(process.env.PORT ?? 8123)
const debugPort = Number(process.env.DEBUG_PORT ?? 9333)
const version = process.env.VERSION ?? 'dev'
const threshold = Number(process.env.THRESHOLD ?? 95)
const paths = extra.length ? extra : ['/', '/concepts/holographic/']
const categories = ['performance', 'accessibility', 'best-practices', 'seo']

const { close } = await serve(root, port)
const browser = await chromium.launch({ args: [`--remote-debugging-port=${debugPort}`] })
let failed = false
const rows = []
try {
  for (const path of paths) {
    for (const [name, config] of [['mobile', undefined], ['desktop', desktopConfig]]) {
      const url = `http://127.0.0.1:${port}/${version}${path}`
      const result = await lighthouse(url, { port: debugPort, output: 'json', logLevel: 'error', onlyCategories: categories }, config)
      const scores = Object.fromEntries(categories.map(c => [c, Math.round(result.lhr.categories[c].score * 100)]))
      rows.push({ path, name, ...scores })
      if (Object.values(scores).some(s => s < threshold)) {
        failed = true
        for (const audit of Object.values(result.lhr.audits)) {
          if (audit.score !== null && audit.score < 0.9 && audit.scoreDisplayMode !== 'informative' && audit.scoreDisplayMode !== 'notApplicable') {
            console.log(`  ${path} [${name}] ${audit.id}: ${audit.title} (${audit.score})`)
          }
        }
      }
      if (process.env.REPORT_DIR) {
        await mkdir(process.env.REPORT_DIR, { recursive: true })
        const slug = path.replace(/\//g, '_').replace(/^_|_$/g, '') || 'home'
        await writeFile(`${process.env.REPORT_DIR}/lighthouse-${slug}-${name}.json`, result.report)
      }
    }
  }
} finally {
  await browser.close()
  await close()
}
console.log('page'.padEnd(28), 'form'.padEnd(8), categories.map(c => c.padEnd(15)).join(''))
for (const r of rows) console.log(r.path.padEnd(28), r.name.padEnd(8), categories.map(c => String(r[c]).padEnd(15)).join(''))
console.log(failed ? `FAIL: a category is below ${threshold}` : `ok: every category is at least ${threshold}`)
process.exit(failed ? 1 : 0)
