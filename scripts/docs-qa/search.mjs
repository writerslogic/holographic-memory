// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Types the queries in queries.txt into the site's search box in a real browser and fails when a
// query returns nothing or the expected page is not in the top five. Usage: node search.mjs <site-root>
import { readFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { serve } from './serve.mjs'

const [root] = process.argv.slice(2)
const port = Number(process.env.PORT ?? 8123)
const prefix = process.env.PREFIX ?? ''
const version = process.env.VERSION ?? 'dev'
const lines = (await readFile(new URL('./queries.txt', import.meta.url), 'utf8')).split('\n').filter(l => l.trim() && !l.startsWith('#'))

const { close } = await serve(root, port, prefix)
const browser = await chromium.launch()
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage()
await page.goto(`http://127.0.0.1:${port}${prefix}/${version}/`, { waitUntil: 'networkidle' })
let bad = 0
try {
  for (const line of lines) {
    const [query, expect] = line.split('|').map(s => s.trim())
    await page.keyboard.press('Escape')
    await page.click('input.md-search__input')
    await page.locator('input.md-search__input').fill('')
    await page.locator('input.md-search__input').pressSequentially(query, { delay: 15 })
    await page.waitForFunction(() => {
      const meta = document.querySelector('.md-search-result__meta')
      return meta && !/Type to start searching/i.test(meta.textContent)
    }, null, { timeout: 5000 }).catch(() => {})
    await page.waitForTimeout(400)
    const hits = await page.$$eval('.md-search-result__item > a, .md-search-result__item a.md-search-result__link', els => els.map(a => a.getAttribute('href')))
    const pages = [...new Set(hits.map(h => h.replace(/[?#].*$/, '')))].slice(0, 5)
    const ok = pages.some(h => expect.split(',').some(e => h.includes(e.trim())))
    if (!hits.length) {
      bad += 1
      console.log(`ZERO  "${query}"`)
    } else if (!ok) {
      bad += 1
      console.log(`MISS  "${query}" wanted ${expect}, got ${pages.map(h => h.replace(/^.*?\/(?:dev|latest)\//, '')).join(', ')}`)
    }
  }
} finally {
  await browser.close()
  await close()
}
console.log(`search: ${lines.length} queries, ${bad} failed`)
process.exit(bad ? 1 : 0)
