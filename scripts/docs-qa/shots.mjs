// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Screenshots of the landing page and one documentation page at desktop, tablet and phone
// widths in light and dark. Usage: node shots.mjs <site-root> <out-dir> [path ...]
// The site must have been built with HMS_SITE_URL pointing at http://127.0.0.1:<port>/<version>/.
import { mkdir } from 'node:fs/promises'
import { chromium } from 'playwright'
import { serve } from './serve.mjs'

const [root, out, ...extra] = process.argv.slice(2)
const port = Number(process.env.PORT ?? 8123)
const version = process.env.VERSION ?? 'dev'
const paths = extra.length ? extra : ['/', '/quickstart/', '/concepts/holographic/']
const widths = { desktop: 1440, tablet: 820, phone: 390 }
const full = process.env.FULL !== '0'
const clip = process.env.CLIP ? process.env.CLIP.split(',').map(Number) : null
const only = process.env.ONLY ? process.env.ONLY.split(',') : null

await mkdir(out, { recursive: true })
const { close } = await serve(root, port)
const browser = await chromium.launch()
try {
  for (const scheme of ['light', 'dark']) {
    for (const [name, width] of Object.entries(widths)) {
      if (only && !only.includes(`${name}-${scheme}`)) continue
      const context = await browser.newContext({
        viewport: { width, height: name === 'phone' ? 844 : 900 },
        colorScheme: scheme,
        deviceScaleFactor: 1,
        reducedMotion: 'reduce',
      })
      const page = await context.newPage()
      for (const path of paths) {
        await page.goto(`http://127.0.0.1:${port}/${version}${path}`, { waitUntil: 'networkidle' })
        await page.evaluate(() => document.fonts.ready)
        const slug = path.replace(/\//g, '_').replace(/^_|_$/g, '') || 'home'
        await page.screenshot({
          path: `${out}/${slug}-${name}-${scheme}.png`,
          fullPage: full,
          ...(clip ? { clip: { x: 0, y: clip[0], width, height: clip[1] } } : {}),
        })
      }
      await context.close()
    }
  }
} finally {
  await browser.close()
  await close()
}
