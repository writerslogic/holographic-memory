// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Runs axe-core on pages of a built docs site at three widths in light and dark, and fails on any
// serious or critical violation. Usage: node axe.mjs <site-root> [path ...]
import { chromium } from 'playwright'
import { AxeBuilder } from '@axe-core/playwright'
import { serve } from './serve.mjs'

const [root, ...extra] = process.argv.slice(2)
const port = Number(process.env.PORT ?? 8123)
const version = process.env.VERSION ?? 'dev'
const paths = extra.length ? extra : (process.env.PAGES ?? '/,/quickstart/,/concepts/holographic/,/reference/faq/,/reference/glossary/,/community/').split(',')
const widths = { desktop: 1440, tablet: 820, phone: 390 }

const { close } = await serve(root, port)
const browser = await chromium.launch()
let failures = 0
let checked = 0
try {
  for (const scheme of ['light', 'dark']) {
    for (const [name, width] of Object.entries(widths)) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme: scheme, reducedMotion: 'reduce' })
      const page = await context.newPage()
      for (const path of paths) {
        await page.goto(`http://127.0.0.1:${port}/${version}${path}`, { waitUntil: 'networkidle' })
        const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice']).analyze()
        checked += 1
        const bad = result.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')
        for (const v of bad) {
          failures += 1
          console.log(`FAIL ${path} [${name}/${scheme}] ${v.id} (${v.impact}): ${v.help}`)
          for (const node of v.nodes.slice(0, 3)) console.log(`    ${node.target.join(' ')}  ${node.failureSummary?.split('\n')[1] ?? ''}`)
        }
        const minor = result.violations.filter(v => !bad.includes(v)).map(v => `${v.id}:${v.impact}`)
        if (minor.length) console.log(`note ${path} [${name}/${scheme}] ${minor.join(', ')}`)
      }
      await context.close()
    }
  }
} finally {
  await browser.close()
  await close()
}
console.log(`axe: ${checked} page views checked, ${failures} serious or critical violations`)
process.exit(failures ? 1 : 0)
