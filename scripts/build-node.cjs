// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

const { spawnSync } = require('node:child_process')
const { dirname, join } = require('node:path')

const manifest = require.resolve('@napi-rs/cli/package.json')
const executable = join(dirname(manifest), require(manifest).bin.napi)
const args = ['build', ...process.argv.slice(2)]
if (!args.includes('--')) args.push('--')
args.push('--lib')
const result = spawnSync(process.execPath, [executable, ...args], { stdio: 'inherit' })
if (result.error) throw result.error
process.exit(result.status ?? 1)
