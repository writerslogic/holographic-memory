// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import * as api from "../../dist/node.js";
import { FileStorage } from "../../dist/file.js";
import { defineSuite } from "../shared/suite.mjs";

const made = [];
after(async () => {
  await Promise.all(made.map((dir) => rm(dir, { recursive: true, force: true })));
});

defineSuite({
  test,
  api,
  FileStorage,
  scratch: async () => {
    const dir = await mkdtemp(join(tmpdir(), "warm-wasm-"));
    made.push(dir);
    return dir;
  },
});
