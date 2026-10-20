// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import * as api from "../../dist/deno.js";
import { FileStorage } from "../../dist/file.js";
import { defineSuite } from "../shared/suite.mjs";

const made = [];

defineSuite({
  test: (name, fn) => Deno.test({ name, sanitizeOps: false, sanitizeResources: false, fn }),
  api,
  FileStorage,
  scratch: async () => {
    const dir = await Deno.makeTempDir({ prefix: "hms-wasm-" });
    made.push(dir);
    return dir;
  },
});

addEventListener("unload", () => {
  for (const dir of made) Deno.removeSync(dir, { recursive: true });
});
