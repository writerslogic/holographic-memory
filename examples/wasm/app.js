// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { openStore } from "./pkg/web.js";
import { IndexedDbStorage } from "./pkg/indexeddb.js";
import { OpfsStorage } from "./pkg/opfs.js";

const $ = (id) => document.getElementById(id);
const SAMPLE = [
  ["Rust ownership", "Rust ownership\nOwnership rules let the compiler prevent data races without a garbage collector. Borrowing keeps memory use predictable."],
  ["Holographic reduced representations", "Holographic reduced representations\nVectors are bound with circular convolution and superposed into a single trace that can be queried by unbinding."],
  ["BM25 ranking", "BM25 ranking\nOkapi BM25 scores passages by term frequency saturation and length normalization over an inverse document frequency weight."],
  ["Spring gardening", "Spring gardening\nPrune roses and plant tomatoes after the last frost has passed in your region."],
  ["Offline first", "Offline first\nLocal-first software keeps working without a network because the data and the search index live on the device."],
];

let store = null;
let storage = null;

const status = (message) => { $("status").textContent = message; };
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function setEnabled(on) {
  for (const id of ["add", "sample", "files", "query", "search", "download", "restore", "clear"]) $(id).disabled = !on;
}

async function refreshStats() {
  const stats = await store.stats();
  status(`Open. ${stats.documents} documents, ${stats.chunks} passages. ${store.simd ? "SIMD" : "Scalar"} WebAssembly.${storage ? " Saved on this device." : " Memory only."}`);
}

function makeStorage(kind) {
  if (kind === "opfs") return new OpfsStorage({ directory: "warm-demo" });
  if (kind === "indexeddb") return new IndexedDbStorage({ database: "warm-demo" });
  return undefined;
}

$("open").addEventListener("click", async () => {
  try {
    if (store) await store.close();
    setEnabled(false);
    storage = makeStorage($("backend").value);
    const passphrase = $("passphrase").value;
    store = await openStore({
      dimensions: 4096,
      storage,
      key: "demo",
      autosave: storage ? { debounceMs: 150 } : false,
      encryption: passphrase ? { passphrase, iterations: 200_000 } : undefined,
    });
    setEnabled(true);
    await refreshStats();
  } catch (error) {
    store = null;
    status(`Could not open the store: ${error.message}`);
  }
});

async function add(id, text) {
  const titleWeight = Number($("titleWeight").value) || 1;
  await store.memorizeDocument({ id, text, titleWeight, sourceUri: `local://${encodeURIComponent(id)}` });
}

$("add").addEventListener("click", async () => {
  const id = $("docId").value.trim();
  const text = $("docText").value;
  if (!id || !text.trim()) return status("Give the document an id and some text.");
  try {
    await add(id, text);
    $("docId").value = "";
    $("docText").value = "";
    await refreshStats();
  } catch (error) {
    status(error.message);
  }
});

$("sample").addEventListener("click", async () => {
  await store.memorizeDocuments(SAMPLE.map(([id, text]) => ({ id, text, titleWeight: 3 })));
  await refreshStats();
});

$("files").addEventListener("change", async (event) => {
  const inputs = [];
  for (const file of event.target.files) inputs.push({ id: file.name, text: await file.text(), sourceUri: `file://${file.name}` });
  if (inputs.length) await store.memorizeDocuments(inputs);
  event.target.value = "";
  await refreshStats();
});

function highlight(text, query) {
  const terms = query.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
  let html = escapeHtml(text);
  for (const term of terms) html = html.replace(new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "giu"), "<mark>$1</mark>");
  return html;
}

async function runSearch() {
  const query = $("query").value;
  const started = performance.now();
  try {
    const hits = await store.searchDocuments(query, {
      k: Number($("k").value) || 5,
      bm25K1: Number($("k1").value),
      bm25B: Number($("b").value),
    });
    const ms = performance.now() - started;
    $("count").textContent = `${hits.length} result${hits.length === 1 ? "" : "s"} in ${ms.toFixed(1)} ms`;
    $("results").innerHTML = hits
      .map((h) => `<li data-testid="hit"><span class="pill">${h.score.toFixed(3)}</span><strong>${escapeHtml(h.documentId)}</strong><div>${highlight((h.text ?? "").slice(0, 280), query)}</div><div class="meta">${escapeHtml(h.sourceUri ?? "")}</div></li>`)
      .join("");
  } catch (error) {
    $("count").textContent = error.message;
    $("results").innerHTML = "";
  }
}

$("search").addEventListener("click", runSearch);
$("query").addEventListener("keydown", (event) => { if (event.key === "Enter") runSearch(); });

$("download").addEventListener("click", async () => {
  const bytes = await store.exportSnapshot();
  const link = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" })),
    download: "writerslogic-warm.warm",
  });
  link.click();
  URL.revokeObjectURL(link.href);
});

$("restore").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    const passphrase = $("passphrase").value;
    const bytes = new Uint8Array(await file.arrayBuffer());
    const restored = await openStore({ snapshot: bytes, storage, key: "demo", autosave: storage ? { debounceMs: 150 } : false, encryption: passphrase ? { passphrase, iterations: 200_000 } : undefined });
    await store.close();
    store = restored;
    if (storage) await store.save();
    await refreshStats();
  } catch (error) {
    status(`Could not restore: ${error.message}`);
  }
  event.target.value = "";
});

$("clear").addEventListener("click", async () => {
  for (const id of await store.documentIds()) await store.deleteDocument(id);
  $("results").innerHTML = "";
  await refreshStats();
});

if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
