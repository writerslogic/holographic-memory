import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { HmsStore } from '../../../integrations/js/packages/core/dist/index.js';
import { OrderedLog } from '../../../target/audit-ordered-log.mjs';

const path = mkdtempSync(join(tmpdir(), 'hms-audit-log-'));
const store = new HmsStore({ path, dimensions: 1024 });
try {
  const log = new OrderedLog(store, 'session');
  await log.append([{ text: 'first' }, { text: 'second' }, { text: 'third' }]);
  const original = store.forget.bind(store);
  store.forget = async ids => {
    await original(ids.slice(0, 1));
    throw new Error('injected interruption after first committed delete');
  };
  try { await log.clear(); } catch (error) { console.log(error.message); }
  store.forget = original;
  const fresh = new OrderedLog(store, 'session');
  console.log(JSON.stringify({ afterInterruptedClear: await fresh.size(), retained: (await store.list()).records.map(record => record.id) }));
  await fresh.clear();
  console.log(JSON.stringify({ afterRetriedClear: (await store.list()).records.map(record => record.id) }));
  await fresh.append([{ text: 'new first' }]);
  console.log(JSON.stringify({ afterAppend: (await fresh.tail()).map(entry => entry.text) }));
} finally {
  await store.close();
  rmSync(path, { recursive: true });
}
