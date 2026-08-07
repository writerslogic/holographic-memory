import { HolographicMemorySystem } from 'holographic-memory';

const memory = new HolographicMemorySystem(16_384, './memory_db');
await memory.memorizeText('berlin', 'Berlin is the capital of Germany');

console.log(memory.explainQuery('capital of Germany', 5));
console.log(memory.indexStatus());
await memory.maintainIndices();
await memory.flush();
console.log(memory.storageHealth());
