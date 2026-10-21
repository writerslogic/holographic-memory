import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const binding = require('./index.js');

export const HolographicMemorySystem = binding.HolographicMemorySystem;
export const LocalEmbedder = binding.LocalEmbedder;
export const SemanticIndex = binding.SemanticIndex;
export default binding;
