// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { WarmMemorySystem } from 'writerslogic-warm'
import { OrganizationalKnowledge, readCuratedFiles } from '../integrations/organizational-knowledge/index.cjs'

export async function importCuratedFiles(memory: WarmMemorySystem, paths: string[]) {
  const knowledge = new OrganizationalKnowledge(memory, { namespace: 'toip' })
  return knowledge.importDocuments(await readCuratedFiles(paths))
}
