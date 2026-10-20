import type { KnowledgeChunkBody, KnowledgePublicResult, ReplyMetadata } from '../types/agent.ts'

type ReferenceMetadata = Pick<ReplyMetadata, 'schema_version'> & { knowledge?: KnowledgePublicResult | null }
export type KnowledgeReferenceView = { windowId: string; citationNumber: number; source: string; title: string; sectionPath: string; chunkIds: readonly string[]; used: boolean }

// Fail closed for incomplete or inconsistent persisted reference projections.
export function knowledgeReferences(metadata?: ReferenceMetadata | null): KnowledgeReferenceView[] {
  const k = metadata?.schema_version === 2 || metadata?.schema_version === 3 ? metadata.knowledge : null
  if (!k || k.decision !== 'sufficient' || !Array.isArray(k.citations) || !Array.isArray(k.chunks) || !Array.isArray(k.used_citation_numbers)) return []
  if (k.chunks.some(b => !b?.chunk || typeof b.chunk.chunk_id !== 'string' || !b.chunk.chunk_id)) return []
  const bodies = new Map(k.chunks.map(b => [b?.chunk?.chunk_id, b]))
  if (bodies.size !== k.chunks.length) return []
  const windows = new Set<string>(), referenced = new Set<string>()
  const views: KnowledgeReferenceView[] = []
  for (const [index, c] of k.citations.entries()) {
    if (!c || c.citation_number !== index + 1 || !c.window_id || windows.has(c.window_id) || c.index_version !== k.index_version || !c.source || !Array.isArray(c.chunk_ids) || !c.chunk_ids.length || new Set(c.chunk_ids).size !== c.chunk_ids.length) return []
    windows.add(c.window_id)
    for (const id of c.chunk_ids) {
      const b = bodies.get(id)
      if (!b || b.chunk.document_id !== c.document_id || b.chunk.source !== c.source || b.section_instance_id !== c.section_instance_id || typeof b.chunk.text !== 'string' || !b.chunk.text) return []
      referenced.add(id)
    }
    views.push({ windowId: c.window_id, citationNumber: c.citation_number, source: c.source, title: c.title, sectionPath: c.section_path, chunkIds: c.chunk_ids, used: k.used_citation_numbers.includes(c.citation_number) })
  }
  if (referenced.size !== bodies.size || new Set(k.used_citation_numbers).size !== k.used_citation_numbers.length || k.used_citation_numbers.some(n => !Number.isInteger(n) || n < 1 || n > views.length)) return []
  return views
}

export function resolveWindowBodies(metadata: ReferenceMetadata, windowId: string): KnowledgeChunkBody[] {
  const view = knowledgeReferences(metadata).find(c => c.windowId === windowId)
  if (!view || !metadata.knowledge) return []
  const bodies = new Map(metadata.knowledge.chunks.map(b => [b.chunk.chunk_id, b]))
  return view.chunkIds.map(id => bodies.get(id)!)
}
