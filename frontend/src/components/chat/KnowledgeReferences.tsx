import type { KnowledgePublicResult } from '../../types/agent'
import { knowledgeReferences, resolveWindowBodies } from '../../lib/knowledgeReferences'

export function KnowledgeReferences({ knowledge }: { knowledge?: KnowledgePublicResult }) {
  const metadata = { schema_version: 2 as const, knowledge }
  const references = knowledgeReferences(metadata)
  if (!references.length) return null
  return <section className="knowledge-sources" aria-label="知识参考资料">
    <p>本轮参考资料（{references.length} 个证据窗口）</p>
    {references.map(reference => <details key={reference.windowId}>
      <summary>[{reference.citationNumber}] {reference.title || reference.source} · {reference.sectionPath} · {reference.used ? '回答已引用' : '本轮参考'}</summary>
      <small>{reference.source}</small>
      {resolveWindowBodies(metadata, reference.windowId).map(body => <article key={body.chunk.chunk_id}>
        <p style={{ whiteSpace: 'pre-wrap' }}>{body.chunk.text}</p>
      </article>)}
    </details>)}
  </section>
}
