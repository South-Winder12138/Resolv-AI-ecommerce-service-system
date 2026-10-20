import type { KnowledgePublicResult, RefundEvidence, RefundOfferView, RefundPublicMoney, RefundPublicResult, RefundReceipt, RefundSubmissionView, ReplyMetadata } from '../types/agent.ts'
import { knowledgeReferences } from './knowledgeReferences.ts'

type ObjectValue = Record<string, unknown>
function object(v: unknown): ObjectValue { if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('invalid_reply_metadata'); return v as ObjectValue }
function check(ok: unknown): asserts ok { if (!ok) throw new Error('invalid_reply_metadata') }
function text(v: unknown): v is string { return typeof v === 'string' && v.length > 0 }
function nullableText(v: unknown) { return v === null || v === undefined || text(v) }
function strings(v: unknown) { return Array.isArray(v) && v.every(x => typeof x === 'string') }
function date(v: unknown) { return text(v) && /T.*(?:Z|[+-]\d\d:\d\d)$/.test(v) && Number.isFinite(Date.parse(v)) }
function keys(v: ObjectValue, allowed: string[]) { check(Object.keys(v).every(k => allowed.includes(k))) }
export function parseRefundMoney(value: unknown): RefundPublicMoney {
  const v = object(value)
  keys(v, ['currency', 'minor_units_decimal', 'unit_precision'])
  check(typeof v.currency === 'string' && /^[A-Z]{3}$/.test(v.currency) && typeof v.minor_units_decimal === 'string' && /^(0|[1-9]\d*)$/.test(v.minor_units_decimal) && Number.isInteger(v.unit_precision) && Number(v.unit_precision) >= 0 && Number(v.unit_precision) <= 9)
  check(BigInt(v.minor_units_decimal) <= 9223372036854775807n)
  return v as RefundPublicMoney
}
function receipt(value: unknown): RefundReceipt {
  const v = object(value)
  keys(v, ['application_id', 'order_source', 'order_id', 'status', 'created', 'money', 'reason', 'created_at'])
  check(text(v.application_id) && text(v.order_source) && text(v.order_id) && ['pending','approved','rejected','completed'].includes(String(v.status)) && typeof v.created === 'boolean' && typeof v.reason === 'string' && date(v.created_at))
  if (v.money !== null) parseRefundMoney(v.money)
  return v as RefundReceipt
}
export function parseRefundSubmission(value: unknown): RefundSubmissionView {
  const v = object(value)
  keys(v, ['submission_id','status','receipt','error_code','retryable'])
  check(text(v.submission_id) && ['created','running','interrupted','completed','failed'].includes(String(v.status)) && nullableText(v.error_code) && typeof v.retryable === 'boolean')
  if (v.receipt !== null) receipt(v.receipt)
  return v as RefundSubmissionView
}
export function parseRefundOffer(value: unknown): RefundOfferView {
  const v = object(value)
  keys(v, ['offer_id','order_id','money','reason','created_at','expires_at','validity','submission'])
  check(text(v.offer_id) && text(v.order_id) && text(v.reason) && date(v.created_at) && date(v.expires_at) && ['available','expired','needs_reassessment','submitted','unavailable'].includes(String(v.validity)))
  parseRefundMoney(v.money)
  if (v.submission !== null) parseRefundSubmission(v.submission)
  return v as RefundOfferView
}
function knowledge(value: unknown): KnowledgePublicResult {
  const k = object(value)
  check(text(k.release_id) && text(k.index_version) && text(k.evidence_hash) && ['sufficient','insufficient'].includes(String(k.decision)) && Array.isArray(k.citations) && Array.isArray(k.chunks) && Array.isArray(k.used_citation_numbers))
  const result = k as KnowledgePublicResult
  if (result.decision === 'insufficient') check(!result.citations.length && !result.chunks.length && !result.used_citation_numbers.length)
  else check(result.citations.length > 0 && knowledgeReferences({schema_version:3,knowledge:result}).length === result.citations.length)
  return result
}
export function refundKnowledge(refund: RefundPublicResult): KnowledgePublicResult | undefined {
  if (!refund.evidence?.windows.length || !refund.release_id || !refund.index_version) return undefined
  return {release_id:refund.release_id,index_version:refund.index_version,evidence_hash:refund.evidence.evidence_hash,decision:'sufficient',citations:refund.evidence.citations,chunks:refund.evidence.chunks,used_citation_numbers:[...new Set(refund.policy_citations.map(c=>c.citation_number))]}
}
function refund(value: unknown): RefundPublicResult {
  const v = object(value)
  keys(v, ['operation','verdict','order_id','reason','missing_information','unverified_conditions','next_action','release_id','index_version','evidence','policy_citations','offer_id','application','status_scope'])
  check([null,'eligibility','apply','status'].includes(v.operation as string | null) && [null,'eligible','ineligible','undetermined'].includes(v.verdict as string | null) && nullableText(v.order_id) && text(v.reason) && strings(v.missing_information) && strings(v.unverified_conditions) && ['show_offer','show_status','request_information','human_draft','none'].includes(String(v.next_action)) && v.status_scope === 'local_only' && nullableText(v.offer_id) && nullableText(v.release_id) && nullableText(v.index_version) && Array.isArray(v.policy_citations))
  check((v.release_id === null) === (v.index_version === null))
  if (v.application !== null) receipt(v.application)
  if (v.evidence !== null) {
    const e = object(v.evidence)
    check(Array.isArray(e.windows) && Array.isArray(e.citations) && Array.isArray(e.chunks) && typeof e.serialized === 'string' && Number.isInteger(e.input_tokens) && Number(e.input_tokens) >= 0 && text(e.evidence_hash))
    const evidence = e as RefundEvidence
    check(evidence.windows.length === evidence.citations.length)
    for (const [i,w] of evidence.windows.entries()) {
      const c = object(evidence.citations[i])
      check(Object.entries(w).every(([k,val])=>JSON.stringify(c[k]) === JSON.stringify(val)))
    }
    if (evidence.windows.length) knowledge(refundKnowledge(v as RefundPublicResult))
    else check(!evidence.chunks.length && !v.policy_citations.length)
    for (const value of v.policy_citations) {
      const p = object(value), c = evidence.citations.find(c=>c.citation_number === p.citation_number)
      check(c && c.window_id === p.window_id && text(p.quote))
      const bodies = new Map(evidence.chunks.map(b=>[b.chunk.chunk_id,b.chunk.text]))
      check(c.chunk_ids.map(id=>bodies.get(id)).join('\n').includes(p.quote))
    }
  } else check(!v.policy_citations.length && v.release_id === null)
  if (v.verdict === 'eligible' || v.verdict === 'ineligible') check(v.evidence && v.policy_citations.length)
  if (v.operation === 'status') check(v.verdict === null)
  if (v.offer_id !== null) check(v.verdict === 'eligible' && v.next_action === 'show_offer' && v.application === null)
  if (v.next_action === 'show_offer') check(text(v.offer_id))
  return v as RefundPublicResult
}

export function parseReplyMetadata(value: unknown): ReplyMetadata {
  const v = object(value)
  check([1,2,3].includes(Number(v.schema_version)) && typeof v.schema_version === 'number')
  check(['answer','clarification','unavailable','step_limit','knowledge_fallback',null].includes(v.reply_kind as string | null) && ['business','knowledge','refund_flow','escalate','fallback_script',null].includes(v.branch as string | null) && nullableText(v.understanding_id) && nullableText(v.primary_task_id) && nullableText(v.clarifies_understanding_id) && Array.isArray(v.suggested_actions))
  for (const action of v.suggested_actions) { const a = object(action), d = object(a.draft); check(text(a.suggestion_id) && ['transfer_human','create_ticket'].includes(String(a.type)) && typeof a.label === 'string' && a.preview_only === true && typeof d.title === 'string' && typeof d.content === 'string') }
  if (v.schema_version !== 1 && v.knowledge != null) knowledge(v.knowledge)
  if (v.refund != null) { check(v.schema_version === 3 && v.branch === 'refund_flow' && v.knowledge == null); refund(v.refund) }
  return v as ReplyMetadata
}
