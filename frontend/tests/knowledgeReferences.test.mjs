import assert from 'node:assert/strict'
import test from 'node:test'
import { knowledgeReferences, resolveWindowBodies } from '../src/lib/knowledgeReferences.ts'
import { applyRunEvent, mergeConversationDetail } from '../src/stores/conversationState.ts'

const body = id => ({ chunk: { chunk_id: id, document_id: 'doc', text: `正文${id}`, answer: `正文${id}`, source: 'policy.md', chunk_index: id.charCodeAt(0) - 97, section_path: '运费' }, section_instance_id: 'section' })
const citation = (n, ids) => ({ citation_number: n, window_id: `w${n}`, document_id: 'doc', section_instance_id: 'section', source: 'policy.md', title: '运费规则', section_path: '运费', chunk_ids: ids, index_version: 'v3', sigmoid_score: 0.8, rerank_score: 1.4 })
export const metadata = { schema_version: 2, branch: 'knowledge', reply_kind: 'answer', understanding_id: 'u', primary_task_id: 'p', suggested_actions: [], knowledge: { release_id: 'release', index_version: 'v3', evidence_hash: 'hash', decision: 'sufficient', citations: [citation(1, ['a', 'b']), citation(2, ['b', 'c'])], chunks: ['a','b','c'].map(body), used_citation_numbers: [1] } }

function frozen(value) { if (value && typeof value === 'object') { Object.freeze(value); Object.values(value).forEach(frozen) }; return value }

test('overlapping windows retain stable numbers and share immutable bodies', () => {
  const m = frozen(structuredClone(metadata)), original = JSON.stringify(m)
  const views = knowledgeReferences(m)
  assert.deepEqual(views.map(v => [v.windowId, v.citationNumber, v.used]), [['w1',1,true],['w2',2,false]])
  assert.deepEqual(resolveWindowBodies(m, 'w1').map(b => b.chunk.chunk_id), ['a','b'])
  assert.equal(resolveWindowBodies(m, 'w1')[1], resolveWindowBodies(m, 'w2')[0])
  assert.equal(JSON.stringify(m), original)
  assert.deepEqual(resolveWindowBodies(m, 'unknown'), [])
})

test('fallback and legacy metadata expose no new window sources', () => {
  assert.deepEqual(knowledgeReferences({...metadata, schema_version: 1}), [])
  assert.deepEqual(knowledgeReferences({...metadata, reply_kind: 'knowledge_fallback', knowledge: {...metadata.knowledge, decision: 'insufficient'}}), [])
})

test('v3 keeps the original knowledge window projection', () => {
  assert.deepEqual(knowledgeReferences({...metadata,schema_version:3,refund:null}),knowledgeReferences(metadata))
})

test('broken numbering, duplicate ids, missing or mismatched bodies never create sources', () => {
  const bad = []
  let m = structuredClone(metadata); m.knowledge.citations[0].citation_number = 7; bad.push(m)
  m = structuredClone(metadata); m.knowledge.chunks.pop(); bad.push(m)
  m = structuredClone(metadata); m.knowledge.chunks[0].chunk.document_id = 'alien'; bad.push(m)
  m = structuredClone(metadata); m.knowledge.chunks[0].section_instance_id = 'alien'; bad.push(m)
  m = structuredClone(metadata); m.knowledge.used_citation_numbers = [9]; bad.push(m)
  m = structuredClone(metadata); m.knowledge.citations[1].window_id = 'w1'; bad.push(m)
  m = structuredClone(metadata); m.knowledge.chunks.push(m.knowledge.chunks[0]); bad.push(m)
  m = structuredClone(metadata); m.knowledge.citations[0].index_version = 'alien'; bad.push(m)
  for (const value of bad) { assert.deepEqual(knowledgeReferences(value), []); assert.deepEqual(resolveWindowBodies(value, 'w1'), []) }
})

test('SSE and detail restore identical knowledge; old generations or runs cannot replace it', () => {
  const run = { run_id:'r', answer_message_id:'a', question_message_id:'q', status:'running', knowledge_snapshot:[] }
  const state = { id:'c', title:'test', createdAt:1, updatedAt:1, streamGeneration:2, currentRun:run, messages:[{id:'a',role:'assistant',content:'',createdAt:1,runId:'r'}] }
  const event = { type:'message', run_id:'r', text:'运费[1]', ...metadata }
  const afterEvent = applyRunEvent(state,event,2)
  const detail = { conversation:{conversation_id:'c',title:'test',created_at:'2026-10-15T00:00:00Z',updated_at:'2026-10-15T00:00:00Z'}, current_run:null, messages:[{message_id:'a',run_id:'r',seq:2,role:'assistant',text:'运费[1]',created_at:'2026-10-15T00:00:00Z'}], runs:[{...run,status:'completed',result_json:metadata}], actions:[] }
  assert.deepEqual(afterEvent.messages[0].knowledge, metadata.knowledge)
  assert.deepEqual(mergeConversationDetail(state,detail).messages[0].knowledge, afterEvent.messages[0].knowledge)
  assert.equal(applyRunEvent(afterEvent,{...event,knowledge:null},1), afterEvent)
  assert.equal(applyRunEvent(afterEvent,{...event,run_id:'other',knowledge:null},2), afterEvent)
  const reset = applyRunEvent(afterEvent,{type:'run_started',run_id:'r',question_message_id:'q',answer_message_id:'a',replace_partial:true},2)
  assert.equal(reset.messages[0].knowledge, undefined)
})
