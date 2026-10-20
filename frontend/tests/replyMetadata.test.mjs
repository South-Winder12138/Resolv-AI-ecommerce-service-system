import assert from 'node:assert/strict'
import test from 'node:test'
import {metadata, refund} from './refundFixtures.mjs'
import {applyRunEvent,mergeConversationDetail} from '../src/stores/conversationState.ts'
import {readFileSync} from 'node:fs'

async function module() {try{return await import('../src/lib/replyMetadata.ts')}catch(e){assert.fail('versioned reply parser missing: '+e.message)}}
test('reply versions and closed v3 references',async()=>{
  const {parseReplyMetadata:parse}=await module()
  for(const v of [1,2,3]) assert.equal(parse({...metadata,schema_version:v,branch:'business',refund:undefined}).schema_version,v)
  assert.deepEqual(parse(metadata).refund,refund)
  assert.throws(()=>parse({...metadata,schema_version:4}))
  const broken=structuredClone(metadata);broken.refund.evidence.chunks=[]
  assert.throws(()=>parse(broken))
  assert.throws(()=>parse({...metadata,refund:{...refund,policy_citations:[{citation_number:1,window_id:'w',quote:'假的政策'}]}}))
  assert.throws(()=>parse({...metadata,branch:'business'}))
  assert.throws(()=>parse({...metadata,refund:{...refund,offer_id:'o',verdict:'undetermined'}}))
  assert.throws(()=>parse({...metadata,refund:{...refund,application:{money:{minor_units_decimal:9007199254740993}}}}))
})
test('SSE and detail preserve the same frozen refund without blocking chat',async()=>{
  const {canAsk}=await import('../src/stores/conversationState.ts')
  const run={run_id:'r',answer_message_id:'a',status:'running',knowledge_snapshot:[]}
  const state={id:'c',streamGeneration:1,currentRun:run,messages:[{id:'a',role:'assistant',content:'',createdAt:1,runId:'r'}]}
  const next=applyRunEvent(state,{type:'message',run_id:'r',text:'符合条件',...metadata},1)
  const detail={conversation:{conversation_id:'c',title:'t',created_at:offerDate,updated_at:offerDate},current_run:null,messages:[{message_id:'a',run_id:'r',seq:2,role:'assistant',text:'符合条件',created_at:offerDate}],runs:[{...run,status:'completed',result_json:metadata}],actions:[]}
  assert.deepEqual(next.messages[0].refund,refund)
  assert.deepEqual(mergeConversationDetail(state,detail).messages[0].refund,refund)
  assert.equal(canAsk(null),true)
  assert.equal(applyRunEvent(next,{type:'message',run_id:'r',...metadata,refund:null},0),next)
  assert.equal(applyRunEvent(next,{type:'run_started',run_id:'r',question_message_id:'q',answer_message_id:'a',replace_partial:true},1).messages[0].refund,undefined)
})
const offerDate='2026-10-16T04:00:00Z'

test('actual Python public projection retains complete citations and rejects unknown DTO fields',async()=>{
  const {parseReplyMetadata:parse,refundKnowledge}=await module()
  const actual=JSON.parse(readFileSync(new URL('./fixtures/refundReply.json',import.meta.url),'utf8'))
  const parsed=parse(actual)
  assert.deepEqual(parsed,actual)
  assert.ok(refundKnowledge(parsed.refund).citations.length)
  assert.throws(()=>parse({...actual,refund:{...actual.refund,subject_ref:'injected'}}))
})

test('unknown version cannot leave a clarification action after rejecting metadata',()=>{
  const run={run_id:'r',answer_message_id:'a',status:'running',knowledge_snapshot:[]}
  const state={id:'c',streamGeneration:1,currentRun:run,messages:[{id:'a',role:'assistant',content:'',createdAt:1,runId:'r'}]}
  const unknown={...metadata,schema_version:99,clarifies_understanding_id:'unsafe'}
  const next=applyRunEvent(state,{type:'message',run_id:'r',text:'unknown',...unknown},1)
  assert.equal(next.clarifiesUnderstandingId,undefined)
  const detail={conversation:{conversation_id:'c',title:'t',created_at:offerDate,updated_at:offerDate},current_run:null,messages:[{message_id:'a',run_id:'r',seq:2,role:'assistant',text:'unknown',created_at:offerDate}],runs:[{...run,status:'completed',result_json:unknown}],actions:[]}
  assert.equal(mergeConversationDetail(state,detail).clarifiesUnderstandingId,undefined)
})
