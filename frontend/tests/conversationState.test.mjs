import assert from 'node:assert/strict'
import test from 'node:test'
import { applyRunEvent, mergeConversationDetail, canAsk } from '../src/stores/conversationState.ts'

const run={run_id:'run',conversation_id:'c',request_id:'request',mode:'agent',status:'interrupted',attempt:1,question:'问题',question_message_id:'q',answer_message_id:'a',knowledge_snapshot:[]}
const base={id:'c',title:'测试',createdAt:1,updatedAt:1,streamGeneration:2,currentRun:run,messages:[{id:'q',role:'user',content:'问题',createdAt:1,runId:'run'},{id:'a',role:'assistant',content:'partial',createdAt:1,runId:'run'}]}

test('retry replaces partial answer without adding a user',()=>{
  const next=applyRunEvent(base,{type:'run_started',run_id:'run',attempt:2,question_message_id:'q',answer_message_id:'a',replace_partial:true},2)
  assert.equal(next.messages.filter(m=>m.role==='user').length,1)
  assert.equal(next.messages.find(m=>m.id==='a').content,'')
})

test('duplicate action event has one card',()=>{
  const event={type:'tool_call',run_id:'run',action_id:'action',tool_call_id:'tc',tool:'query_order',arguments:{order_id:'MH001'}}
  const once=applyRunEvent(base,event,2)
  const twice=applyRunEvent(once,event,2)
  assert.equal(twice.messages.find(m=>m.id==='a').toolCalls.length,1)
})

test('stale generation is ignored',()=>{
  assert.deepEqual(applyRunEvent(base,{type:'text_delta',run_id:'run',text:'late'},1),base)
})

test('server snapshot overrides cache and restores saved sources',()=>{
  const detail={conversation:{conversation_id:'c',title:'服务端',created_at:'2026-10-12T00:00:00Z',updated_at:'2026-10-12T00:00:00Z'},messages:[{message_id:'q',run_id:'run',seq:1,role:'user',text:'问题',created_at:'2026-10-12T00:00:00Z'},{message_id:'a',run_id:'run',seq:2,role:'assistant',text:'saved answer',created_at:'2026-10-12T00:00:00Z'}],current_run:null,actions:[],runs:[{...run,status:'completed',knowledge_snapshot:[{chunk_id:'chunk',source:'refund.md',section_path:'退款',chunk_index:0,answer:'条款',score:1}]}],next_before_seq:null}
  const next=mergeConversationDetail(base,detail)
  assert.equal(next.messages.find(m=>m.id==='a').content,'saved answer')
  assert.equal(next.messages.find(m=>m.id==='a').knowledgeSources[0].source,'refund.md')
  assert.equal(next.currentRun,null)
})

test('pending active interrupted and finalizing disable asking',()=>{
  for(const status of ['running','waiting_approval','interrupted','finalizing'])assert.equal(canAsk(status),false)
  for(const status of [null,'completed','failed'])assert.equal(canAsk(status),true)
})

test('original request id survives disconnect before run_started',()=>{
  const state={...base,currentRun:null,pendingRequest:{requestId:'original',question:'问题',questionMessageId:'optimistic-q',answerMessageId:'optimistic-a'}}
  const next=applyRunEvent(state,{type:'error',run_id:null,code:'connection_lost',retryable:true},2)
  assert.equal(next.pendingRequest.requestId,'original')
})

test('optimistic question binds server id once',()=>{
  const state={...base,currentRun:null,messages:[{id:'optimistic-q',role:'user',content:'问题',createdAt:1},{id:'optimistic-a',role:'assistant',content:'',createdAt:1}],pendingRequest:{requestId:'original',question:'问题',questionMessageId:'optimistic-q',answerMessageId:'optimistic-a'}}
  const next=applyRunEvent(state,{type:'run_started',run_id:'run',attempt:1,question_message_id:'q',answer_message_id:'a',replace_partial:false},2)
  assert.deepEqual(next.messages.map(m=>m.id),['q','a'])
})

test('unknown run accepted on retry restores one original question',()=>{
  const state={...base,currentRun:null,messages:[],pendingRequest:{requestId:'original',question:'问题',questionMessageId:'optimistic-q',answerMessageId:'optimistic-a'}}
  const next=applyRunEvent(state,{type:'run_started',run_id:'run',attempt:1,question_message_id:'q',answer_message_id:'a',replace_partial:false},2)
  assert.deepEqual(next.messages.map(m=>[m.id,m.role,m.content]),[['q','user','问题'],['a','assistant','']])
})

const metadata={schema_version:1,reply_kind:'answer',branch:'escalate',understanding_id:'u',primary_task_id:'task',clarifies_understanding_id:null,suggested_actions:[{suggestion_id:'s',type:'create_ticket',label:'投诉工单',preview_only:true,draft:{title:'t',content:'主诉求'}}]}
test('final SSE and details rebuild the same drafts without duplicates',()=>{
  const event={type:'message',run_id:'run',text:'安抚',...metadata}
  const once=applyRunEvent(base,event,2)
  const twice=applyRunEvent(once,event,2)
  assert.deepEqual(twice.messages.find(m=>m.id==='a').suggestedActions,metadata.suggested_actions)
  const detail={conversation:{conversation_id:'c',title:'服务端',created_at:'2026-10-14T00:00:00Z',updated_at:'2026-10-14T00:00:00Z'},messages:[{message_id:'a',run_id:'run',seq:2,role:'assistant',text:'安抚',created_at:'2026-10-14T00:00:00Z'}],current_run:null,actions:[],runs:[{...run,status:'completed',result_json:metadata}],next_before_seq:null}
  assert.deepEqual(mergeConversationDetail(base,detail).messages[0].suggestedActions,metadata.suggested_actions)
  assert.deepEqual(applyRunEvent(base,{...event,understanding_id:'stale'},1),base)
})

test('understanding clarification links are restored but explicit new question dismisses them',()=>{
  const event={type:'message',run_id:'run',text:'哪单？',...metadata,reply_kind:'clarification',suggested_actions:[],clarifies_understanding_id:'u'}
  const next=applyRunEvent(base,event,2)
  assert.equal(next.clarifiesUnderstandingId,'u')
  assert.equal(applyRunEvent({...next,dismissedClarificationId:'u'},event,2).clarifiesUnderstandingId,undefined)
})
