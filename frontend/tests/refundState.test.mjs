import assert from 'node:assert/strict'
import test from 'node:test'
import {offer,command,receipt,fakeClock} from './refundFixtures.mjs'
async function module(){try{return await import('../src/lib/refundState.ts')}catch(e){assert.fail('refund controller missing: '+e.message)}}
function harness(create,overrides={}){
  const calls=[],clock=fakeClock(),states=[]
  const api={getRefundOffer:async(...a)=>{calls.push(['get',...a]);return structuredClone(offer)},submitRefundOffer:async(...a)=>{calls.push(['submit',...a]);return command},getRefundSubmission:async(...a)=>{calls.push(['poll',...a]);return command},resumeRefundSubmission:async(...a)=>{calls.push(['resume',...a]);return command},...overrides}
  const controller=create({api,clock,onChange:s=>states.push(s)})
  return {controller,calls,clock,states,api}
}
test('card open close and refresh never submit; money is exact',async()=>{
  const {createRefundCardController:create,formatRefundMoney:format}=await module()
  assert.equal(format(offer.money),'CNY 90071992547409.93')
  assert.equal(format({currency:'JPY',minor_units_decimal:'5',unit_precision:0}),'JPY 5')
  assert.throws(()=>format({...offer.money,minor_units_decimal:'1e5'}))
  const h=harness(create);await h.controller.open('c','o');await h.controller.refresh();h.controller.close();h.controller.dispose()
  assert.deepEqual(h.calls.map(c=>c[0]),['get','get'])
})
test('command identity and bounded polling; only manual resume posts',async()=>{
  const {createRefundCardController:create}=await module(),h=harness(create)
  await h.controller.open('c','o');await Promise.all([h.controller.submit(),h.controller.submit()])
  assert.deepEqual(h.calls.filter(c=>c[0]==='submit'),[['submit','c','o',{request_id:'request-1',confirm:true}]])
  for(let i=0;i<12;i++){if(h.clock.tasks.length)assert.equal(h.clock.tasks[0].ms,2000);await h.clock.tick()}
  assert.equal(h.calls.filter(c=>c[0]==='poll').length,10)
  assert.equal(h.clock.tasks.length,0)
  await h.controller.resume()
  assert.deepEqual(h.calls.filter(c=>c[0]==='resume'),[['resume','c','s']])
  h.controller.close();assert.equal(h.clock.tasks.length,0)
})
test('unknown submit uses the same request and a 503 retains original command',async()=>{
  const {createRefundCardController:create}=await module()
  const {ApiError,responseError}=await import('../src/api/client.ts')
  const error=await responseError(new Response(JSON.stringify({detail:{code:'refund_store_failed',retryable:true,submission_id:'s'}}),{status:503}))
  assert.equal(error.submissionId,'s');assert.equal(error.retryable,true);assert.equal(new ApiError(500,'old','r').runId,'r')
  const sent=[];let n=0
  const h=harness(create,{submitRefundOffer:async(c,o,body)=>{sent.push(body);if(++n===1)throw new TypeError('network');throw error}})
  await h.controller.open('c','o');await h.controller.submit();await h.controller.submit()
  assert.equal(sent.length,2);assert.deepEqual(sent[0],sent[1]);assert.equal(h.controller.getState().submissionId,'s')
  await h.controller.resume();assert.equal(h.calls.at(-1)[0],'resume')
})
test('stale card generations and command responses cannot replace another card',async()=>{
  const {createRefundCardController:create,reduceRefundCard:reduce}=await module()
  let resolve
  const h=harness(create,{getRefundOffer:async(c)=>c==='old'?new Promise(r=>resolve=r):{...offer,offer_id:'new'}})
  const first=h.controller.open('old','o');await h.controller.open('new','new');resolve(offer);await first
  assert.equal(h.controller.getState().conversationId,'new');assert.equal(h.controller.getState().offer.offer_id,'new')
  const s=h.controller.getState()
  assert.equal(reduce(s,{type:'offer',generation:s.generation-1,conversationId:'new',offerId:'new',offer}),s)
  await h.controller.submit();h.controller.dispose();assert.equal(h.clock.tasks.length,0)
})
test('server expired or unavailable never submits; receipt survives expiry',async()=>{
  const {createRefundCardController:create}=await module()
  for(const validity of ['expired','unavailable','needs_reassessment']){
    const h=harness(create,{getRefundOffer:async()=>({...offer,validity})})
    await h.controller.open('c','o');await h.controller.submit();assert.equal(h.calls.length,0)
  }
  const h=harness(create,{getRefundOffer:async()=>({...offer,validity:'submitted',submission:{...command,status:'completed',receipt}})})
  await h.controller.open('c','o');assert.deepEqual(h.controller.getState().submission.receipt,receipt)
})
test('switching conversation cancels polls and late results',async()=>{
  const {createRefundCardController:create}=await module(),h=harness(create)
  await h.controller.open('c','o');await h.controller.submit();assert.equal(h.clock.tasks.length,1)
  await h.controller.open('other','o');assert.equal(h.clock.tasks.length,0)
  h.controller.dispose()
})

test('an in-flight old poll cannot overwrite a completed manual resume',async()=>{
  const {createRefundCardController:create}=await module()
  let resolve
  const h=harness(create,{getRefundSubmission:async()=>new Promise(r=>resolve=r),resumeRefundSubmission:async()=>({...command,status:'completed',receipt})})
  await h.controller.open('c','o');await h.controller.submit()
  const polling=h.clock.tick();await Promise.resolve()
  await h.controller.resume();assert.equal(h.controller.getState().submission.status,'completed')
  resolve(command);await polling
  assert.equal(h.controller.getState().submission.status,'completed')
  assert.equal(h.clock.tasks.length,0)
})

test('an interrupted command with a receipt can resume publication using the same command',async()=>{
  const {createRefundCardController:create}=await module()
  const interrupted={...command,status:'interrupted',receipt,error_code:'refund_submission_publish_failed',retryable:true}
  const h=harness(create,{getRefundOffer:async()=>({...offer,validity:'submitted',submission:interrupted}),getRefundSubmission:async()=>interrupted,resumeRefundSubmission:async(...args)=>{h.calls.push(['resume',...args]);return {...command,status:'completed',receipt}}})
  await h.controller.open('c','o')
  await h.controller.submit();assert.equal(h.calls.filter(c=>c[0]==='submit').length,0)
  await h.controller.resume()
  assert.deepEqual(h.calls.filter(c=>c[0]==='resume'),[['resume','c','s']])
  assert.equal(h.controller.getState().submission.status,'completed')
  assert.deepEqual(h.controller.getState().submission.receipt,receipt)
  await h.controller.resume();assert.equal(h.calls.filter(c=>c[0]==='resume').length,1)
})

test('four refund APIs preserve exact body, command and safe path identity',async()=>{
  const api=await import('../src/api/refunds.ts'),old=globalThis.fetch,sent=[]
  globalThis.fetch=async(url,init)=>{sent.push([url,init]);return new Response(JSON.stringify(url.endsWith('/o')?offer:command),{status:url.endsWith('submit')?202:200})}
  try{
    await api.getRefundOffer('c /','o');await api.submitRefundOffer('c /','o',{request_id:'fixed',confirm:true});await api.getRefundSubmission('c /','s');await api.resumeRefundSubmission('c /','s')
    assert.equal(sent[0][0],'/api/conversations/c%20%2F/refund-offers/o')
    assert.deepEqual(JSON.parse(sent[1][1].body),{request_id:'fixed',confirm:true})
    assert.equal(sent[2][1].method,undefined);assert.equal(sent[3][1].body,'{}')
  }finally{globalThis.fetch=old}
})
