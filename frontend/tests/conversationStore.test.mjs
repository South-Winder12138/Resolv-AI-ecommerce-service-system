import assert from 'node:assert/strict'
import test from 'node:test'

const cache=new Map()
globalThis.localStorage={getItem:key=>cache.get(key)??null,setItem:(key,value)=>cache.set(key,value),removeItem:key=>cache.delete(key)}
globalThis.window={localStorage:globalThis.localStorage}
const {useAppStore}=await import('../src/stores/useAppStore.ts')
const originalFetch=globalThis.fetch
const item={id:'c',title:'服务器会话',createdAt:1,updatedAt:1,messages:[],currentRun:null,streamGeneration:2,verified:true}

test('late stream error does not affect current global UI',()=>{
  useAppStore.setState({conversations:[item],currentId:'c',error:null})
  useAppStore.getState().streamError('c',1,new Error('late failure'))
  assert.equal(useAppStore.getState().error,null)
  assert.deepEqual(useAppStore.getState().conversations[0],item)
})

test('create failure produces no ghost conversation',async()=>{
  useAppStore.setState({conversations:[],currentId:null,error:null})
  globalThis.fetch=async()=>new Response(JSON.stringify({detail:{code:'conversation_store_failed'}}),{status:503})
  try{
    await assert.rejects(useAppStore.getState().newConversation())
    assert.equal(useAppStore.getState().currentId,null)
    assert.deepEqual(useAppStore.getState().conversations,[])
  }finally{globalThis.fetch=originalFetch}
})

test('archive failure retains the server conversation',async()=>{
  useAppStore.setState({conversations:[item],currentId:'c',error:null})
  globalThis.fetch=async()=>new Response(JSON.stringify({detail:{code:'conversation_store_failed'}}),{status:503})
  try{
    await useAppStore.getState().deleteConversation('c')
    assert.equal(useAppStore.getState().currentId,'c')
    assert.equal(useAppStore.getState().conversations.length,1)
    assert.equal(useAppStore.getState().error,'conversation_store_failed')
  }finally{globalThis.fetch=originalFetch}
})

test('routing links stay with the original request across disconnect and retry',()=>{
  useAppStore.setState({conversations:[item],currentId:'c',error:null})
  const links={continue_task_id:'pending-task'}
  const generation=useAppStore.getState().beginRequest('c','取消订单 MH002','original',links)
  links.continue_task_id='changed'
  useAppStore.getState().streamError('c',generation,new Error('offline'))
  const request=useAppStore.getState().conversations[0].pendingRequest
  assert.equal(request.requestId,'original')
  assert.deepEqual(request.routingInput,{continue_task_id:'pending-task'})
})
