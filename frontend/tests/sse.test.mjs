import assert from 'node:assert/strict'
import test from 'node:test'
import { parseSseFrames, readSseEvents } from '../src/api/sse.ts'

test('chunked UTF8 CRLF and multiple frames preserve characters',async()=>{
  const bytes=new TextEncoder().encode('event: text_delta\r\ndata: {"type":"text_delta","text":"你好😀"}\r\n\r\nevent: done\r\ndata: {"type":"done"}\r\n\r\n')
  const stream=new ReadableStream({start(controller){for(let i=0;i<bytes.length;i+=1)controller.enqueue(bytes.slice(i,i+1));controller.close()}})
  const events=[]
  for await(const event of readSseEvents(stream))events.push(event)
  assert.deepEqual(events,[{type:'text_delta',text:'你好😀'},{type:'done'}])
})

test('partial frame stays buffered',()=>{
  const parsed=parseSseFrames('data: {"type":"done"}\n\ndata: {"type":"text_delta"')
  assert.deepEqual(parsed.events,[{type:'done'}])
  assert.equal(parsed.remainder,'data: {"type":"text_delta"')
})
