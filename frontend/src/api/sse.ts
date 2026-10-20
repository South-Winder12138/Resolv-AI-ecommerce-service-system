import type { AgentEvent } from '../types/agent.ts'

export function parseSseFrames(buffer: string): { events: AgentEvent[]; remainder: string } {
  const events: AgentEvent[] = []
  let remaining = buffer
  let match: RegExpExecArray | null
  while ((match = /\r?\n\r?\n/.exec(remaining))) {
    const frame = remaining.slice(0, match.index)
    remaining = remaining.slice(match.index + match[0].length)
    const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).replace(/^ /, '')).join('\n')
    if (data) {
      const event = JSON.parse(data) as AgentEvent
      if (!event || typeof event.type !== 'string') throw new Error('无效的 SSE 事件')
      events.push(event)
    }
  }
  return { events, remainder: remaining }
}

export async function* readSseEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<AgentEvent> {
  const reader = stream.getReader()
  const decoder = new TextDecoder('utf-8', { fatal: true })
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true })
      const parsed = parseSseFrames(buffer)
      buffer = parsed.remainder
      yield* parsed.events
      if (done) break
    }
    if (buffer.trim()) throw new Error('SSE 连接提前中断')
  } finally {
    reader.releaseLock()
  }
}
