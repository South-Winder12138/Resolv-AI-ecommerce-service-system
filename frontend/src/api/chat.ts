import { API_BASE, ApiError, request, responseError } from './client'
import { readSseEvents } from './sse'
import type { AgentEvent, ReplyMetadata, RoutingInput } from '../types/agent'

async function stream(path: string, body: object | undefined, onEvent: (event: AgentEvent) => void, signal?: AbortSignal) {
  const response = await fetch(API_BASE + path, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: body ? JSON.stringify(body) : undefined, signal })
  if (!response.ok) throw await responseError(response)
  if (!response.body) throw new Error('浏览器无法读取流式响应')
  let finished = false
  for await (const event of readSseEvents(response.body)) {
    onEvent(event)
    if (event.type === 'error') throw new ApiError(503, String(event.code || 'runtime_error'), typeof event.run_id === 'string' ? event.run_id : null)
    if (event.type === 'done') finished = true
  }
  if (!finished) throw new Error('连接提前中断，请继续原运行')
}
export function streamAgent(sessionId: string, requestId: string, question: string, onEvent: (event: AgentEvent) => void, signal?: AbortSignal, routingInput?: RoutingInput) {
  return stream('/agent/stream', { session_id: sessionId, request_id: requestId, question, ...routingInput }, onEvent, signal)
}
export function resumeConversation(sessionId: string, runId: string, onEvent: (event: AgentEvent) => void, signal?: AbortSignal) {
  return stream('/conversations/' + encodeURIComponent(sessionId) + '/runs/' + encodeURIComponent(runId) + '/resume', undefined, onEvent, signal)
}
export type ApprovalResponse = ({ status: 'completed'; run_id: string; answer: string } & ReplyMetadata) | { status: 'approval_required'; run_id: string; approval_id: string; action_id: string; message: string; action: { tool: string; arguments: Record<string, unknown> } } | { status: 'running' | 'waiting_approval' | 'interrupted' | 'finalizing' | 'failed'; run_id: string }
export function decideApproval(sessionId: string, approvalId: string, approved: boolean) {
  return request<ApprovalResponse>('/agent/approvals/' + encodeURIComponent(approvalId), { method: 'POST', body: JSON.stringify({ session_id: sessionId, approved }) })
}
export function getHealth() { return request<{ status: string }>('/health') }
