import type { AgentEvent, Conversation, KnowledgeSource, Message, RunStatus, ToolCall, ReplyMetadata } from '../types/agent.ts'
import type { ConversationDetail } from '../api/conversations.ts'
import { parseReplyMetadata } from '../lib/replyMetadata.ts'

export type ConversationViewState = Conversation

function validatedReply(value?: ReplyMetadata | null): ReplyMetadata | null {
  try { return value ? parseReplyMetadata(value) : null } catch { return null }
}

function replyFields(value?: ReplyMetadata | null): Partial<Message> {
  const metadata = validatedReply(value)
  return { knowledge: metadata && metadata.schema_version !== 1 ? metadata.knowledge || undefined : undefined, refund: metadata?.schema_version === 3 ? metadata.refund || undefined : undefined, branch: metadata?.branch || undefined, replyKind: metadata?.reply_kind || undefined, understandingId: metadata?.understanding_id || undefined, primaryTaskId: metadata?.primary_task_id || undefined, suggestedActions: metadata?.suggested_actions || [] }
}

export function canAsk(status: RunStatus | null): boolean {
  return status === null || status === 'completed' || status === 'failed'
}

export function applyRunEvent(state: ConversationViewState, event: AgentEvent, generation: number): ConversationViewState {
  if (generation !== state.streamGeneration) return state
  const runId = typeof event.run_id === 'string' ? event.run_id : state.currentRun?.run_id
  if (state.currentRun && runId && state.currentRun.run_id !== runId) return state
  let messages = state.messages.map(message => ({ ...message }))
  let run = state.currentRun
  let pending = state.pendingRequest
  if (event.type === 'run_started' && runId) {
    const questionId = String(event.question_message_id)
    const answerId = String(event.answer_message_id)
    const question = pending?.question || run?.question || messages.find(m => m.id === questionId)?.content || ''
    messages = messages.map(message => {
      if (message.id === pending?.questionMessageId) return { ...message, id: questionId, runId }
      if (message.id === pending?.answerMessageId) return { ...message, id: answerId, runId }
      return message
    }).filter((message, index, all) => all.findIndex(m => m.id === message.id) === index)
    if (pending && !messages.some(m => m.id === questionId)) messages.push({ id: questionId, runId, role: 'user', content: question, createdAt: Date.now() })
    if (!messages.some(m => m.id === answerId)) messages.push({ id: answerId, runId, role: 'assistant', content: '', createdAt: Date.now() })
    messages = messages.map(message => message.id === answerId ? { ...message, content: event.replace_partial ? '' : message.content, state: 'loading', error: undefined, approval: undefined, ...replyFields(null) } : message)
    run = { run_id: runId, conversation_id: state.id, request_id: pending?.requestId || run?.request_id || '', mode: run?.mode || 'agent', status: 'running', attempt: Number(event.attempt), question, question_message_id: questionId, answer_message_id: answerId, knowledge_snapshot: run?.knowledge_snapshot || [] }
  }
  const answerId = run?.answer_message_id || pending?.answerMessageId
  messages = messages.map(message => {
    if (message.id !== answerId) return message
    const rawEvents = [...(message.rawEvents || []), event]
    if (event.type === 'text_delta') return { ...message, content: message.content + String(event.text || ''), rawEvents }
    if (event.type === 'message') return { ...message, content: String(event.text || ''), state: 'complete', rawEvents, ...replyFields(event as unknown as ReplyMetadata) }
    if (event.type === 'knowledge_sources') return { ...message, knowledgeSources: (event.hits || event.sources || []) as KnowledgeSource[], rawEvents }
    if (event.type === 'tool_call') {
      const id = String(event.action_id)
      const call: ToolCall = { id, toolCallId: String(event.tool_call_id), tool: String(event.tool), arguments: event.arguments as Record<string, unknown>, status: 'running' }
      const calls = message.toolCalls || []
      return { ...message, toolCalls: calls.some(item => item.id === id) ? calls : [...calls, call], rawEvents }
    }
    if (event.type === 'tool_result') return { ...message, toolCalls: (message.toolCalls || []).map(call => call.id === event.action_id ? { ...call, result: event.result, status: 'completed' } : call), rawEvents }
    if (event.type === 'approval_required') return { ...message, approval: { action_id: String(event.action_id), approval_id: String(event.approval_id), tool: String(event.tool), arguments: event.arguments as Record<string, unknown>, message: String(event.message || event.summary || '请确认操作'), status: 'pending' }, state: 'complete', rawEvents }
    if (event.type === 'error') return { ...message, state: 'error', error: String(event.code || '连接中断'), rawEvents }
    if (event.type === 'done') return { ...message, state: event.status === 'interrupted' || event.status === 'finalizing' ? 'error' : 'complete', rawEvents }
    return { ...message, rawEvents }
  })
  if (run && event.type === 'approval_required') run = { ...run, status: 'waiting_approval' }
  if (run && event.type === 'error') run = { ...run, status: 'interrupted' }
  if (run && (event.type === 'done' || event.type === 'run_snapshot') && typeof event.status === 'string') run = { ...run, status: event.status as RunStatus }
  if (run?.status === 'completed' || run?.status === 'failed') { run = null; pending = undefined }
  const clarification = event.type === 'message' ? validatedReply(event as unknown as ReplyMetadata)?.clarifies_understanding_id || undefined : state.clarifiesUnderstandingId
  return { ...state, messages, currentRun: run, pendingRequest: pending, clarifiesUnderstandingId: clarification === state.dismissedClarificationId ? undefined : clarification, updatedAt: Date.now() }
}

export function mergeConversationDetail(state: ConversationViewState, detail: ConversationDetail): ConversationViewState {
  const runs = new Map(detail.runs.map(run => [run.run_id, run]))
  if (detail.current_run) runs.set(detail.current_run.run_id, detail.current_run)
  const messages: Message[] = detail.messages.map(message => ({ id: message.message_id, runId: message.run_id, seq: message.seq, role: message.role, content: message.text, createdAt: Date.parse(message.created_at), state: 'complete' }))
  if (detail.current_run && !messages.some(message => message.id === detail.current_run!.answer_message_id)) {
    const run = detail.current_run
    messages.push({ id: run.answer_message_id, runId: run.run_id, role: 'assistant', content: '', createdAt: Date.now(), state: run.status === 'interrupted' || run.status === 'finalizing' ? 'error' : run.status === 'running' ? 'loading' : 'complete' })
  }
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    const run = runs.get(message.runId || '')
    message.knowledgeSources = run?.knowledge_snapshot
    Object.assign(message, replyFields(run?.result_json))
    message.toolCalls = detail.actions.filter(action => action.run_id === message.runId && action.execution_status === 'completed').map(action => ({ id: action.action_id, toolCallId: action.tool_call_id, tool: action.tool_name, arguments: action.arguments, result: action.result, status: 'completed' }))
    const approval = detail.actions.find(action => action.run_id === message.runId && action.approval_id && action.execution_status === 'pending' && action.approval_status !== 'not_required')
    if (approval) message.approval = { action_id: approval.action_id, approval_id: approval.approval_id!, tool: approval.tool_name, arguments: approval.arguments, message: '请确认原操作，参数已保存。', status: approval.approval_status as 'pending' | 'approved' | 'rejected' }
  }
  const pendingRun = state.pendingRequest ? detail.runs.find(run => run.request_id === state.pendingRequest!.requestId) : null
  const pendingRequest = pendingRun?.status === 'completed' || pendingRun?.status === 'failed' ? undefined : state.pendingRequest
  const lastAnswer = [...messages].reverse().find(message => message.role === 'assistant')
  const lastMetadata = validatedReply(runs.get(lastAnswer?.runId || '')?.result_json)
  const clarification = !detail.current_run ? lastMetadata?.clarifies_understanding_id || undefined : undefined
  return { ...state, clarifiesUnderstandingId: clarification === state.dismissedClarificationId ? undefined : clarification, id: detail.conversation.conversation_id, title: detail.conversation.title, createdAt: Date.parse(detail.conversation.created_at), updatedAt: Date.parse(detail.conversation.updated_at), messages, currentRun: detail.current_run, pendingRequest, verified: true }
}
