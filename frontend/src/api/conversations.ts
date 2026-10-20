import { request } from './client.ts'
import type { KnowledgeSource, RunRecord, UnderstandingTask } from '../types/agent'

export type ConversationRecord = { conversation_id: string; title: string; created_at: string; updated_at: string; run_status?: RunRecord['status'] | null }
export type ActionRecord = { action_id: string; run_id: string; tool_name: string; tool_call_id: string; arguments: Record<string, unknown>; approval_id: string | null; approval_status: 'not_required' | 'pending' | 'approved' | 'rejected'; execution_status: 'pending' | 'completed'; result: unknown }
export type ConversationDetail = { conversation: ConversationRecord; messages: { message_id: string; run_id: string; seq: number; role: 'user' | 'assistant'; text: string; created_at: string }[]; current_run: RunRecord | null; actions: ActionRecord[]; runs: (RunRecord & { knowledge_snapshot: KnowledgeSource[] })[]; next_before_seq: number | null }
export function createConversation() { return request<ConversationRecord>('/conversations', { method: 'POST' }) }
export async function listConversations() {
  const records: ConversationRecord[] = []
  let cursor: string | null = null
  do {
    const page: { conversations: ConversationRecord[]; next_cursor: string | null } = await request('/conversations?limit=100' + (cursor ? '&cursor=' + encodeURIComponent(cursor) : ''))
    records.push(...page.conversations)
    cursor = page.next_cursor
  } while (cursor)
  return records
}
export async function getConversation(id: string) {
  const path = '/conversations/' + encodeURIComponent(id)
  const detail = await request<ConversationDetail>(path + '?limit=100')
  let cursor = detail.next_before_seq
  while (cursor) {
    const page = await request<ConversationDetail>(path + '?limit=100&before_seq=' + cursor)
    detail.messages = [...page.messages, ...detail.messages]
    detail.actions = [...page.actions, ...detail.actions]
    detail.runs = [...page.runs, ...detail.runs]
    cursor = page.next_before_seq
  }
  detail.actions = [...new Map(detail.actions.map(action => [action.action_id, action])).values()]
  detail.runs = [...new Map(detail.runs.map(run => [run.run_id, run])).values()]
  return detail
}
export function archiveConversation(id: string) { return request<{ status: string }>('/chat/sessions/' + encodeURIComponent(id), { method: 'DELETE' }) }

export async function getUnderstandingTasks(conversationId: string, status: 'pending' | 'selected' = 'pending') {
  const records: UnderstandingTask[] = []
  let cursor: string | null = null
  do {
    const page: { requests: UnderstandingTask[]; next_cursor: string | null } = await request('/conversations/' + encodeURIComponent(conversationId) + '/understanding-tasks?limit=100&status=' + status + (cursor ? '&cursor=' + encodeURIComponent(cursor) : ''))
    records.push(...page.requests)
    cursor = page.next_cursor
  } while (cursor)
  return records
}
