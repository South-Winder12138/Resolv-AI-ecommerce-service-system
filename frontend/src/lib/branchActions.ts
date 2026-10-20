import type { SuggestedAction } from '../types/agent.ts'

export type DraftState = { suggestionId: string; type: 'transfer_human' | 'create_ticket'; title: string; content: string }

export function openDraft(action: SuggestedAction): DraftState {
  if (action.preview_only !== true || !['transfer_human', 'create_ticket'].includes(action.type) || typeof action.draft?.title !== 'string' || typeof action.draft?.content !== 'string') throw new Error('草稿入口无效')
  return { suggestionId: action.suggestion_id, type: action.type, title: action.draft.title, content: action.draft.content }
}
export function editDraft(state: DraftState, field: 'title' | 'content', value: string): DraftState { return { ...state, [field]: value } }
export function closeDraft(): null { return null }
