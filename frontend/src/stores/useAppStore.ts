import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { archiveConversation, createConversation, getConversation, listConversations } from '../api/conversations.ts'
import type { AgentEvent, Conversation, RoutingInput } from '../types/agent'
import { applyRunEvent, canAsk, mergeConversationDetail } from './conversationState.ts'

type Store = {
  conversations: Conversation[]; currentId: string | null; view: 'customer' | 'debug'; theme: 'light' | 'dark'; sidebarOpen: boolean; inspectorOpen: boolean
  ready: boolean; syncing: boolean; creating: boolean; error: string | null
  newConversation: () => Promise<string>; selectConversation: (id: string) => Promise<void>; deleteConversation: (id: string) => Promise<void>
  refreshConversations: () => Promise<void>; refreshConversation: (id: string) => Promise<void>
  beginRequest: (id: string, question: string, requestId: string, routingInput?: RoutingInput) => number; clearClarification: (id: string) => void; beginResume: (id: string) => number
  applyEvent: (id: string, event: AgentEvent, generation: number) => void; streamError: (id: string, generation: number, error: unknown) => void
  setView: (view: 'customer' | 'debug') => void; toggleTheme: () => void; toggleSidebar: () => void; toggleInspector: () => void
}
const change = (items: Conversation[], id: string, update: (conversation: Conversation) => Conversation) => items.map(item => item.id === id ? update(item) : item)
const errorText = (error: unknown) => error instanceof Error ? error.message : '无法同步服务端会话'
let creation: Promise<string> | null = null

export const useAppStore = create<Store>()(persist((set, get) => ({
  conversations: [], currentId: null, view: 'debug', theme: 'light', sidebarOpen: true, inspectorOpen: true,
  ready: false, syncing: false, creating: false, error: null,
  newConversation: () => {
    if (creation) return creation
    set({ creating: true, error: null })
    creation = (async () => {
      try {
        const record = await createConversation()
        const item: Conversation = { id: record.conversation_id, title: record.title, createdAt: Date.parse(record.created_at), updatedAt: Date.parse(record.updated_at), messages: [], currentRun: null, streamGeneration: 0, verified: true }
        set(state => ({ conversations: [item, ...state.conversations], currentId: item.id }))
        return item.id
      } catch (error) { set({ error: errorText(error) }); throw error }
      finally { set({ creating: false }); creation = null }
    })()
    return creation
  },
  selectConversation: async id => {
    set(state => ({ currentId: id, error: null, conversations: state.conversations.map(item => item.id === id || item.id === state.currentId ? { ...item, streamGeneration: item.streamGeneration + 1, verified: item.id === id ? false : item.verified } : item) }))
    try { await get().refreshConversation(id) } catch (error) { set({ error: errorText(error) }) }
  },
  deleteConversation: async id => {
    const item = get().conversations.find(item => item.id === id)
    if (!item?.verified || !canAsk(item.currentRun?.status || null) || item.pendingRequest) return
    try {
      await archiveConversation(id)
      set(state => ({ conversations: state.conversations.filter(item => item.id !== id), currentId: state.currentId === id ? state.conversations.find(item => item.id !== id)?.id || null : state.currentId, error: null }))
      if (get().currentId) await get().selectConversation(get().currentId!)
    } catch (error) { set({ error: errorText(error) }) }
  },
  refreshConversation: async id => {
    const generation = get().conversations.find(item => item.id === id)?.streamGeneration
    const detail = await getConversation(id)
    set(state => ({ conversations: change(state.conversations, id, item => item.streamGeneration === generation ? mergeConversationDetail(item, detail) : item) }))
  },
  refreshConversations: async () => {
    if (get().syncing) return
    set({ syncing: true, error: null })
    try {
      const records = await listConversations()
      set(state => ({ conversations: records.map(record => {
        const cached = state.conversations.find(item => item.id === record.conversation_id)
        return { ...cached, id: record.conversation_id, title: record.title, createdAt: Date.parse(record.created_at), updatedAt: Date.parse(record.updated_at), messages: cached?.messages || [], currentRun: cached?.currentRun || null, streamGeneration: (cached?.streamGeneration || 0) + 1, pendingRequest: cached?.pendingRequest, verified: false }
      }), currentId: records.some(record => record.conversation_id === state.currentId) ? state.currentId : records[0]?.conversation_id || null }))
      if (get().currentId) await get().refreshConversation(get().currentId!)
      set({ ready: true })
    } catch (error) { set({ error: errorText(error), ready: false }) }
    finally { set({ syncing: false }) }
  },
  beginRequest: (id, question, requestId, routingInput) => {
    const item = get().conversations.find(item => item.id === id)
    if (!item?.verified || !canAsk(item.currentRun?.status || null) || item.pendingRequest) throw new Error('请先处理原运行')
    const generation = item.streamGeneration + 1
    const questionId = crypto.randomUUID(), answerId = crypto.randomUUID()
    set(state => ({ conversations: change(state.conversations, id, conversation => ({ ...conversation, streamGeneration: generation, pendingRequest: { requestId, question, routingInput: routingInput ? { ...routingInput } : undefined, questionMessageId: questionId, answerMessageId: answerId }, messages: [...conversation.messages, { id: questionId, role: 'user', content: question, createdAt: Date.now() }, { id: answerId, role: 'assistant', content: '', createdAt: Date.now(), state: 'loading', toolCalls: [] }] })) }))
    return generation
  },
  clearClarification: id => set(state => ({ conversations: change(state.conversations, id, item => ({ ...item, dismissedClarificationId: item.clarifiesUnderstandingId, clarifiesUnderstandingId: undefined })) })),
  beginResume: id => {
    const item = get().conversations.find(item => item.id === id)
    if (!item) throw new Error('会话不存在')
    const generation = item.streamGeneration + 1
    set(state => ({ conversations: change(state.conversations, id, conversation => ({ ...conversation, streamGeneration: generation })) }))
    return generation
  },
  applyEvent: (id, event, generation) => set(state => ({ conversations: change(state.conversations, id, item => applyRunEvent(item, event, generation)) })),
  streamError: (id, generation, error) => set(state => {
    const item = state.conversations.find(conversation => conversation.id === id)
    if (item && item.streamGeneration !== generation) return state
    return { error: errorText(error), conversations: change(state.conversations, id, item => applyRunEvent(item, { type: 'error', code: errorText(error) }, generation)) }
  }),
  setView: view => set({ view }), toggleTheme: () => set({ theme: get().theme === 'light' ? 'dark' : 'light' }), toggleSidebar: () => set(state => ({ sidebarOpen: !state.sidebarOpen })), toggleInspector: () => set(state => ({ inspectorOpen: !state.inspectorOpen })),
}), { name: 'resolv-web-v2', partialize: state => ({ conversations: state.conversations.map(item => ({ ...item, verified: false })), currentId: state.currentId, view: state.view, theme: state.theme, sidebarOpen: state.sidebarOpen, inspectorOpen: state.inspectorOpen }) }))
