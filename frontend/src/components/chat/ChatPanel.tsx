import { useEffect, useRef, useState } from 'react'
import { ArrowRight, AlertCircle, Check, ClipboardList, Clock3, Lightbulb, LoaderCircle, MessageCircleMore, PackageX, RotateCcw, Send, Shell, ShieldCheck, Truck, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { useAppStore } from '../../stores/useAppStore'
import { decideApproval, resumeConversation, streamAgent } from '../../api/chat'
import type { Message, RoutingInput } from '../../types/agent'
import { Button } from '../ui/Button'
import { BranchActions, PendingRequests } from './BranchActions'
import { KnowledgeReferences } from './KnowledgeReferences'
import { RefundCard, RefundApplicationStatus } from './RefundCard'
import { refundKnowledge } from '../../lib/replyMetadata'
import { formatTime } from '../../lib/utils'
import { friendlyError } from '../../lib/errorPresentation'

const quickActions = [
  { label: '查订单', detail: '快速查询你的\n订单信息', question: '查询订单 MH001', icon: ClipboardList, tone: 'blue' },
  { label: '查物流', detail: '查看包裹最新\n配送动态', question: '订单 MH001 的物流到哪里了？', icon: Truck, tone: 'teal' },
  { label: '申请退款', detail: '提交退款申请\n我们会尽快处理', question: '申请退还订单 MH001', icon: PackageX, tone: 'coral' },
  { label: '退款进度', detail: '查询退款处理\n的最新状态', question: '查看订单 MH001 的退款进度', icon: Clock3, tone: 'amber' },
]
const suggestions = ['我的订单发货了吗？', '物流到哪里了？', '我想申请退款', '退款进度怎么样？']

import { canAsk } from '../../stores/conversationState'
export function ChatPanel() {
  const { conversations, currentId, newConversation, ready, error, syncing, refreshConversations, refreshConversation, beginRequest, beginResume, applyEvent, streamError, clearClarification } = useAppStore()
  const conversation = conversations.find(item => item.id === currentId)
  const [draft, setDraft] = useState('')
  const [reassessment, setReassessment] = useState<{ conversationId: string; offerId: string } | null>(null)
  const [activeRequests, setActiveRequests] = useState<Record<string, boolean>>({})
  const [approvalPendingId, setApprovalPendingId] = useState<string | null>(null)
  const busy = !!activeRequests[currentId || '__new__'] || conversation?.currentRun?.status === 'running'
  const approvalBusy = approvalPendingId !== null && approvalPendingId === currentId
  const canSend = ready && !busy && !approvalBusy && (!conversation || (conversation.verified && canAsk(conversation.currentRun?.status || null) && !conversation.pendingRequest))
  const approvalMessage = conversation?.messages.find(message => message.approval)
  const canRecover = conversation?.verified && !busy && (conversation.currentRun?.status === 'interrupted' || conversation.currentRun?.status === 'finalizing' || (conversation.currentRun?.status === 'waiting_approval' && approvalMessage?.approval?.status !== 'pending') || (!conversation.currentRun && !!conversation.pendingRequest))
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => { void refreshConversations() }, [refreshConversations])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [conversation?.messages])
  useEffect(() => { if (textareaRef.current) { textareaRef.current.style.height = 'auto'; textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 170) + 'px' } }, [draft])
  useEffect(() => { const listener = (event: KeyboardEvent) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); void newConversation().then(() => textareaRef.current?.focus()).catch(() => {}) } }; window.addEventListener('keydown', listener); return () => window.removeEventListener('keydown', listener) }, [newConversation])
  useEffect(() => {
    if (!currentId || conversation?.currentRun?.status !== 'running' || activeRequests[currentId]) return
    const timer = window.setInterval(() => { void refreshConversation(currentId).catch(() => {}) }, 3000)
    return () => window.clearInterval(timer)
  }, [currentId, conversation?.currentRun?.status, activeRequests, refreshConversation])
  const active = (id: string, value: boolean) => setActiveRequests(previous => ({ ...previous, [id]: value }))

  const send = async (question: string, routingInput?: RoutingInput) => {
    const text = question.trim()
    const links = routingInput || (reassessment?.conversationId === currentId ? { reassesses_refund_offer_id: reassessment.offerId } : conversation?.clarifiesUnderstandingId ? { clarifies_understanding_id: conversation.clarifiesUnderstandingId } : undefined)
    if (!text || !canSend) return
    let sessionId = currentId || '__new__'
    let generation = 0
    active(sessionId, true)
    try {
      if (!currentId) {
        sessionId = await newConversation()
        active('__new__', false)
        active(sessionId, true)
      }
      const requestId = crypto.randomUUID()
      generation = beginRequest(sessionId, text, requestId, links)
      setDraft('')
      setReassessment(null)
      await streamAgent(sessionId, requestId, text, event => applyEvent(sessionId, event, generation), undefined, links)
    } catch (cause) {
      streamError(sessionId, generation, cause)
    } finally {
      active(sessionId, false)
      if (sessionId !== '__new__') await refreshConversation(sessionId).catch(() => {})
    }
  }

  const recover = async () => {
    if (!conversation || !canRecover) return
    const id = conversation.id
    let generation = 0
    active(id, true)
    try {
      await refreshConversation(id)
      generation = beginResume(id)
      let latest = useAppStore.getState().conversations.find(item => item.id === id)!
      if (latest.currentRun) {
        await resumeConversation(id, latest.currentRun.run_id, event => applyEvent(id, event, generation))
      } else if (latest.pendingRequest) {
        const original = latest.pendingRequest
        await streamAgent(id, original.requestId, original.question, event => applyEvent(id, event, generation), undefined, original.routingInput)
        latest = useAppStore.getState().conversations.find(item => item.id === id)!
        if (latest.currentRun && latest.currentRun.status !== 'waiting_approval') await resumeConversation(id, latest.currentRun.run_id, event => applyEvent(id, event, generation))
      }
    } catch (cause) { streamError(id, generation, cause) }
    finally { active(id, false); await refreshConversation(id).catch(() => {}) }
  }

  const handleApproval = async (message: Message, approved: boolean) => {
    if (!conversation?.verified || !ready || !message.approval || message.approval.status !== 'pending' || approvalBusy) return
    const id = conversation.id
    setApprovalPendingId(id)
    try { await decideApproval(id, message.approval.approval_id, approved) }
    catch (cause) { streamError(id, conversation.streamGeneration, cause) }
    finally { setApprovalPendingId(null); await refreshConversation(id).catch(() => {}) }
  }

  const latestRunId = conversation?.currentRun?.run_id || conversation?.messages.at(-1)?.runId || ''

  return <main className="chat-panel">
    <div className="chat-scroll">
      {error && <div className="message-error" role="alert" title={error}><AlertCircle size={15}/>{friendlyError(error)}<Button size="sm" variant="outline" disabled={syncing} onClick={() => void refreshConversations()}>重新同步</Button></div>}
      {!ready && <p className="seed-note">正在同步服务端会话…</p>}
      {!conversation?.messages.length ? <div className="welcome">
        <div className="welcome-hero">
          <img src="/images/sea-helper.png" alt="黄色海底客服伙伴向顾客挥手" className="welcome-mascot" />
          <div className="welcome-copy"><h1>欢迎来到海底客服站 <Shell aria-hidden="true"/></h1><p className="welcome-intro">你的海底购物小帮手，很高兴为你服务！</p><p>无论是订单查询、物流跟踪，还是退款申请，<br className="mobile-copy-break"/>我们都会全力帮助你。</p></div>
          <span className="hero-bubble">Hi!</span>
        </div>
        <div className="quick-actions">{quickActions.map(action => <button className={`quick-card ${action.tone}`} key={action.label} disabled={!canSend} onClick={() => void send(action.question)}><span className="quick-icon"><action.icon size={29} strokeWidth={2.1}/></span><strong>{action.label}</strong><span>{action.detail}</span><span className="quick-arrow"><ArrowRight size={18}/></span></button>)}</div>
        <div className="suggestion-area"><h2><Lightbulb size={19}/> 你可以这样问：</h2><div className="suggestion-list">{suggestions.map((item, index) => <button key={item} disabled={!canSend} onClick={() => void send(quickActions[index].question)}>“{item}”</button>)}</div></div>
      </div> : <div className="message-list">{conversation.messages.map(message => <div className={`message ${message.role}`} key={message.id}>
        <div className="message-avatar">{message.role === 'assistant' ? <img src="/images/sea-helper.png" alt=""/> : '你'}</div>
        <div className="message-body"><div className="message-meta"><strong>{message.role === 'assistant' ? '海底客服' : '你'}</strong><span>{formatTime(message.createdAt)}</span></div>
          {message.content ? <div className="markdown"><ReactMarkdown>{message.content}</ReactMarkdown></div> : message.state === 'loading' && !message.approval ? <div className="thinking"><LoaderCircle size={15} className="spin"/> 正在为你查询…</div> : null}
          {!message.content && message.role === 'assistant' && message.state !== 'loading' && !message.approval && <p className="message-placeholder">回复暂未生成；当前运行需要恢复时，可以点击下方“继续／重试”。</p>}
          {message.approval && <div className="approval-card"><div className="approval-title"><ShieldCheck size={17}/> {message.approval.status === 'approved' ? '已确认操作，等待恢复' : message.approval.status === 'rejected' ? '已拒绝操作，等待恢复' : '需要你确认操作'}</div><p>{message.approval.message}</p><p>订单：{String(message.approval.arguments.order_id || '')}{message.approval.arguments.reason ? ' · 原因：' + String(message.approval.arguments.reason) : ''}</p>{message.approval.status === 'pending' && <div className="approval-actions"><Button size="sm" onClick={() => void handleApproval(message, true)} disabled={approvalBusy || !ready || !conversation.verified || conversation.currentRun?.status !== 'waiting_approval'}><Check size={14}/> 确认执行</Button><Button size="sm" variant="outline" onClick={() => void handleApproval(message, false)} disabled={approvalBusy || !ready || !conversation.verified || conversation.currentRun?.status !== 'waiting_approval'}><X size={14}/> 暂不执行</Button></div>}</div>}
          {message.replyKind === 'step_limit' && <p className="branch-step-limit" role="status">本轮未完成</p>}
          {message.role === 'assistant' && <BranchActions key={`${conversation.id}:${latestRunId}:${message.id}`} actions={message.suggestedActions || []} conversationId={conversation.id} runId={message.runId || message.id}/>}
          {message.role === 'assistant' && <KnowledgeReferences knowledge={message.knowledge}/>}
          {message.role === 'assistant' && message.refund && <KnowledgeReferences knowledge={refundKnowledge(message.refund)}/>}
          {message.role === 'assistant' && message.refund?.application && <RefundApplicationStatus receipt={message.refund.application}/>}
          {message.role === 'assistant' && message.refund?.offer_id && <RefundCard key={`${conversation.id}:${message.refund.offer_id}`} conversationId={conversation.id} offerId={message.refund.offer_id} onReassess={() => { setReassessment({ conversationId: conversation.id, offerId: message.refund!.offer_id! }); setDraft(conversation.messages.find(m => m.role === 'user' && m.runId === message.runId)?.content || `请重新判断订单 ${message.refund!.order_id} 的整单退款资格，原因：`); textareaRef.current?.focus() }}/>}
          {!message.knowledge && message.knowledgeSources && <details className="knowledge-sources"><summary>召回的参考资料（{message.knowledgeSources.length}）</summary>{message.knowledgeSources.length ? message.knowledgeSources.map(source => <article key={source.chunk_id}><strong>{source.section_path || source.source}</strong><small>{source.source} · 分块 {source.chunk_index + 1} · 相似度 {source.score.toFixed(3)}</small><p>{source.answer}</p></article>) : <p>未召回已确认的知识条款。</p>}</details>}
          {message.error && <div className="message-error" title={message.error}><AlertCircle size={15}/>{friendlyError(message.error)}</div>}

        </div>
      </div>)}<div ref={bottomRef}/></div>}
    </div>
    {conversation && <PendingRequests key={conversation.id} conversationId={conversation.id} runId={latestRunId} disabled={!canSend} onContinue={(text, input) => void send(text, input)}/>}
    {reassessment?.conversationId === currentId && <p className="refund-reassessment" role="status">修改完整诉求后发送，将重新判断退款条件。<button type="button" onClick={() => setReassessment(null)}>取消重新判断</button></p>}
    <div className="composer-wrap">{conversation?.clarifiesUnderstandingId && <Button size="sm" variant="outline" disabled={!canSend} onClick={() => { clearClarification(conversation.id); setDraft(''); textareaRef.current?.focus() }}>发起新问题</Button>}{canRecover && <div className="approval-card"><p>原运行尚未完成。继续会沿用原问题、操作和已保存的决定。</p><Button size="sm" disabled={busy || approvalBusy} onClick={() => void recover()}><RotateCcw size={14}/> 继续／重试</Button></div>}<div className="composer"><MessageCircleMore size={23} className="composer-icon" aria-hidden="true"/><textarea disabled={!canSend} ref={textareaRef} value={draft} onChange={event => setDraft(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(draft) } }} placeholder="请输入你的问题…" rows={1} aria-label="消息输入框"/><Button size="icon" disabled={!draft.trim() || !canSend} onClick={() => void send(draft)} aria-label="发送消息">{busy ? <LoaderCircle size={20} className="spin"/> : <Send size={21}/>}</Button></div><div className="composer-footer"><div className="composer-categories"><span>订单</span><span>物流</span><span>退款</span><span>售后</span></div><span className="composer-shortcut">Enter 发送 · Shift + Enter 换行</span></div><div className="composer-disclaimer">{conversation?.currentRun ? '请先完成原运行或处理审批。 ' : !conversation?.messages.length ? '快捷问题使用测试订单 MH001 · ' : ''}重要业务信息请核对，写操作需要你的确认。</div></div>
  </main>
}
