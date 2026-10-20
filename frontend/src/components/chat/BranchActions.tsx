import { useEffect, useRef, useState } from 'react'
import type { RoutingInput, SuggestedAction, UnderstandingTask } from '../../types/agent'
import { getUnderstandingTasks } from '../../api/conversations'
import { closeDraft, editDraft, openDraft } from '../../lib/branchActions'
import type { DraftState } from '../../lib/branchActions'
import './branchActions.css'

function DraftDialog({ initial, onClose }: { initial: DraftState; onClose: () => void }) {
  const [draft, setDraft] = useState(initial)
  const [notice, setNotice] = useState('')
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = ref.current!
    const previous = document.activeElement as HTMLElement | null
    dialog.showModal()
    return () => { dialog.close(); previous?.focus() }
  }, [])
  const copy = async () => {
    try { await navigator.clipboard.writeText(draft.title + '\n\n' + draft.content); setNotice('草稿已复制') }
    catch { setNotice('复制失败，请手动选择草稿复制。') }
  }
  return <dialog ref={ref} className="branch-draft" aria-labelledby="branch-draft-heading" onCancel={onClose} onClick={event => { if (event.target === event.currentTarget) onClose() }}>
    <h2 id="branch-draft-heading">{draft.type === 'transfer_human' ? '转人工内容确认' : '投诉工单内容确认'}</h2>
    <p className="branch-notice">仅整理草稿，尚未转接或提交</p>
    <label>标题<input value={draft.title} onChange={event => setDraft(editDraft(draft, 'title', event.target.value))}/></label>
    <label>内容<textarea rows={7} value={draft.content} onChange={event => setDraft(editDraft(draft, 'content', event.target.value))}/></label>
    <div className="branch-buttons"><button type="button" onClick={() => void copy()}>复制草稿</button><button type="button" onClick={onClose}>关闭</button></div>
    <p role="status">{notice}</p>
  </dialog>
}

export function BranchActions({ actions, conversationId, runId }: { actions: SuggestedAction[]; conversationId: string; runId: string }) {
  const [draft, setDraft] = useState<DraftState | null>(null)
  if (!actions.length) return null
  return <div className="branch-actions" data-conversation={conversationId} data-run={runId}>
    <div className="branch-buttons">{actions.map(action => <button type="button" key={action.suggestion_id} onClick={() => setDraft(openDraft(action))}>{action.label}草稿</button>)}</div>
    {draft && <DraftDialog key={`${conversationId}:${runId}:${draft.suggestionId}`} initial={draft} onClose={() => setDraft(closeDraft())}/>}
  </div>
}

export function PendingRequests({ conversationId, runId, disabled, onContinue }: { conversationId: string; runId: string; disabled: boolean; onContinue: (text: string, input: RoutingInput) => void }) {
  const [loaded, setLoaded] = useState<{ id: string; tasks: UnderstandingTask[]; error?: string } | null>(null)
  useEffect(() => {
    if (disabled) return
    let ignore = false
    getUnderstandingTasks(conversationId).then(tasks => { if (!ignore) setLoaded({ id: conversationId, tasks }) }).catch(() => { if (!ignore) setLoaded({ id: conversationId, tasks: [], error: '待处理诉求暂时无法加载，请重新同步会话。' }) })
    return () => { ignore = true }
  }, [conversationId, runId, disabled])
  if (loaded?.id !== conversationId) return null
  return <section className="branch-pending" aria-label="待处理诉求">
    {loaded.error && <p role="status">{loaded.error}</p>}
    {!!loaded.tasks.length && <><h3>待处理诉求（尚未执行）</h3>{loaded.tasks.map(task => <div key={task.task_id}><span>{task.text}</span><button type="button" disabled={disabled} onClick={() => onContinue(task.text, { continue_task_id: task.task_id })}>继续处理</button></div>)}</>}
  </section>
}
