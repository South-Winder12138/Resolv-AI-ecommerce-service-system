import { useEffect, useRef, useState } from 'react'
import * as api from '../../api/refunds'
import { canResumeRefund, canSubmitRefund, createRefundCardController, formatRefundMoney } from '../../lib/refundState'
import type { RefundCardState } from '../../lib/refundState'
import type { RefundReceipt } from '../../types/agent'
import './refundCard.css'

type Actions = { onSubmit: () => void; onResume: () => void; onReassess: () => void; onRefresh: () => void; onClose: () => void }
const validityText = {available:'可提交申请',expired:'入口已过期',needs_reassessment:'条件已变化，需要重新判断',submitted:'已登记申请',unavailable:'当前不可提交'}
const statusText = {pending:'等待审核',approved:'审核通过',rejected:'审核未通过',completed:'申请已完成'}
export function RefundApplicationStatus({receipt}: {receipt: RefundReceipt}) {
  return <section className="refund-card" aria-label="退款申请进度"><p>申请编号：{receipt.application_id}</p><p>订单：{receipt.order_id}</p><p role="status">{statusText[receipt.status]}</p>{receipt.money ? <p>申请金额：{formatRefundMoney(receipt.money)}</p> : <p>申请金额未记录</p>}<p>申请原因：{receipt.reason}</p><p>本地申请状态；审核与资金退回进度请联系人工确认。</p></section>
}
function errorText(code: string) {
  if (code === 'refund_offer_expired') return '入口已过期，请重新判断。'
  if (code === 'refund_needs_reassessment') return '条件已变化，请重新判断。'
  if (code === 'refund_conversation_archived') return '会话已归档，不能创建新申请。'
  if (code === 'refund_network_unknown') return '连接中断，提交结果暂不明确。刷新可查询结果；再次提交会沿用原请求。'
  if (code === 'refund_identity_unavailable' || code === 'refund_unavailable') return '当前无法核验身份或订单，请联系人工处理。'
  return '暂时无法完成此操作，请刷新查询；已有提交可手动恢复。'
}

// Dynamic command status is separate from the immutable judgement in the message.
export function RefundCardView({state,onSubmit,onResume,onReassess,onRefresh,onClose}: {state: RefundCardState} & Actions) {
  const offer = state.offer, receipt = state.submission?.receipt
  return <section className="refund-card" aria-label="整单退款申请" aria-busy={state.busy}>
    <div className="refund-card-heading"><strong>整单退款申请</strong><button type="button" onClick={onClose} aria-label="收起退款申请">收起</button></div>
    {!receipt && offer && <dl><dt>订单</dt><dd>{offer.order_id}</dd><dt>申请金额</dt><dd>{formatRefundMoney(offer.money)}</dd><dt>申请原因</dt><dd>{offer.reason}</dd><dt>入口有效期至</dt><dd><time dateTime={offer.expires_at}>{new Date(offer.expires_at).toLocaleString('zh-CN')}</time></dd></dl>}
    {receipt ? <RefundApplicationStatus receipt={receipt}/> : <><p role="status" aria-live="polite">{state.busy ? '正在查询或核验…' : offer ? validityText[offer.validity] : '尚未读取入口状态'}</p><p>提交后将创建本地 pending 申请，等待审核。提交前会重新核验订单、政策和金额。</p></>}
    {state.submissionId && !receipt && <p>提交编号：{state.submissionId}。{state.submission?.status === 'failed' ? '本次提交未完成。' : '可查询或恢复此提交。'}</p>}
    {state.error && <p className="refund-card-error" role="alert">{errorText(state.error)}</p>}
    {state.pollCount >= 10 && !receipt && <p>自动查询已暂停，可手动刷新或恢复。</p>}
    <div className="refund-card-actions">
      {!receipt && <button type="button" disabled={!canSubmitRefund(state)} onClick={onSubmit}>确认提交退款申请</button>}
      <button type="button" disabled={state.busy} onClick={onRefresh}>刷新状态</button>
      {state.submissionId && state.submission?.status !== 'completed' && state.submission?.status !== 'failed' && <button type="button" disabled={!canResumeRefund(state)} onClick={onResume}>恢复原提交</button>}
      {!receipt && <button type="button" disabled={state.busy} onClick={onReassess}>重新判断</button>}
    </div>
  </section>
}

export function RefundCard({conversationId,offerId,onReassess}: {conversationId: string; offerId: string; onReassess: () => void}) {
  const [state,setState] = useState<RefundCardState | null>(null)
  const controller = useRef<ReturnType<typeof createRefundCardController> | null>(null)
  useEffect(()=>{
    const current = createRefundCardController({api,onChange:setState})
    controller.current = current
    void current.open(conversationId,offerId)
    return ()=>{ current.dispose(); if (controller.current === current) controller.current = null }
  },[conversationId,offerId])
  if (!state || state.conversationId !== conversationId || state.offerId !== offerId) return <p role="status">正在读取退款入口…</p>
  if (!state.open) return <button className="refund-card-open" type="button" onClick={()=>void controller.current?.open(conversationId,offerId)}>查看退款申请</button>
  return <RefundCardView state={state} onSubmit={()=>void controller.current?.submit()} onResume={()=>void controller.current?.resume()} onRefresh={()=>void controller.current?.refresh()} onClose={()=>controller.current?.close()} onReassess={onReassess}/>
}
