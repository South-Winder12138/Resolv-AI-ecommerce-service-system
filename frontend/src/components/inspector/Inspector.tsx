import { useState } from 'react'
import { Activity, Braces, ChevronDown, CheckCircle2, Clock3, Database, PackageSearch, PanelRightClose, Wrench, Zap } from 'lucide-react'
import { useAppStore } from '../../stores/useAppStore'
import type { ToolCall } from '../../types/agent'
import { KnowledgeReferences } from '../chat/KnowledgeReferences'

const fieldLabels: Record<string, string> = { order_id: '订单号', tracking_no: '物流单号', refund_id: '退款号', status: '状态', location: '当前位置', found: '查询结果', reason: '原因', changed: '是否变更', amount: '金额' }
const statusLabels: Record<string, string> = { shipped: '已发货', in_transit: '运输中', delivered: '已送达', processing: '处理中', pending: '待处理', cancelled: '已取消' }
const inspectorTabs = [{ id: 'overview', label: '业务概览' }, { id: 'tools', label: '工具调用' }, { id: 'raw', label: '原始事件' }] as const
const toolLabels: Record<string, string> = { query_order: '订单信息', query_logistics: '物流动态', query_refund_status: '退款进度', apply_refund: '退款申请', cancel_order: '订单取消' }
function asRows(value: unknown): [string, unknown][] { return value && typeof value === 'object' && !Array.isArray(value) ? Object.entries(value) : [] }
function formatValue(key: string, value: unknown) { if (key === 'status' && typeof value === 'string') return statusLabels[value] || value; if (typeof value === 'boolean') return value ? '是' : '否'; if (value === null) return '—'; return typeof value === 'object' ? JSON.stringify(value) : String(value) }
function JsonBlock({ value }: { value: unknown }) { return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre> }
function ToolCard({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false)
  return <div className="tool-card"><button className="tool-heading" onClick={() => setOpen(!open)} aria-expanded={open}><div className="tool-icon"><Wrench size={16}/></div><div><strong>{call.tool}</strong><span>{call.status === 'completed' ? '已完成' : '执行中'}</span></div><ChevronDown size={15} className={open ? 'rotate' : ''}/></button>{open && <div className="tool-detail"><span>请求参数</span><JsonBlock value={call.arguments}/>{call.result !== undefined && <><span>工具结果</span><JsonBlock value={call.result}/></>}</div>}</div>
}

export function Inspector() {
  const { conversations, currentId, toggleInspector } = useAppStore()
  const [tab, setTab] = useState<'overview' | 'tools' | 'raw'>('overview')
  const conversation = conversations.find(item => item.id === currentId)
  const message = [...(conversation?.messages || [])].reverse().find(item => item.role === 'assistant')
  const calls = message?.toolCalls || []
  const businessCalls = calls.filter(call => call.result !== undefined && asRows(call.result).length > 0)
  const activeStatus = conversation?.currentRun?.status
  const status = activeStatus === 'waiting_approval' ? '等待确认' : activeStatus === 'interrupted' ? '等待恢复' : activeStatus === 'finalizing' ? '正在保存' : activeStatus === 'running' || message?.state === 'loading' ? '正在处理' : activeStatus === 'failed' || message?.state === 'error' ? '处理出错' : message ? '已完成' : '等待新问题'

  return <>
    <button className="panel-backdrop inspector-backdrop" onClick={toggleInspector} aria-label="关闭检查器"/>
    <aside className="inspector" aria-label="Agent 检查器">
      <div className="inspector-hero"><div className="inspector-heading-copy"><h2>蟹老板工作台</h2><p>精打细算 · 服务至上</p></div><img src="/images/mr-krabs.png" alt="蟹老板"/><span className="krabs-motto">顾客开心，<br/>生意才会红火！</span><button className="icon-button inspector-collapse" onClick={toggleInspector} aria-label="收起检查器"><PanelRightClose size={17}/></button></div>
      <div className="inspector-tabs" role="tablist" aria-label="检查器选项卡">{inspectorTabs.map((item, index) => <button key={item.id} id={`inspector-tab-${item.id}`} role="tab" tabIndex={tab === item.id ? 0 : -1} aria-selected={tab === item.id} aria-controls="inspector-tabpanel" className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id)} onKeyDown={event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
        event.preventDefault()
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? inspectorTabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + inspectorTabs.length) % inspectorTabs.length
        setTab(inspectorTabs[next].id)
        event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('button')[next]?.focus()
      }}>{item.label}</button>)}</div>
      <div className="inspector-content" id="inspector-tabpanel" role="tabpanel" aria-labelledby={`inspector-tab-${tab}`}>
        {tab === 'overview' && <>
          <section className="inspector-section"><div className="section-heading"><PackageSearch size={17}/> 订单与售后</div>{businessCalls.length ? businessCalls.map(call => <div className="business-card" key={call.id}><div className="business-title"><CheckCircle2 size={16}/><strong>{toolLabels[call.tool] || call.tool}</strong><span>实时结果</span></div>{asRows(call.result).map(([key, value]) => <div className="business-row" key={key}><span>{fieldLabels[key] || key}</span><strong className={key === 'status' ? 'business-status' : undefined}>{formatValue(key, value)}</strong></div>)}</div>) : <div className="business-empty"><PackageSearch size={37} strokeWidth={1.4}/><strong>等你开启第一段查询</strong><p>订单、物流与退款结果，<br/>都会在这里为你清楚呈现。</p><span>试试「查订单」</span></div>}</section>
          <section className="inspector-section"><div className="section-heading"><Activity size={16}/> Agent 运行</div><div className="run-card"><div className="run-status"><span className={`status-dot ${status === '处理出错' ? 'error-dot' : ''}`}/><strong>{status}</strong>{message && <span className="run-tag">LIVE</span>}</div><div className="run-row"><span><Clock3 size={14}/> 用时</span><strong>{message?.durationMs === undefined ? '—' : `${(message.durationMs / 1000).toFixed(1)} 秒`}</strong></div><div className="run-row"><span><Zap size={14}/> 模型 / Tokens</span><strong>暂未提供</strong></div><div className="run-row"><span><Database size={14}/> 会话 ID</span><strong className="session-id" title={currentId || ''}>{currentId ? currentId.slice(0, 8) : '—'}</strong></div></div></section>
          {message?.knowledge && <KnowledgeReferences knowledge={message.knowledge}/>}
          {!message?.knowledge && message?.knowledgeSources?.length ? <section className="inspector-section"><div className="section-heading"><Database size={16}/> 参考资料</div><div className="inspector-empty">本轮检索到 {message.knowledgeSources.length} 份资料，展开聊天回复下的参考资料查看条款。</div></section> : null}
        </>}
        {tab === 'tools' && <section className="inspector-section"><div className="section-heading"><Wrench size={16}/> 工具调用 <span className="count">{calls.length}</span></div>{calls.length ? <div className="tool-list">{calls.map(call => <ToolCard key={call.id} call={call}/>)}</div> : <div className="inspector-empty">此轮没有工具调用。试试查询订单 MH001 或物流信息。</div>}</section>}
        {tab === 'raw' && <section className="inspector-section"><div className="section-heading"><Braces size={16}/> SSE 原始事件</div>{message?.rawEvents?.length ? <JsonBlock value={message.rawEvents}/> : <div className="inspector-empty">本轮暂无原始流事件记录，可在业务概览或工具调用中查看已返回的数据。</div>}</section>}
      </div>
      <div className="inspector-footnote"><CheckCircle2 size={12}/> 业务数据来自当前 Agent 的真实结果</div>
    </aside>
  </>
}
