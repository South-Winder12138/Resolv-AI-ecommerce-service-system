import { useState } from 'react'
import { Archive, Heart, MessageCircleMore, PanelLeftClose, Plus, Search, Waves, X } from 'lucide-react'
import { useAppStore } from '../../stores/useAppStore'
import { canAsk } from '../../stores/conversationState'
import { cn, formatTime } from '../../lib/utils'

export function Sidebar() {
  const { conversations, currentId, newConversation, selectConversation, deleteConversation, toggleSidebar, creating, ready } = useAppStore()
  const [search, setSearch] = useState('')
  const today = new Date().toDateString()
  const visible = conversations.filter(item => item.title.toLowerCase().includes(search.trim().toLowerCase()))
  const groups = [
    { name: '今天', items: visible.filter(item => new Date(item.updatedAt).toDateString() === today) },
    { name: '更早的对话', items: visible.filter(item => new Date(item.updatedAt).toDateString() !== today) },
  ]

  return <aside className="sidebar" aria-label="客服会话">
    <div className="brand-row">
      <img className="brand-buoy" src="/images/lifebuoy-logo.png" alt=""/>
      <div className="brand-name"><strong>Resolv</strong><span>海底客服站</span></div>
      <button className="icon-button sidebar-collapse" onClick={toggleSidebar} aria-label="收起侧栏"><PanelLeftClose size={17}/></button>
    </div>
    <p className="sidebar-slogan">让每一次购物，<br/>都有回音 <Heart size={14}/></p>
    <button className="new-chat" disabled={creating || !ready} onClick={() => void newConversation().catch(() => {})}><Plus size={19} strokeWidth={3}/> 新对话 <kbd>⌘ K</kbd></button>
    <div className="sidebar-section-title"><MessageCircleMore size={17}/> 全部对话 <span>{conversations.length}</span></div>
    <label className="conversation-search"><Search size={15}/><input value={search} onChange={event => setSearch(event.target.value)} placeholder="搜索你的对话" aria-label="搜索对话"/>{search && <button onClick={() => setSearch('')} aria-label="清空搜索"><X size={13}/></button>}</label>
    <nav className="conversation-list" aria-label="会话列表">
      {visible.length === 0 && <p className="sidebar-empty">{search ? '没有找到匹配的对话。' : '还没有对话，开启你的海底服务之旅吧。'}</p>}
      {groups.map(group => group.items.length > 0 && <div key={group.name}>
        <div className="group-heading">{group.name}</div>
        {group.items.map(item => {
          const preview = [...item.messages].reverse().find(message => message.content.trim())?.content.replace(/[*_`#>]/g, '').replace(/\s+/g, ' ')
          return <div className={cn('conversation-row', currentId === item.id && 'selected')} key={item.id}>
            <button className="conversation-select" title={item.title} aria-current={currentId === item.id ? 'page' : undefined} onClick={() => { void selectConversation(item.id); if (window.matchMedia('(max-width: 700px)').matches) useAppStore.setState({ sidebarOpen: false }) }}>
              <MessageCircleMore size={16}/><span className="conversation-copy"><strong>{item.title}</strong><small>{preview || '点击继续这段对话'}</small></span><time>{new Date(item.updatedAt).toDateString() === today ? formatTime(item.updatedAt) : new Intl.DateTimeFormat('zh-CN', {month:'numeric',day:'numeric'}).format(item.updatedAt)}</time>
            </button>
            <button className="delete-chat" aria-label={`归档 ${item.title}`} title="归档会话" disabled={!item.verified || !canAsk(item.currentRun?.status || null) || !!item.pendingRequest} onClick={() => void deleteConversation(item.id)}><Archive size={14}/></button>
          </div>
        })}
      </div>)}
    </nav>
    <div className="sidebar-bottom"><Waves size={23}/><div>A brighter Bikini Bottom,<br/>together.<small>每个问题，都值得被认真回应。</small></div></div>
  </aside>
}
