import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Moon, Sun, PanelLeftOpen, PanelRightOpen, Wifi, WifiOff } from 'lucide-react'
import { Sidebar } from './components/layout/Sidebar'
import { ChatPanel } from './components/chat/ChatPanel'
import { Inspector } from './components/inspector/Inspector'
import { getHealth } from './api/chat'
import { useAppStore } from './stores/useAppStore'

export default function App() {
  const { view, setView, theme, toggleTheme, sidebarOpen, inspectorOpen, toggleSidebar, toggleInspector } = useAppStore()
  const health = useQuery({ queryKey: ['health'], queryFn: getHealth, refetchInterval: 30000 })
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])
  useEffect(() => {
    const compact = window.matchMedia('(max-width: 700px)')
    const medium = window.matchMedia('(max-width: 1180px)')
    const adjustPanels = () => useAppStore.setState({ sidebarOpen: !compact.matches, inspectorOpen: !medium.matches })
    const closePanels = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && medium.matches) useAppStore.setState({ inspectorOpen: false, ...(compact.matches ? { sidebarOpen: false } : {}) })
    }
    adjustPanels()
    compact.addEventListener('change', adjustPanels)
    medium.addEventListener('change', adjustPanels)
    window.addEventListener('keydown', closePanels)
    return () => { compact.removeEventListener('change', adjustPanels); medium.removeEventListener('change', adjustPanels); window.removeEventListener('keydown', closePanels) }
  }, [])

  return <div className="app-shell">
    {sidebarOpen && <Sidebar/>}
    <div className="workspace">
      <header className="sea-banner">
        <div className="topbar">
          <div className="topbar-left">{!sidebarOpen && <button className="icon-button banner-button" onClick={toggleSidebar} aria-label="展开侧栏"><PanelLeftOpen size={20}/></button>}</div>
          <div className="topbar-right">
            <div className="view-switch" role="group" aria-label="视图切换"><button aria-pressed={view === 'customer'} className={view === 'customer' ? 'active' : ''} onClick={() => setView('customer')}>客服视图</button><button aria-pressed={view === 'debug'} className={view === 'debug' ? 'active' : ''} onClick={() => setView('debug')}>调试视图</button></div>
            <div className={`health-pill ${health.isSuccess ? 'online' : ''}`} role="status">{health.isSuccess ? <Wifi size={14}/> : <WifiOff size={14}/>}<span>{health.isSuccess ? '客服在线' : health.isPending ? '连接中' : '后端离线'}</span></div>
            <button className="icon-button banner-button" onClick={toggleTheme} aria-label={theme === 'light' ? '切换深色模式' : '切换浅色模式'}>{theme === 'light' ? <Moon size={18}/> : <Sun size={18}/>}</button>
            {view === 'debug' && !inspectorOpen && <button className="icon-button banner-button" onClick={toggleInspector} aria-label="展开检查器"><PanelRightOpen size={18}/></button>}
          </div>
        </div>
        <p className="sea-banner-caption">比奇堡的朋友们，<br/>我们一直在这里！</p>
        <img className="krusty-sign" src="/images/krusty-sign.png" alt="蟹堡王 THE KRUSTY KRAB"/>
        <p className="sea-banner-motto">Good customers.<br/>Make a Krabby day!</p>
      </header>
      <div className="content-area"><ChatPanel/>{view === 'debug' && inspectorOpen && <Inspector/>}</div>
    </div>
    {sidebarOpen && <button className="panel-backdrop sidebar-backdrop" onClick={toggleSidebar} aria-label="关闭侧栏"/>}
  </div>
}
