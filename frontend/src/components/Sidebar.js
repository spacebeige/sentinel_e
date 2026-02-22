import React from 'react';
import { Plus, Sun, Moon, BookOpen, Wifi, WifiOff, Skull, MessageSquare } from 'lucide-react';

const relativeTime = (iso) => {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (m < 1) return 'now';
  if (m < 60) return `${m}m`;
  if (h < 24) return `${h}h`;
  return `${d}d`;
};

const Sidebar = ({
  history, onSelectRun, onNewChat, activeMode, setMode,
  subMode, setSubMode, killActive, setKillActive,
  rounds, setRounds, activeChatId, showLearning, onToggleLearning,
  darkMode, toggleDarkMode, serverStatus,
}) => {

  const displayHistory = history.slice(0, 50);

  return (
    <div className="flex flex-col h-screen" style={{
      width: 'var(--sidebar-width)',
      minWidth: 'var(--sidebar-width)',
      backgroundColor: 'var(--bg-sidebar)',
      color: 'var(--text-sidebar)',
    }}>
      
      {/* Top: Brand + New Chat */}
      <div className="p-3 space-y-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div className="flex items-center justify-between px-1">
          <span className="text-sm font-semibold tracking-wide">Sentinel-E</span>
          <div className="flex items-center gap-1">
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded" 
              style={{ backgroundColor: serverStatus === 'online' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)', 
                       color: serverStatus === 'online' ? 'var(--accent-green)' : 'var(--accent-red)' }}>
              {serverStatus === 'online' ? <Wifi className="w-3 h-3 inline" /> : <WifiOff className="w-3 h-3 inline" />}
            </span>
          </div>
        </div>
        
        <button onClick={onNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-all"
          style={{ border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-sidebar)' }}
          onMouseEnter={e => e.target.style.backgroundColor = 'var(--bg-sidebar-hover)'}
          onMouseLeave={e => e.target.style.backgroundColor = 'transparent'}
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Mode Selector */}
      <div className="px-3 pt-3 pb-1 space-y-2">
        <div className="text-[10px] font-mono uppercase tracking-widest px-1" style={{ color: 'var(--text-sidebar-muted)' }}>
          Mode
        </div>
        <div className="flex gap-1 p-1 rounded-lg" style={{ backgroundColor: 'rgba(0,0,0,0.3)' }}>
          {['standard', 'experimental'].map(m => (
            <button key={m} onClick={() => { setMode(m); if (m !== activeMode) { onNewChat(); setKillActive(false); } }}
              className="flex-1 text-[10px] uppercase font-semibold tracking-wider py-1.5 rounded-md transition-all"
              style={{
                backgroundColor: activeMode === m ? (m === 'standard' ? 'rgba(34,197,94,0.2)' : 'rgba(249,115,22,0.2)') : 'transparent',
                color: activeMode === m ? (m === 'standard' ? 'var(--accent-green)' : 'var(--accent-orange)') : 'var(--text-sidebar-muted)',
                border: activeMode === m ? `1px solid ${m === 'standard' ? 'rgba(34,197,94,0.3)' : 'rgba(249,115,22,0.3)'}` : '1px solid transparent',
              }}
            >
              {m === 'standard' ? 'STD' : 'EXP'}
            </button>
          ))}
        </div>

        {/* Sub-mode selector (experimental only) */}
        {activeMode === 'experimental' && (
          <div className="space-y-1.5">
            <div className="flex gap-1">
              {['debate', 'glass', 'evidence'].map(sm => (
                <button key={sm} onClick={() => { setSubMode(sm); setKillActive(false); }}
                  className="flex-1 text-[9px] uppercase font-semibold tracking-wider py-1 rounded-md transition-all"
                  style={{
                    backgroundColor: subMode === sm ? 'rgba(255,255,255,0.1)' : 'transparent',
                    color: subMode === sm ? 'var(--text-sidebar)' : 'var(--text-sidebar-muted)',
                    border: subMode === sm ? '1px solid rgba(255,255,255,0.15)' : '1px solid transparent',
                  }}
                >
                  {sm}
                </button>
              ))}
            </div>

            {/* Kill toggle (glass only) */}
            {subMode === 'glass' && (
              <button onClick={() => setKillActive(k => !k)}
                className="w-full flex items-center justify-center gap-1.5 text-[9px] uppercase font-bold py-1.5 rounded-md transition-all"
                style={{
                  backgroundColor: killActive ? 'rgba(239,68,68,0.2)' : 'transparent',
                  color: killActive ? 'var(--accent-red)' : 'var(--text-sidebar-muted)',
                  border: killActive ? '1px solid rgba(239,68,68,0.3)' : '1px solid rgba(255,255,255,0.08)',
                }}
              >
                <Skull className="w-3 h-3" />
                Kill {killActive ? 'Active' : 'Switch'}
              </button>
            )}

            {/* Rounds */}
            <div className="flex items-center gap-2 px-1">
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-sidebar-muted)' }}>Rounds</span>
              <input type="number" min="1" max="6" value={rounds}
                onChange={e => setRounds(Math.max(1, Math.min(6, parseInt(e.target.value) || 1)))}
                className="w-10 text-center text-xs font-mono rounded px-1 py-0.5 focus:outline-none"
                style={{ backgroundColor: 'rgba(0,0,0,0.3)', color: 'var(--text-sidebar)', border: '1px solid rgba(255,255,255,0.1)' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
        <div className="text-[10px] font-mono uppercase tracking-widest px-2 py-1" style={{ color: 'var(--text-sidebar-muted)' }}>
          History
        </div>
        {displayHistory.map(run => (
          <button key={run.id} onClick={() => onSelectRun(run)}
            className="sidebar-item w-full text-left flex items-center gap-2 text-[13px] truncate"
            style={{
              backgroundColor: activeChatId === run.id ? 'var(--bg-sidebar-hover)' : 'transparent',
              borderLeft: activeChatId === run.id ? '2px solid var(--accent-blue)' : '2px solid transparent',
            }}
          >
            <MessageSquare className="w-3.5 h-3.5 flex-shrink-0" style={{ opacity: 0.5 }} />
            <span className="truncate flex-1">{run.summary}</span>
            <span className="text-[9px] flex-shrink-0" style={{ color: 'var(--text-sidebar-muted)' }}>{relativeTime(run.timestamp)}</span>
          </button>
        ))}
        {displayHistory.length === 0 && (
          <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--text-sidebar-muted)' }}>
            No conversations yet
          </div>
        )}
      </div>

      {/* Bottom: Learning + Theme */}
      <div className="p-3 space-y-1" style={{ borderTop: '1px solid rgba(255,255,255,0.08)' }}>
        <button onClick={onToggleLearning}
          className="sidebar-item w-full flex items-center gap-2 text-[13px]"
          style={{ color: showLearning ? 'var(--accent-purple)' : 'var(--text-sidebar-muted)' }}
        >
          <BookOpen className="w-4 h-4" />
          <span>Learning Dashboard</span>
        </button>
        <button onClick={toggleDarkMode}
          className="sidebar-item w-full flex items-center gap-2 text-[13px]"
          style={{ color: 'var(--text-sidebar-muted)' }}
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;