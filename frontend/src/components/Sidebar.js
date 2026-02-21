import React from 'react';
import { MessageSquare, Plus, History, ShieldAlert, BarChart2, Sun, Moon } from 'lucide-react';

const relativeTime = (iso) => {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  if (h < 24) return `${h}h ago`;
  return `${d}d ago`;
};

const Sidebar = ({ history, onSelectRun, onNewChat, activeMode, activeChatId, onToggleLearning, showLearning, darkMode, toggleDarkMode }) => {

  // Fix: standard mode shows 'standard' chats; experimental mode shows sigma/forensic/conversational/experimental
  const filteredHistory = history.filter(run => {
    if (activeMode === 'standard') return run.mode === 'standard';
    return ['conversational', 'forensic', 'experimental'].includes(run.mode) || run.mode === 'experimental';
  });
  // Show all chats if filter yields nothing (graceful fallback)
  const displayHistory = filteredHistory.length > 0 ? filteredHistory : history;

  return (
    <div className="w-64 bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-300 h-screen flex flex-col border-r border-slate-200 dark:border-slate-800 transition-colors duration-200">
      
      {/* Brand / New Chat */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 space-y-4">
        <div className="flex items-center space-x-2 px-2">
            <ShieldAlert className={`w-5 h-5 ${activeMode === 'experimental' ? 'text-amber-500' : 'text-emerald-500'}`} />
            <span className="font-semibold text-slate-900 dark:text-white tracking-wide text-sm">SENTINEL-E</span>
        </div>
        
        <div className="flex gap-2">
            <button 
                onClick={() => onNewChat()}
                className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md border transition-all duration-200 text-xs font-medium ${
                    !showLearning 
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white border-slate-300 dark:border-slate-700 shadow-sm' 
                    : 'bg-transparent text-slate-500 dark:text-slate-400 border-transparent hover:bg-slate-200 dark:hover:bg-slate-800'
                }`}
            >
              <Plus className="w-3 h-3" />
              <span>Chat</span>
            </button>
            <button 
                onClick={onToggleLearning}
                className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md border transition-all duration-200 text-xs font-medium ${
                    showLearning 
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white border-slate-300 dark:border-slate-700 shadow-sm' 
                    : 'bg-transparent text-slate-500 dark:text-slate-400 border-transparent hover:bg-slate-200 dark:hover:bg-slate-800'
                }`}
            >
              <BarChart2 className="w-3 h-3" />
              <span>Learning</span>
            </button>
        </div>
      </div>
      
      {/* History List (Only show if NOT in Learning View) */}
      {!showLearning && (
          <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 scrollbar-track-transparent">
            <div className="flex items-center justify-between px-2 mt-4 mb-2">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
                    {activeMode === 'standard' ? 'Standard Logs' : 'Experimental Logs'}
                </h3>
                <span className="text-[10px] bg-slate-200 dark:bg-slate-800 px-1.5 py-0.5 rounded text-slate-600 dark:text-slate-500">
                    {displayHistory.length}
                </span>
            </div>
            
            {displayHistory.length === 0 && (
                <div className="text-xs text-slate-500 italic ml-4 py-2">No chats yet.</div>
            )}

            {displayHistory.map((run) => {
              const isActive = run.id === activeChatId;
              return (
              <button 
                key={run.id}
                onClick={() => onSelectRun(run)}
                className={`w-full text-left p-2 rounded-md transition-colors flex items-center space-x-3 text-sm mb-1 group ${
                  isActive
                    ? 'bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20'
                    : 'hover:bg-slate-200 dark:hover:bg-slate-800 border border-transparent'
                }`}
              >
                {run.mode === 'standard' ? (
                    <MessageSquare className={`w-4 h-4 ${isActive ? 'text-emerald-500' : 'text-emerald-500/80 dark:text-emerald-400 opacity-60 group-hover:opacity-100'} transition-opacity`} />
                ) : (
                    <History className={`w-4 h-4 ${isActive ? 'text-amber-500' : 'text-amber-500/80 dark:text-amber-400 opacity-60 group-hover:opacity-100'} transition-opacity`} />
                )}
                <div className="flex-1 truncate">
                    <span className={`block truncate text-sm ${isActive ? 'text-slate-900 dark:text-white font-medium' : 'text-slate-600 dark:text-slate-400 group-hover:text-slate-900 dark:group-hover:text-white'} transition-colors`}>
                        {run.summary || 'Chat'}
                    </span>
                    <span className="text-[10px] text-slate-400 dark:text-slate-600 block">
                        {relativeTime(run.timestamp)}
                    </span>
                </div>
              </button>
              );
            })}
          </div>
      )}

      {/* Footer / User */}
      <div className="p-4 border-t border-slate-200 dark:border-slate-800">
        <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 text-xs text-slate-500">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-xs ring-1 ring-white/10 ${
                    activeMode === 'experimental' ? 'bg-amber-600' : 'bg-emerald-600'
                }`}>
                    AI
                </div>
                <div className="flex flex-col">
                    <span className="text-slate-700 dark:text-slate-300 font-medium">System Operator</span>
                    <span className="text-slate-400 dark:text-slate-600">{activeMode === 'standard' ? 'Standard Mode' : 'Sigma Mode'}</span>
                </div>
            </div>
            
            {/* Theme Toggle */}
            <button 
                onClick={toggleDarkMode}
                className="p-2 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-white/5 transition-colors"
                title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
                {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
