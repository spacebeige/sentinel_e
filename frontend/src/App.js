import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { ChevronDown, Wifi, WifiOff } from 'lucide-react';
import Sidebar from './components/Sidebar';
import InputArea from './components/InputArea';
import ChatThread from './components/ChatThread';
import ResponseViewer from './components/ResponseViewer';
import LearningDashboard from './components/LearningDashboard';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [mode, setMode] = useState('standard');
  const [experimentalMode, setExperimentalMode] = useState('conversational');
  const [rounds, setRounds] = useState(6);
  const [killSwitch, setKillSwitch] = useState(false);
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);      // full conversation thread
  const [activeChatId, setActiveChatId] = useState(null); // persistent session ID
  const [currentResult, setCurrentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showLearning, setShowLearning] = useState(false);
  const [serverStatus, setServerStatus] = useState('unknown');

  // Added for ngrok deployment
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000'; // updated line for local development

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.classList.add('light');
    }
  }, [darkMode]);

  const checkHealth = useCallback(async () => {
    try {
      await axios.get(`${API_BASE_URL}/`, { timeout: 3000 });
      setServerStatus('online');
    } catch {
      setServerStatus('offline');
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/history`);
        const formatted = res.data.map(item => ({
          id: item.id,
          timestamp: item.updated_at || item.created_at || new Date().toISOString(),
          mode: item.mode,
          summary: item.chat_name || item.preview || 'Chat',
          filename: item.id,
          data: null,
        }));
        setHistory(formatted);
      } catch (err) {
        console.error('Failed to load history:', err);
      }
    };
    fetchHistory();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSend = async ({ text, file }) => {
    if (!text && !file) return;
    const chatId = activeChatId; // capture before any state reset
    setLoading(true);
    setShowLearning(false);

    // Optimistically add user message to thread immediately
    const userMsg = { role: 'user', content: text || `[File: ${file?.name}]`, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);

    const formData = new FormData();
    if (text) formData.append('text', text);
    if (file) formData.append('file', file);
    if (chatId && UUID_REGEX.test(chatId)) {
      formData.append('chat_id', chatId);
    }

    const endpoint = mode === 'standard'
      ? `${API_BASE_URL}/run/standard`
      : `${API_BASE_URL}/run/experimental`;

    try {
      if (mode === 'experimental') {
        formData.append('mode', experimentalMode);
        formData.append('rounds', rounds);
        formData.append('kill_switch', killSwitch);
      }

      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const result = response.data;
      const returnedChatId = result.chat_id ? String(result.chat_id) : null;
      const answerText = result.data?.priority_answer || result.data?.human_layer || result.priority_answer || 'No response.';

      // Add assistant reply to thread
      setMessages(prev => [...prev, { role: 'assistant', content: answerText, timestamp: new Date().toISOString() }]);
      setCurrentResult(result);

      // Persist session ID for next message (key fix for new-chat-each-time bug)
      if (returnedChatId && UUID_REGEX.test(returnedChatId)) {
        setActiveChatId(returnedChatId);
      }

      // Update sidebar history entry
      const effectiveChatId = chatId || returnedChatId;
      if (effectiveChatId && UUID_REGEX.test(effectiveChatId)) {
        setHistory(prev => {
          const exists = prev.some(item => item.id === effectiveChatId);
          if (exists) {
            return prev.map(item =>
              item.id === effectiveChatId
                ? { ...item, timestamp: new Date().toISOString(), summary: text ? text.substring(0, 40) : item.summary, data: result }
                : item
            );
          }
          return [{ id: effectiveChatId, timestamp: new Date().toISOString(), mode, summary: text ? text.substring(0, 40) : 'Chat', filename: effectiveChatId, data: result }, ...prev];
        });
      }

      setServerStatus('online');
    } catch (error) {
      console.error(error);
      setServerStatus('offline');
      setMessages(prev => prev.slice(0, -1)); // remove optimistic user message
      alert('Error: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleSelectRun = async (run) => {
    setShowLearning(false);
    // Restore UI mode from logged run mode
    if (run.mode === 'standard') {
      setMode('standard');
    } else if (['conversational', 'forensic', 'experimental'].includes(run.mode)) {
      setMode('experimental');
      setExperimentalMode(run.mode);
    }
    setActiveChatId(run.id);
    setMessages([]);
    setCurrentResult(null);
    setLoading(true);
    try {
      // Load full message history for this chat
      const res = await axios.get(`${API_BASE_URL}/api/chat/${run.id}/messages`);
      const loaded = (res.data || [])
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role, content: m.content, timestamp: m.timestamp || run.timestamp }));
      setMessages(loaded);
    } catch (err) {
      console.error('Failed to load messages:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setCurrentResult(null);
    setShowLearning(false);
  };

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-anthropic-bg text-slate-900 dark:text-slate-200 font-sans overflow-hidden antialiased selection:bg-poly-accent/30 selection:text-white transition-colors duration-200">
      
      <Sidebar
        history={history}
        onSelectRun={handleSelectRun}
        onNewChat={handleNewChat}
        activeMode={mode}
        activeChatId={activeChatId}
        showLearning={showLearning}
        onToggleLearning={() => setShowLearning(true)}
        darkMode={darkMode}
        toggleDarkMode={() => setDarkMode(d => !d)}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative w-full h-full min-w-0">
        
        {!showLearning && (
        <header className="absolute top-0 left-0 right-0 z-20 px-8 py-4 bg-gradient-to-b from-white dark:from-anthropic-bg via-white/80 dark:via-anthropic-bg/80 to-transparent pointer-events-none flex flex-col items-center justify-center space-y-2">
            <div className="pointer-events-auto flex items-center space-x-3">
                <div className={`flex items-center space-x-1.5 text-[10px] font-mono px-2 py-1 rounded-full border transition-colors ${
                  serverStatus === 'online' ? 'text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/20'
                  : serverStatus === 'offline' ? 'text-red-500 bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/20'
                  : 'text-slate-400 bg-slate-100 dark:bg-white/5 border-slate-200 dark:border-white/10'
                }`}>
                  {serverStatus === 'online' ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
                  <span>{serverStatus === 'online' ? 'Connected' : serverStatus === 'offline' ? 'Disconnected' : '...'}</span>
                </div>
                <button
                    className="flex items-center space-x-2 text-sm font-medium text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-white/5 py-1.5 px-3 rounded-lg transition-all border border-transparent hover:border-slate-300 dark:hover:border-white/5"
                    onClick={() => { setMode(m => m === 'standard' ? 'experimental' : 'standard'); handleNewChat(); }}
                >
                    <span className={mode === 'experimental' ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400'}>
                        Sentinel {mode === 'standard' ? 'v2' : 'Sigma'}
                    </span>
                    <ChevronDown className="w-3 h-3 opacity-50" />
                </button>
            </div>
            
            {/* Sub-mode Selector for Experimental */}
            {mode === 'experimental' && (
                <div className="flex flex-col items-center space-y-2 pointer-events-auto">
                    {/* Mode Toggle */}
                    <div className="flex items-center bg-slate-200/80 dark:bg-black/40 rounded-full p-1 border border-slate-300/60 dark:border-white/5 backdrop-blur-md">
                        {['conversational', 'forensic', 'experimental'].map(m => (
                            <button
                                key={m}
                                onClick={() => setExperimentalMode(m)}
                                className={`px-3 py-1 text-[10px] uppercase font-bold tracking-wider rounded-full transition-all ${
                                    experimentalMode === m
                                    ? 'bg-amber-500/20 text-amber-700 dark:text-amber-300 border border-amber-500/40'
                                    : 'text-slate-500 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
                                }`}
                            >
                                {m}
                            </button>
                        ))}
                    </div>

                    {/* V4 Parameters */}
                    <div className="flex items-center space-x-4">
                        {/* Rounds (only for experimental mode) */}
                        {experimentalMode === 'experimental' && (
                            <div className="flex items-center space-x-2 bg-slate-200/80 dark:bg-black/40 rounded-full px-3 py-1 border border-slate-300/60 dark:border-white/5">
                                <span className="text-[10px] font-mono text-slate-600 dark:text-slate-500">ROUNDS:</span>
                                <input 
                                    type="number" 
                                    min="1" max="10" 
                                    value={rounds} 
                                    onChange={(e) => setRounds(parseInt(e.target.value) || 1)}
                                    className="w-8 bg-transparent text-center text-xs font-mono text-slate-900 dark:text-white focus:outline-none border-b border-transparent focus:border-amber-500/50"
                                />
                            </div>
                        )}

                        {/* Kill Switch (only for forensic mode) */}
                        {experimentalMode === 'forensic' && (
                            <button 
                                onClick={() => setKillSwitch(!killSwitch)}
                                className={`flex items-center space-x-2 px-3 py-1 rounded-full text-[10px] font-mono border transition-all ${
                                    killSwitch
                                    ? 'bg-red-500/20 border-red-500/50 text-red-600 dark:text-red-400'
                                    : 'bg-slate-200/80 dark:bg-black/40 border-slate-300/60 dark:border-slate-800 text-slate-600 dark:text-slate-500 hover:text-slate-900 dark:hover:text-slate-300'
                                }`}
                            >
                                <span className="w-2 h-2 rounded-full bg-current animate-pulse"></span>
                                <span>KILL_SWITCH: {killSwitch ? 'ON' : 'OFF'}</span>
                            </button>
                        )}
                    </div>
                </div>
            )}
        </header>
        )}

        {/* View Switching */}
        {showLearning ? (
            <div className="flex-1 flex flex-col w-full h-full pt-4">
                <LearningDashboard />
            </div>
        ) : (
            <>
                {/* Chat Thread - shows full conversation history */}
                <div className="flex-1 overflow-y-auto w-full h-full scroll-smooth pt-24 pb-36 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 scrollbar-track-transparent">
                    <ChatThread messages={messages} loading={loading} mode={mode} />
                    {/* Machine layer panel for experimental mode */}
                    {mode === 'experimental' && currentResult && !loading && (
                      <div className="w-full max-w-3xl mx-auto px-4 md:px-6 mt-2 mb-4">
                        <ResponseViewer data={currentResult} mode={mode} onFeedback={() => {}} />
                      </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="absolute bottom-0 left-0 right-0 z-30 p-4 md:p-6 bg-gradient-to-t from-slate-50 via-slate-50/95 dark:from-anthropic-bg dark:via-anthropic-bg/95 to-transparent">
                  <div className="max-w-3xl mx-auto w-full">
                    <InputArea onSend={handleSend} loading={loading} mode={mode} />
                    <div className="mt-2 text-center">
                      <p className="text-[10px] text-slate-400 dark:text-slate-600 font-mono tracking-tight">
                        {mode === 'standard' ? 'SENTINEL-E v2.1.0 · STANDARD' : `SENTINEL-Σ · ${experimentalMode.toUpperCase()}`}
                        {activeChatId ? <span className="ml-2 text-emerald-500/60">● session active</span> : null}
                      </p>
                    </div>
                  </div>
                </div>
            </>
        )}
      </main>
    </div>
  );
}

export default App;
