import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ShieldAlert, User, Copy, Check, Zap, Brain } from 'lucide-react';

/* ─── Prose styles injected once into <head> ────────────────────── */
const PROSE_STYLES = `
  .ct-prose { font-size: 0.875rem; line-height: 1.7; color: #0f172a; }
  .dark .ct-prose { color: #e2e8f0; }
  .ct-prose p { margin: 0 0 0.75em; }
  .ct-prose p:last-child { margin-bottom: 0; }
  .ct-prose h1,.ct-prose h2,.ct-prose h3 { font-weight: 700; margin: 1.1em 0 0.4em; color: #0f172a; }
  .dark .ct-prose h1,.dark .ct-prose h2,.dark .ct-prose h3 { color: #f1f5f9; }
  .ct-prose h1 { font-size: 1.2em; } .ct-prose h2 { font-size: 1.05em; } .ct-prose h3 { font-size: 0.95em; }
  .ct-prose ul,.ct-prose ol { padding-left: 1.4em; margin: 0.5em 0 0.75em; }
  .ct-prose li { margin-bottom: 0.25em; }
  .ct-prose code { background: rgba(0,0,0,0.06); color: #047857; padding: 0.15em 0.45em; border-radius: 0.3rem; font-size: 0.8em; }
  .dark .ct-prose code { background: rgba(16,185,129,0.12); color: #34d399; }
  .ct-prose pre { background: #f1f5f9; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 0.6rem; overflow-x: auto; margin: 0.75em 0; }
  .dark .ct-prose pre { background: rgba(0,0,0,0.45); border-color: rgba(255,255,255,0.06); }
  .ct-prose pre code { background: none; color: inherit; padding: 0; font-size: 0.82em; }
  .ct-prose a { color: #0284c7; text-decoration: underline; text-underline-offset: 2px; }
  .dark .ct-prose a { color: #38bdf8; }
  .ct-prose strong { font-weight: 700; color: #0f172a; }
  .dark .ct-prose strong { color: #fde68a; }
  .ct-prose em { color: #475569; } .dark .ct-prose em { color: #94a3b8; }
  .ct-prose blockquote { border-left: 3px solid #0ea5e9; padding-left: 1em; margin: 0.75em 0; color: #475569; font-style: italic; }
  .dark .ct-prose blockquote { border-color: #38bdf8; color: #94a3b8; }
  .ct-prose table { border-collapse: collapse; width: 100%; margin: 0.75em 0; font-size: 0.82em; }
  .ct-prose th { background: #f8fafc; font-weight: 600; color: #374151; border: 1px solid #e2e8f0; padding: 0.5rem 0.75rem; text-align: left; }
  .dark .ct-prose th { background: rgba(255,255,255,0.05); color: #cbd5e1; border-color: rgba(255,255,255,0.08); }
  .ct-prose td { border: 1px solid #e2e8f0; padding: 0.45rem 0.75rem; }
  .dark .ct-prose td { border-color: rgba(255,255,255,0.07); }
  .ct-prose hr { border: none; border-top: 1px solid #e2e8f0; margin: 1em 0; }
  .dark .ct-prose hr { border-color: rgba(255,255,255,0.08); }
`;
let _stylesInjected = false;
function injectStyles() {
  if (_stylesInjected || typeof document === 'undefined') return;
  const el = document.createElement('style');
  el.textContent = PROSE_STYLES;
  document.head.appendChild(el);
  _stylesInjected = true;
}

/* ─── Typing indicator ──────────────────────────────────────────── */
const TypingIndicator = ({ mode }) => (
  <div className="flex items-end space-x-3 mb-4 px-2">
    <AvatarAI mode={mode} size="sm" />
    <div className="flex items-center space-x-1.5 bg-white dark:bg-slate-800/80 border border-slate-200/80 dark:border-slate-700/40 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
      {[0, 150, 300].map(d => (
        <span key={d} className="w-2 h-2 rounded-full bg-slate-400 dark:bg-slate-500 animate-bounce inline-block" style={{ animationDelay: `${d}ms` }} />
      ))}
    </div>
  </div>
);

/* ─── Avatars ───────────────────────────────────────────────────── */
const AvatarAI = ({ mode, size = 'md' }) => {
  const s = size === 'sm' ? 'w-7 h-7' : 'w-8 h-8';
  const ic = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4';
  const isExp = mode === 'experimental';
  return (
    <div className={`${s} rounded-full flex-shrink-0 flex items-center justify-center ring-1 ${
      isExp ? 'bg-amber-500 ring-amber-400/40' : 'bg-emerald-600 ring-emerald-500/30'
    }`}>
      {isExp ? <Brain className={`${ic} text-white`} /> : <ShieldAlert className={`${ic} text-white`} />}
    </div>
  );
};

const AvatarUser = () => (
  <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-slate-200 dark:bg-slate-700 ring-1 ring-slate-300/60 dark:ring-slate-600/40">
    <User className="w-4 h-4 text-slate-500 dark:text-slate-400" />
  </div>
);

/* ─── User bubble ───────────────────────────────────────────────── */
const UserBubble = ({ message }) => (
  <div className="flex items-end justify-end space-x-2.5 mb-5 group">
    <div className="max-w-[72%] flex flex-col items-end">
      <div className="bg-slate-900 dark:bg-slate-700 text-white rounded-2xl rounded-br-sm px-4 py-3 shadow-sm">
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
      </div>
      <span className="text-[10px] text-slate-400 dark:text-slate-600 mt-1 pr-0.5">
        {formatTime(message.timestamp)}
      </span>
    </div>
    <AvatarUser />
  </div>
);

/* ─── Assistant bubble ──────────────────────────────────────────── */
const AssistantBubble = ({ message, mode }) => {
  injectStyles();
  const [copied, setCopied] = React.useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="flex items-end space-x-2.5 mb-5 group">
      <AvatarAI mode={mode} />
      <div className="max-w-[80%] min-w-0 flex flex-col">
        <span className="text-[10px] font-semibold tracking-wide uppercase text-slate-400 dark:text-slate-500 mb-1 pl-0.5">
          {mode === 'experimental' ? 'Sigma' : 'Sentinel'}
        </span>
        <div className="bg-white dark:bg-slate-800/70 border border-slate-200/80 dark:border-slate-700/40 rounded-2xl rounded-bl-sm px-5 py-4 shadow-sm">
          <div className="ct-prose">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        </div>
        <div className="flex items-center justify-between mt-1 px-0.5">
          <span className="text-[10px] text-slate-400 dark:text-slate-600">
            {formatTime(message.timestamp)}
          </span>
          <button
            onClick={handleCopy}
            title="Copy response"
            className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-all duration-150 py-0.5 px-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-700/60"
          >
            {copied
              ? <><Check className="w-3 h-3 text-emerald-500" /><span className="text-[10px] text-emerald-500 font-medium">Copied</span></>
              : <><Copy className="w-3 h-3 text-slate-400" /><span className="text-[10px] text-slate-400">Copy</span></>
            }
          </button>
        </div>
      </div>
    </div>
  );
};

/* ─── Helpers ───────────────────────────────────────────────────── */
function formatTime(ts) {
  if (!ts) return '';
  try { return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
  catch { return ''; }
}

function dayLabel(ts) {
  if (!ts) return null;
  const d = new Date(ts);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

/* ─── Date separator ────────────────────────────────────────────── */
const DateSeparator = ({ label }) => (
  <div className="flex items-center my-5 px-2">
    <div className="flex-1 h-px bg-slate-200 dark:bg-slate-700/60" />
    <span className="mx-3 text-[10px] font-medium uppercase tracking-wider text-slate-400 dark:text-slate-600 whitespace-nowrap">{label}</span>
    <div className="flex-1 h-px bg-slate-200 dark:bg-slate-700/60" />
  </div>
);

/* ─── Empty state ───────────────────────────────────────────────── */
const HINTS = {
  standard: [
    { icon: '💡', text: 'Explain quantum entanglement simply' },
    { icon: '🐍', text: 'Write a Python function to parse JSON safely' },
    { icon: '⚖️', text: 'Compare microservices vs monolith architecture' },
    { icon: '📊', text: 'Summarize the latest AI trends' },
  ],
  experimental: [
    { icon: '🔍', text: 'Forensic: Is this argument logically valid?' },
    { icon: '⚔️', text: 'Debate: AGI — optimistic vs pessimistic' },
    { icon: '🛡️', text: 'Shadow check: evaluate this claim for bias' },
    { icon: '📈', text: 'Risk analysis: this investment strategy' },
  ],
};

const EmptyState = ({ mode }) => {
  const isExp = mode === 'experimental';
  const hints = isExp ? HINTS.experimental : HINTS.standard;
  return (
    <div className="flex flex-col items-center justify-center min-h-[62vh] px-6 select-none">
      <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-5 shadow-lg ${
        isExp ? 'bg-amber-500 shadow-amber-500/25' : 'bg-emerald-600 shadow-emerald-600/25'
      }`}>
        {isExp ? <Brain className="w-7 h-7 text-white" /> : <ShieldAlert className="w-7 h-7 text-white" />}
      </div>
      <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-1 tracking-tight">
        {isExp ? 'Sentinel-Σ' : 'Sentinel-E'}
      </h2>
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-xs text-center leading-relaxed mb-8">
        {isExp
          ? 'Multi-model forensic reasoning, debate rounds, and shadow safety evaluation.'
          : 'Structured, intelligent responses powered by multi-model AI.'}
      </p>
      <div className="grid grid-cols-2 gap-2.5 w-full max-w-lg">
        {hints.map(h => (
          <div key={h.text} className="group flex items-start space-x-2.5 p-3 rounded-xl bg-white dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/50 hover:border-emerald-400/50 dark:hover:border-emerald-500/30 hover:shadow-sm transition-all cursor-default">
            <span className="text-base leading-none mt-0.5 flex-shrink-0">{h.icon}</span>
            <p className="text-xs text-slate-500 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-300 transition-colors leading-snug">{h.text}</p>
          </div>
        ))}
      </div>
      <div className="mt-10 flex items-center space-x-1.5 text-[10px] font-mono text-slate-400 dark:text-slate-600 uppercase tracking-widest">
        <Zap className="w-3 h-3" />
        <span>{isExp ? 'Sigma Engine · Multi-Model' : 'Sentinel Engine · Standard'}</span>
      </div>
    </div>
  );
};

/* ─── Main component ────────────────────────────────────────────── */
const ChatThread = ({ messages, loading, mode }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  if (!messages || messages.length === 0) {
    return loading
      ? <div className="px-4 md:px-0 pt-4"><TypingIndicator mode={mode} /></div>
      : <EmptyState mode={mode} />;
  }

  // Insert date separators between messages from different days
  const items = [];
  let lastDay = null;
  messages.forEach((msg, idx) => {
    const day = dayLabel(msg.timestamp);
    if (day && day !== lastDay) {
      items.push({ type: 'date', label: day, key: `sep-${idx}` });
      lastDay = day;
    }
    items.push({ type: 'msg', msg, key: idx });
  });

  return (
    <div className="w-full max-w-3xl mx-auto px-4 md:px-2 pt-2">
      {items.map(item =>
        item.type === 'date'
          ? <DateSeparator key={item.key} label={item.label} />
          : item.msg.role === 'user'
            ? <UserBubble key={item.key} message={item.msg} />
            : <AssistantBubble key={item.key} message={item.msg} mode={mode} />
      )}
      {loading && <TypingIndicator mode={mode} />}
      <div ref={bottomRef} className="h-4" />
    </div>
  );
};

export default ChatThread;
