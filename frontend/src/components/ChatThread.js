import React, { useEffect, useRef, useState } from 'react';
import { Copy, Check, Zap, Brain, ShieldAlert } from 'lucide-react';
import FeedbackButton from './FeedbackButton';
import { normalizeResponse, isCodeResponse } from '../engines/responseNormalizer';

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

/* ─── Typing indicator ──────────────────────────────────────────── */
const TypingIndicator = () => (
  <div className="flex items-start gap-3 mb-4 px-1">
    <div className="w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center"
      style={{ backgroundColor: 'var(--accent-green)', opacity: 0.8 }}>
      <ShieldAlert className="w-3.5 h-3.5 text-white" />
    </div>
    <div className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-tl-sm"
      style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-secondary)' }}>
      {[0, 150, 300].map(d => (
        <span key={d} className="w-1.5 h-1.5 rounded-full animate-bounce inline-block"
          style={{ backgroundColor: 'var(--text-tertiary)', animationDelay: `${d}ms` }} />
      ))}
    </div>
  </div>
);

/* ─── User message ──────────────────────────────────────────────── */
const UserBubble = ({ message }) => (
  <div className="flex justify-end mb-5">
    <div className="max-w-[72%] flex flex-col items-end">
      <div className="px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed whitespace-pre-wrap"
        style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}>
        {message.content}
      </div>
      <span className="text-[10px] mt-1 pr-1" style={{ color: 'var(--text-tertiary)' }}>
        {formatTime(message.timestamp)}
      </span>
    </div>
  </div>
);

/* ─── Assistant message ─────────────────────────────────────────── */
const AssistantBubble = ({ message, mode, subMode }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const labelText = mode === 'experimental' ? `Omega · ${subMode || 'debate'}` : 'Sentinel';

  return (
    <div className="flex items-start gap-3 mb-5 group">
      <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center mt-0.5"
        style={{ backgroundColor: mode === 'experimental' ? 'var(--accent-orange)' : 'var(--accent-green)' }}>
        {mode === 'experimental'
          ? <Brain className="w-4 h-4 text-white" />
          : <ShieldAlert className="w-4 h-4 text-white" />
        }
      </div>
      <div className="flex-1 min-w-0 max-w-[85%]">
        <span className="text-[10px] font-semibold tracking-wide uppercase mb-1 block pl-0.5"
          style={{ color: 'var(--text-tertiary)' }}>
          {labelText}
        </span>
        <div className="card px-5 py-4 rounded-2xl rounded-tl-sm">
          <div className="text-sm leading-relaxed">
            {isCodeResponse(message.content) ? (
              <pre className="bg-[#f5f5f7] rounded-lg p-3 overflow-x-auto text-xs font-mono text-[#1d1d1f] whitespace-pre-wrap">
                <code>{message.content.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '')}</code>
              </pre>
            ) : (
              normalizeResponse(message.content).split('\n\n').filter(Boolean).map((para, i) => (
                <p key={i} className="mb-2 last:mb-0" style={{ color: 'var(--text-primary)' }}>
                  {para}
                </p>
              ))
            )}
          </div>
        </div>
        <div className="flex items-center justify-between mt-1.5 px-0.5">
          <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
            {formatTime(message.timestamp)}
          </span>
          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button onClick={handleCopy} title="Copy"
              className="flex items-center gap-1 py-0.5 px-1.5 rounded-md transition-colors"
              style={{ color: copied ? 'var(--accent-green)' : 'var(--text-tertiary)' }}>
              {copied
                ? <><Check className="w-3 h-3" /><span className="text-[10px] font-medium">Copied</span></>
                : <><Copy className="w-3 h-3" /><span className="text-[10px]">Copy</span></>
              }
            </button>
            {message.runId && (
              <FeedbackButton runId={message.runId} mode={mode} subMode={subMode} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

/* ─── Date separator ────────────────────────────────────────────── */
const DateSeparator = ({ label }) => (
  <div className="flex items-center my-5 px-2">
    <div className="flex-1 h-px" style={{ backgroundColor: 'var(--border-secondary)' }} />
    <span className="mx-3 text-[10px] font-medium uppercase tracking-wider whitespace-nowrap"
      style={{ color: 'var(--text-tertiary)' }}>{label}</span>
    <div className="flex-1 h-px" style={{ backgroundColor: 'var(--border-secondary)' }} />
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
    { icon: '⚔️', text: 'Multi-model debate: Is AGI near?' },
    { icon: '🔬', text: 'Glass analysis: Evaluate this claim' },
    { icon: '📚', text: 'Evidence check: cite sources for this' },
    { icon: '📈', text: 'Risk analysis: assess this strategy' },
  ],
};

const EmptyState = ({ mode, subMode }) => {
  const isExp = mode === 'experimental';
  const hints = isExp ? HINTS.experimental : HINTS.standard;
  return (
    <div className="flex flex-col items-center justify-center min-h-[62vh] px-6 select-none">
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
        style={{
          backgroundColor: isExp ? 'var(--accent-orange)' : 'var(--accent-green)',
          boxShadow: `0 8px 24px ${isExp ? 'rgba(249,115,22,0.25)' : 'rgba(34,197,94,0.25)'}`,
        }}>
        {isExp ? <Brain className="w-7 h-7 text-white" /> : <ShieldAlert className="w-7 h-7 text-white" />}
      </div>
      <h2 className="text-2xl font-bold mb-1 tracking-tight" style={{ color: 'var(--text-primary)' }}>
        {isExp ? `Sentinel-E · ${subMode || 'Experimental'}` : 'Sentinel-E'}
      </h2>
      <p className="text-sm max-w-xs text-center leading-relaxed mb-8" style={{ color: 'var(--text-secondary)' }}>
        {isExp
          ? 'Multi-model adversarial reasoning with debate rounds, glass transparency, and evidence verification.'
          : 'Structured, intelligent responses powered by multi-model AI.'}
      </p>
      <div className="grid grid-cols-2 gap-2.5 w-full max-w-lg">
        {hints.map(h => (
          <div key={h.text} className="card group flex items-start gap-2.5 p-3 cursor-default hover:scale-[1.01] transition-transform">
            <span className="text-base leading-none mt-0.5 flex-shrink-0">{h.icon}</span>
            <p className="text-xs leading-snug" style={{ color: 'var(--text-secondary)' }}>{h.text}</p>
          </div>
        ))}
      </div>
      <div className="mt-10 flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-widest"
        style={{ color: 'var(--text-tertiary)' }}>
        <Zap className="w-3 h-3" />
        <span>Sentinel-E · {isExp ? 'Multi-Model' : 'Standard'}</span>
      </div>
    </div>
  );
};

/* ─── Main component ────────────────────────────────────────────── */
const ChatThread = ({ messages, loading, mode, subMode }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  if (!messages || messages.length === 0) {
    return loading
      ? <div className="pt-4"><TypingIndicator /></div>
      : <EmptyState mode={mode} subMode={subMode} />;
  }

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
    <div className="w-full pt-2">
      {items.map(item =>
        item.type === 'date'
          ? <DateSeparator key={item.key} label={item.label} />
          : item.msg.role === 'user'
            ? <UserBubble key={item.key} message={item.msg} />
            : <AssistantBubble key={item.key} message={item.msg} mode={mode} subMode={subMode} />
      )}
      {loading && <TypingIndicator />}
      <div ref={bottomRef} className="h-4" />
    </div>
  );
};

export default ChatThread;
