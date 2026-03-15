import React, { useEffect, useRef, useState } from 'react';
import { Copy, Check, Zap, Brain, ShieldAlert, Pencil, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import FeedbackButton from './FeedbackButton';
import { normalizeResponse, isCodeResponse } from '../engines/responseNormalizer';
import { editMessage, regenerateMessage } from '../services/api';

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
const UserBubble = ({ message, onMessageEdited }) => {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { console.warn('Clipboard access denied'); }
  };

  const handleSave = async () => {
    if (!draft.trim() || draft === message.content) { setEditing(false); return; }
    setSaving(true);
    try {
      await editMessage(message.id, draft);
      if (onMessageEdited) onMessageEdited(message.id, draft);
      setEditing(false);
    } catch { console.warn('Edit failed'); }
    setSaving(false);
  };

  return (
    <div className="flex justify-end mb-5 group">
      <div className="max-w-[72%] flex flex-col items-end">
        {message.image_b64 && message.image_mime !== 'application/pdf' && (
          <img
            src={`data:${message.image_mime || 'image/png'};base64,${message.image_b64}`}
            alt="Attached"
            className="max-w-full max-h-48 rounded-xl mb-2 object-contain"
          />
        )}
        {message.image_b64 && message.image_mime === 'application/pdf' && (
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-700/50 rounded-xl mb-2 text-sm">
            <span className="text-red-400 text-lg">📄</span>
            <span className="text-gray-200">{message.pdf_filename || 'Document.pdf'}</span>
            <span className="text-gray-400 text-xs">PDF attached</span>
          </div>
        )}
        {editing ? (
          <div className="w-full">
            <textarea
              className="w-full px-3 py-2 rounded-xl text-sm leading-relaxed resize-none border"
              style={{ borderColor: 'var(--border-secondary)', backgroundColor: 'var(--bg-card)', color: 'var(--text-primary)' }}
              value={draft} onChange={e => setDraft(e.target.value)} rows={3} autoFocus
            />
            <div className="flex gap-1.5 mt-1 justify-end">
              <button onClick={() => { setDraft(message.content); setEditing(false); }}
                className="text-[10px] px-2 py-0.5 rounded-md" style={{ color: 'var(--text-tertiary)' }}>Cancel</button>
              <button onClick={handleSave} disabled={saving}
                className="text-[10px] px-2 py-0.5 rounded-md font-medium" style={{ color: 'var(--accent-blue)' }}>
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        ) : (
          <div className="px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed whitespace-pre-wrap"
            style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}>
            {message.content}
          </div>
        )}
        <div className="flex items-center gap-2 mt-1 pr-1">
          <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
            {formatTime(message.timestamp)}
          </span>
          {!editing && (
            <>
              <button onClick={handleCopy} title="Copy"
                className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-0.5 py-0.5 px-1 rounded-md"
                style={{ color: copied ? 'var(--accent-green)' : 'var(--text-tertiary)' }}>
                {copied
                  ? <><Check className="w-3 h-3" /><span className="text-[10px]">Copied</span></>
                  : <><Copy className="w-3 h-3" /><span className="text-[10px]">Copy</span></>
                }
              </button>
              <button onClick={() => setEditing(true)} title="Edit"
                className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-0.5 py-0.5 px-1 rounded-md"
                style={{ color: 'var(--text-tertiary)' }}>
                <Pencil className="w-3 h-3" /><span className="text-[10px]">Edit</span>
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

/* ─── Assistant message ─────────────────────────────────────────── */
const AssistantBubble = ({ message, mode, subMode, onRegenerate }) => {
  const [copied, setCopied] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      console.warn('Clipboard access denied');
    }
  };

  const handleRegenerate = async () => {
    if (!message.id || regenerating) return;
    setRegenerating(true);
    try {
      await regenerateMessage(message.id);
      if (onRegenerate) onRegenerate(message.id);
    } catch { console.warn('Regenerate failed'); }
    setRegenerating(false);
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
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalizeResponse(message.content)}</ReactMarkdown>
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
            <button onClick={handleRegenerate} title="Regenerate" disabled={regenerating}
              className="flex items-center gap-1 py-0.5 px-1.5 rounded-md transition-colors"
              style={{ color: 'var(--text-tertiary)' }}>
              <RefreshCw className={`w-3 h-3 ${regenerating ? 'animate-spin' : ''}`} />
              <span className="text-[10px]">{regenerating ? 'Regenerating…' : 'Regenerate'}</span>
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
const ChatThread = ({ messages, loading, mode, subMode, onMessageEdited, onRegenerate }) => {
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
            ? <UserBubble key={item.key} message={item.msg} onMessageEdited={onMessageEdited} />
            : <AssistantBubble key={item.key} message={item.msg} mode={mode} subMode={subMode} onRegenerate={onRegenerate} />
      )}
      {loading && <TypingIndicator />}
      <div ref={bottomRef} className="h-4" />
    </div>
  );
};

export default ChatThread;
