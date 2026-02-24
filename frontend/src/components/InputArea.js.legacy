import React, { useState, useRef } from 'react';
import { Paperclip, X, CornerDownLeft } from 'lucide-react';

const InputArea = ({ onSend, loading, mode, subMode }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [isFocused, setIsFocused] = useState(false);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const handleSend = () => {
    if (!text.trim() && !file) return;
    onSend({ text: text.trim(), file });
    setText('');
    setFile(null);
    if (textareaRef.current) textareaRef.current.style.height = '52px';
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const autoResize = (e) => {
    setText(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const modeLabel = mode === 'experimental' ? `Omega · ${subMode || 'debate'}` : 'Sentinel';

  return (
    <div className="w-full">
      <div className="rounded-2xl transition-all" style={{
        backgroundColor: 'var(--bg-input)',
        border: isFocused ? '1px solid var(--border-focus)' : '1px solid var(--border-primary)',
        boxShadow: isFocused ? 'var(--shadow-md)' : 'var(--shadow-sm)',
      }}>
        {/* File preview */}
        {file && (
          <div className="px-4 pt-3 flex items-center">
            <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
              style={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-secondary)', color: 'var(--text-primary)' }}>
              <Paperclip className="w-3 h-3" style={{ color: 'var(--accent-blue)' }} />
              <span className="truncate max-w-[200px]">{file.name}</span>
              <button onClick={() => setFile(null)} className="ml-1 transition-colors" style={{ color: 'var(--accent-red)' }}>
                <X className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={autoResize}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyPress}
          placeholder={`Message ${modeLabel}...`}
          className="w-full bg-transparent focus:outline-none resize-none px-4 py-3.5 text-sm leading-relaxed scrollbar-thin"
          style={{ color: 'var(--text-primary)', minHeight: '52px', maxHeight: '200px' }}
          rows={1}
        />

        {/* Toolbar */}
        <div className="flex items-center justify-between px-2 pb-2">
          <div className="flex items-center gap-1">
            <button onClick={() => fileInputRef.current?.click()}
              className="p-2 rounded-lg transition-colors"
              style={{ color: 'var(--text-tertiary)' }}
              onMouseEnter={e => e.target.style.color = 'var(--text-primary)'}
              onMouseLeave={e => e.target.style.color = 'var(--text-tertiary)'}>
              <Paperclip className="w-5 h-5" />
            </button>
            <input type="file" ref={fileInputRef} onChange={e => e.target.files[0] && setFile(e.target.files[0])} className="hidden" />
            
            <span className="text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded"
              style={{ color: 'var(--text-tertiary)', backgroundColor: 'var(--bg-tertiary)' }}>
              {modeLabel}
            </span>
          </div>

          <button onClick={handleSend} disabled={loading || (!text.trim() && !file)}
            className="p-2 rounded-lg transition-all flex items-center justify-center"
            style={{
              backgroundColor: (text.trim() || file) ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              color: (text.trim() || file) ? '#fff' : 'var(--text-tertiary)',
              cursor: loading || (!text.trim() && !file) ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}>
            {loading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <CornerDownLeft className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
      
      <div className="text-center mt-2">
        <p className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
          Sentinel-E may produce inaccurate results. Verify critical information independently.
        </p>
      </div>
    </div>
  );
};

export default InputArea;
