/**
 * AdvancedCopyMenu — Multi-format clipboard export for assistant messages
 *
 * Supports:  Plain Text · Markdown · Code Blocks · With Citations
 * Reusable across Standard, Experimental (Debate), and Individual model modes.
 *
 * CI-safe: all hooks at top, no conditional hooks, no unstable deps.
 */
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Copy, Check, ChevronDown, FileText, Code2, BookOpen, AlertCircle } from 'lucide-react';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

/** Duration (ms) for the "Copied ✓" confirmation state */
const CONFIRMATION_MS = 1500;

/**
 * @param {{ message: object, className?: string }} props
 *   message.content         — rendered text / markdown
 *   message.citations       — optional citation array
 *   message.raw_output      — optional raw markdown source
 *   message.refined_output  — optional refined text
 */
export default function AdvancedCopyMenu({ message, className = '' }) {
  // ── All hooks at the very top — CI-safe ordering ──────────────
  const [open, setOpen] = useState(false);
  const [copiedType, setCopiedType] = useState(null);
  const [error, setError] = useState(null);
  const menuRef = useRef(null);
  const timerRef = useRef(null);

  const safeContent = useMemo(
    () => message?.content || message?.refined_output || message?.raw_output || '',
    [message?.content, message?.refined_output, message?.raw_output],
  );

  const safeCitations = useMemo(
    () => message?.citations || [],
    [message?.citations],
  );

  const codeBlocks = useMemo(() => {
    const matches = safeContent.match(/```[\s\S]*?```/g) || [];
    return matches.map(block =>
      block.replace(/^```\w*\n?/, '').replace(/\n?```$/, ''),
    );
  }, [safeContent]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // ── Copy handler (stable ref via useCallback) ─────────────────
  const handleCopy = useCallback(
    async (type) => {
      let text = '';

      switch (type) {
        case 'text':
          text = safeContent;
          break;

        case 'markdown':
          // Prefer raw markdown source when available
          text = message?.raw_output || safeContent;
          break;

        case 'code':
          if (codeBlocks.length === 0) {
            setError('No code blocks found');
            if (timerRef.current) clearTimeout(timerRef.current);
            timerRef.current = setTimeout(() => setError(null), CONFIRMATION_MS);
            setOpen(false);
            return;
          }
          text = codeBlocks.join('\n\n');
          break;

        case 'citations': {
          text = safeContent;
          if (safeCitations.length > 0) {
            const footer = safeCitations
              .map((c, i) => `[${i + 1}] ${typeof c === 'string' ? c : c.url || c.source || JSON.stringify(c)}`)
              .join('\n');
            text += `\n\n---\nSources:\n${footer}`;
          }
          break;
        }

        default:
          text = safeContent;
      }

      try {
        await navigator.clipboard.writeText(text);
        setCopiedType(type);
        setError(null);
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => setCopiedType(null), CONFIRMATION_MS);
      } catch {
        setError('Clipboard access denied');
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => setError(null), CONFIRMATION_MS);
      }

      setOpen(false);
    },
    [safeContent, codeBlocks, safeCitations, message?.raw_output],
  );

  // ── Menu items ────────────────────────────────────────────────
  const items = useMemo(
    () => [
      { key: 'text', label: 'Copy Text', icon: FileText },
      { key: 'markdown', label: 'Copy Markdown', icon: Copy },
      { key: 'code', label: 'Copy Code Blocks', icon: Code2 },
      { key: 'citations', label: 'Copy With Citations', icon: BookOpen },
    ],
    [],
  );

  // ── Render ────────────────────────────────────────────────────
  return (
    <div ref={menuRef} className={`relative inline-flex ${className}`}>
      {/* Trigger button */}
      <button
        onClick={() => setOpen(prev => !prev)}
        className="flex items-center gap-0.5 p-1 rounded-md transition-colors
                   hover:bg-black/5 dark:hover:bg-white/10 text-[#aeaeb2] hover:text-[#6e6e73] dark:hover:text-white"
        aria-label="Copy options"
        style={{ fontFamily: FONT }}
      >
        {copiedType ? (
          <span className="flex items-center gap-1 text-[#34c759] transition-opacity duration-200"
            style={{ fontSize: '11px', fontWeight: 600, fontFamily: FONT }}>
            <Check className="w-3 h-3" />
            Copied
          </span>
        ) : error ? (
          <span className="flex items-center gap-1 text-[#ef4444] transition-opacity duration-200"
            style={{ fontSize: '11px', fontWeight: 500, fontFamily: FONT }}>
            <AlertCircle className="w-3 h-3" />
            {error}
          </span>
        ) : (
          <>
            <Copy className="w-3.5 h-3.5" />
            <ChevronDown className="w-2.5 h-2.5" />
          </>
        )}
      </button>

      {/* Dropdown menu — absolute, no layout shift */}
      {open && (
        <div
          className="absolute right-0 top-full mt-1 z-50 min-w-[180px] py-1
                     rounded-xl shadow-lg shadow-black/10
                     bg-white dark:bg-[#1c1c1e]
                     border border-black/5 dark:border-white/10
                     backdrop-blur-xl"
          style={{ fontFamily: FONT }}
        >
          {items.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => handleCopy(key)}
              className="w-full flex items-center gap-2.5 px-3 py-2 text-left
                         text-[#1d1d1f] dark:text-white
                         hover:bg-[#f5f5f7] dark:hover:bg-white/10
                         transition-colors"
              style={{ fontSize: '13px', fontWeight: 500 }}
            >
              <Icon className="w-3.5 h-3.5 text-[#6e6e73] dark:text-[#94a3b8] flex-shrink-0" />
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
