import React, { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router';
import { Loader2 } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';
import { supabase } from '../lib/supabase';

// ============================================================
// AuthCallbackPage
//
// Single responsibility: handle the OAuth PKCE callback.
//
// Flow:
//   1. Read 'code' from URL (PKCE flow)
//   2. If code present → exchangeCodeForSession
//   3. Check for error params in URL
//   4. getSession() to confirm session established
//   5. session exists → navigate('/chat')
//   6. session missing → show error
//
// ALL diagnostic logs are preserved intentionally.
// They are required for OAuth debugging.
// ============================================================
export default function AuthCallbackPage() {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [callbackError, setCallbackError] = useState<string | null>(null);
  const [status, setStatus] = useState('Verifying credentials...');
  const exchangeAttempted = useRef(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const isDark = theme === 'dark';

  useEffect(() => {
    if (!supabase) {
      setCallbackError('Supabase client is not configured.');
      return;
    }

    const handleCallback = async () => {
      try {
        const url = new URL(window.location.href);
        const code = url.searchParams.get('code');
        const error = url.searchParams.get('error');
        const errorCode = url.searchParams.get('error_code');
        const errorDescription = url.searchParams.get('error_description');

        // ── Diagnostic: always log the full callback context ──────────
        console.log('[CALLBACK RUNTIME]', {
          origin: window.location.origin,
          href: window.location.href,
          mode: import.meta.env.MODE,
          supabaseUrl: import.meta.env.VITE_SUPABASE_URL,
        });
        console.log('[CALLBACK URL]', window.location.href);
        console.log('[CALLBACK SEARCH]', window.location.search);
        console.log('[CALLBACK HASH]', window.location.hash);
        console.log('[CALLBACK ERROR PARAMS]', {
          error,
          error_code: errorCode,
          error_description: errorDescription,
        });
        // ─────────────────────────────────────────────────────────────

        // If Supabase returned an error in the redirect URL, surface it.
        if (error || errorCode || errorDescription) {
          const formatted = [
            error && `error=${error}`,
            errorCode && `error_code=${errorCode}`,
            errorDescription && `error_description=${errorDescription}`,
          ]
            .filter(Boolean)
            .join(' | ');
          console.error('[CALLBACK ERROR PARAMS]', formatted);
          setCallbackError(formatted || 'OAuth exchange failed.');
          return;
        }

        // If there is a PKCE code, exchange it for a session.
        if (code) {
          if (exchangeAttempted.current) {
            console.log('[CALLBACK RUNTIME] Exchange already attempted (StrictMode lock). Skipping.');
          } else {
            exchangeAttempted.current = true;
            setStatus('Exchanging authorization code...');
            const exchangeResult = await supabase.auth.exchangeCodeForSession(code);
            // ── Diagnostic ──────────────────────────────────────────────
            console.log(
              '[RAW EXCHANGE RESULT]',
              JSON.stringify(exchangeResult, null, 2)
            );
            // ─────────────────────────────────────────────────────────────
            if (exchangeResult?.error) {
              throw exchangeResult.error;
            }
          }
        }

        // Verify the session is now established.
        setStatus('Confirming session...');
        const sessionResult = await supabase.auth.getSession();
        // ── Diagnostic ──────────────────────────────────────────────
        console.log(
          '[RAW SESSION RESULT]',
          JSON.stringify(sessionResult, null, 2)
        );
        // ─────────────────────────────────────────────────────────────

        const { data } = sessionResult;
        if (data?.session) {
          console.log('[CALLBACK] Session confirmed for user:', data.session.user.id);
          navigate('/chat', { replace: true });
        } else {
          console.error('[CALLBACK] No session after exchange — OAuth failed.');
          setCallbackError('No session established after OAuth exchange.');
        }
      } catch (err: any) {
        console.error('[CALLBACK] Exception:', err);
        setCallbackError(err?.message || 'OAuth exchange failed.');
      }
    };

    handleCallback();
  }, [navigate]);

  if (!mounted) return null;

  return (
    <div
      className={`min-h-screen flex flex-col items-center justify-center p-6 transition-colors duration-500 ${
        isDark ? 'bg-[#08090e]' : 'bg-[#f7f8fc]'
      }`}
    >
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: isDark
            ? 'radial-gradient(circle 800px at 50% 50%, rgba(139,92,246,0.08), transparent 70%)'
            : 'radial-gradient(circle 800px at 50% 50%, rgba(99,102,241,0.06), transparent 70%)',
        }}
      />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col items-center gap-6 relative z-10"
      >
        {!callbackError && (
          <Loader2
            className="w-8 h-8 animate-spin"
            style={{ color: isDark ? '#f5f5f7' : '#1d1d1f' }}
          />
        )}

        <span
          className="text-[13px] font-medium tracking-widest uppercase"
          style={{ color: isDark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)' }}
        >
          {callbackError ? 'Authentication failed' : status}
        </span>

        {callbackError && (
          <div
            className="max-w-[460px] text-center text-[12px] leading-relaxed rounded-xl border px-4 py-3"
            style={{
              color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.6)',
              borderColor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
              background: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.6)',
            }}
          >
            <p className="text-red-400 font-medium mb-2">OAuth Error</p>
            <p className="font-mono text-[11px]">{callbackError}</p>
            <button
              onClick={() => navigate('/login', { replace: true })}
              className="mt-4 text-[#8b5cf6] hover:underline text-[12px]"
            >
              Return to Login
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}
