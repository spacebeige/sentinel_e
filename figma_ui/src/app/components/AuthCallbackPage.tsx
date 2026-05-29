import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { Loader2 } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';
import { supabase } from '../lib/supabase';

export default function AuthCallbackPage() {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [callbackError, setCallbackError] = useState<string | null>(null);

  useEffect(() => {
    console.log('[AUTH CALLBACK MOUNTED]');
    setMounted(true);
  }, []);
  const isDark = theme === "dark";

  useEffect(() => {
    // Explicitly await the session exchange from Supabase to prevent race condition
    // where loading=false before the PKCE exchange finishes.
    const checkSession = async () => {
      try {
        const url = new URL(window.location.href);
        const code = url.searchParams.get('code');
        const returnTo = url.searchParams.get('returnTo');
        const destination = returnTo && returnTo.startsWith('/') ? returnTo : '/chat';
        const error = url.searchParams.get('error');
        const errorCode = url.searchParams.get('error_code');
        const errorDescription = url.searchParams.get('error_description');

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

        if (error || errorCode || errorDescription) {
          const formatted = [
            error && `error=${error}`,
            errorCode && `error_code=${errorCode}`,
            errorDescription && `error_description=${errorDescription}`,
          ].filter(Boolean).join(' | ');
          setCallbackError(formatted || 'OAuth exchange failed.');
        }

        if (code) {
          const exchangeResult = await supabase.auth.exchangeCodeForSession(code);
          console.log(
            '[RAW EXCHANGE RESULT]',
            JSON.stringify(exchangeResult, null, 2)
          );
          if (exchangeResult?.error) {
            throw exchangeResult.error;
          }
        }

        const sessionResult = await supabase.auth.getSession();
        console.log(
          '[RAW SESSION RESULT]',
          JSON.stringify(sessionResult, null, 2)
        );
        const { data } = sessionResult;
        if (data?.session) {
          navigate(destination, { replace: true });
        } else {
          setCallbackError('No session established after OAuth exchange.');
        }
      } catch (err) {
        setCallbackError(err?.message || 'OAuth exchange failed.');
      }
    };
    checkSession();
  }, [navigate]);

  if (!mounted) return null;

  return (
    <div className={`min-h-screen flex flex-col items-center justify-center p-6 transition-colors duration-500 ${isDark ? "bg-[#08090e]" : "bg-[#f7f8fc]"}`}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: isDark
            ? "radial-gradient(circle 800px at 50% 50%, rgba(139,92,246,0.08), transparent 70%)"
            : "radial-gradient(circle 800px at 50% 50%, rgba(99,102,241,0.06), transparent 70%)"
        }}
      />

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col items-center gap-6 relative z-10"
      >
        <Loader2 className="w-8 h-8 animate-spin" style={{ color: isDark ? "#f5f5f7" : "#1d1d1f" }} />
        <span
          className="text-[13px] font-medium tracking-widest uppercase"
          style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}
        >
          {callbackError ? 'Authentication failed' : 'Verifying credentials...'}
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
            {callbackError}
          </div>
        )}
      </motion.div>
    </div>
  );
}
