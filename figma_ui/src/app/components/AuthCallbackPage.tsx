import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { useAuthContext } from '../providers/AuthProvider';
import { Loader2 } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';
import { supabase } from '../lib/supabase';

export default function AuthCallbackPage() {
  const { isAuthenticated, authResolved } = useAuthContext();
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => setMounted(true), []);
  const isDark = theme === "dark";

  useEffect(() => {
    // Explicitly await the session exchange from Supabase to prevent race condition
    // where loading=false before the PKCE exchange finishes.
    const checkSession = async () => {
      try {
        const { data, error } = await supabase.auth.getSession();
        if (data?.session) {
          navigate('/chat', { replace: true });
        } else if (authResolved) {
          // If we waited and still no session, and the context says we're done, go to login
          navigate('/login', { replace: true });
        }
      } catch (err) {
        navigate('/login', { replace: true });
      }
    };
    checkSession();
  }, [authResolved, navigate]);

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
          Verifying credentials...
        </span>
      </motion.div>
    </div>
  );
}
