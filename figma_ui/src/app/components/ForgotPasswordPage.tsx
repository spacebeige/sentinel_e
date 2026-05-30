import React, { useState, useEffect } from 'react';
import { Link } from 'react-router';
import { Mail, Loader2, ArrowLeft } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';
import { useSupabaseAuth } from '../hooks/useSupabaseAuth';

// ============================================================
// ForgotPasswordPage
//
// Responsibilities:
//   - Accept email input
//   - Call resetPasswordForEmail via useSupabaseAuth
//   - Show success/error
//   - No navigation ownership
// ============================================================
export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [localError, setLocalError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [mounted, setMounted] = useState(false);

  const { theme } = useTheme();
  const { resetPasswordForEmail } = useSupabaseAuth();

  useEffect(() => setMounted(true), []);

  const isDark = theme === 'dark';

  const handleResetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');
    setSuccessMsg('');

    if (!email) {
      setLocalError('Please enter your email address');
      return;
    }

    setIsSubmitting(true);
    try {
      await resetPasswordForEmail(email);
      setSuccessMsg('Check your email for the password reset link.');
    } catch (err: any) {
      setLocalError(err.message || 'Failed to send reset email');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!mounted) return null;

  return (
    <div className={`min-h-screen flex flex-col items-center justify-center p-6 transition-colors duration-500 ${isDark ? 'bg-[#08090e]' : 'bg-[#f7f8fc]'}`}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background: isDark
            ? 'radial-gradient(circle 800px at 50% 0%, rgba(139,92,246,0.08), transparent 70%)'
            : 'radial-gradient(circle 800px at 50% 0%, rgba(99,102,241,0.06), transparent 70%)',
        }}
      />

      <Link to="/" className="fixed top-8 left-8 z-50 transition-transform hover:scale-105">
        <img src="/logo.png" alt="Sentinel-E" className="h-7 w-auto" />
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-[420px] relative z-10"
      >
        <Link to="/login" className="inline-flex items-center text-[13px] font-medium mb-6 hover:underline" style={{ color: isDark ? 'rgba(255,255,255,0.6)' : 'rgba(0,0,0,0.6)' }}>
          <ArrowLeft className="w-3.5 h-3.5 mr-1.5" /> Back to Login
        </Link>

        <div className="text-center mb-8">
          <h1
            className="text-3xl font-bold mb-2 tracking-tight"
            style={{ color: isDark ? '#f5f5f7' : '#1d1d1f', fontFamily: "'Inter', sans-serif" }}
          >
            Reset Password
          </h1>
          <p style={{ color: isDark ? 'rgba(255,255,255,0.45)' : 'rgba(0,0,0,0.5)' }}>
            Enter your email to receive a recovery link
          </p>
        </div>

        <div
          className="rounded-3xl p-8 shadow-2xl relative overflow-hidden"
          style={{
            background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(255,255,255,0.7)',
            backdropFilter: 'blur(24px) saturate(180%)',
            WebkitBackdropFilter: 'blur(24px) saturate(180%)',
            border: isDark ? '1px solid rgba(255,255,255,0.08)' : '1px solid rgba(0,0,0,0.06)',
          }}
        >
          <div
            className="absolute inset-0 pointer-events-none opacity-[0.03] dark:opacity-[0.05]"
            style={{
              backgroundImage: 'linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)',
              backgroundSize: '20px 20px',
              maskImage: 'radial-gradient(circle at center, black, transparent 80%)',
            }}
          />

          {localError && (
            <div className="mb-6 rounded-xl bg-red-500/10 p-4 text-[13px] text-red-500 border border-red-500/20 relative z-10">
              {localError}
            </div>
          )}

          {successMsg && (
            <div className="mb-6 rounded-xl bg-green-500/10 p-4 text-[13px] text-green-500 border border-green-500/20 relative z-10">
              {successMsg}
            </div>
          )}

          <form onSubmit={handleResetSubmit} className="space-y-4 relative z-10">
            <div>
              <label className="mb-1.5 block text-[13px] font-medium" style={{ color: isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)' }}>
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-3 h-4 w-4" style={{ color: isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.3)' }} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-xl py-2.5 pl-10 pr-4 text-[14px] outline-none transition-all"
                  style={{
                    background: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.03)',
                    border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.08)',
                    color: isDark ? '#f5f5f7' : '#1d1d1f',
                  }}
                  placeholder="user@sentinel.dev"
                  autoComplete="email"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting || !!successMsg}
              className="group relative flex w-full items-center justify-center rounded-xl py-3 text-[14px] font-semibold transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 mt-6"
              style={{
                background: isDark ? '#f5f5f7' : '#1d1d1f',
                color: isDark ? '#1d1d1f' : '#ffffff',
                boxShadow: isDark ? '0 4px 14px rgba(255,255,255,0.15)' : '0 4px 14px rgba(0,0,0,0.2)',
              }}
            >
              {isSubmitting ? <Loader2 className="animate-spin h-4 w-4" /> : 'Send Reset Link'}
            </button>
          </form>
        </div>
      </motion.div>
    </div>
  );
}
