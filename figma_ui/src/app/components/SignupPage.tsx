import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router';
import { useAuthContext } from '../providers/AuthProvider';
import { Mail, Lock, Loader2, User } from 'lucide-react';
import { useTheme } from 'next-themes';
import { motion } from 'motion/react';

export default function SignupPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [mounted, setMounted] = useState(false);
  
  const { theme } = useTheme();
  const { signUpWithEmail, handleSignIn, authError, setAuthError, isAuthenticated } = useAuthContext();
  const navigate = useNavigate();
  const [signupSuccess, setSignupSuccess] = useState(false);
  
  useEffect(() => setMounted(true), []);

  const isDark = theme === "dark";

  const handleSignupSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');
    setAuthError('');
    
    if (!name || !email || !password) {
      setLocalError('Please fill in all fields');
      return;
    }

    if (password.length < 6) {
      setLocalError('Password must be at least 6 characters');
      return;
    }

    setIsSubmitting(true);
    try {
      await signUpWithEmail(email, password, name);
      console.log(
        "[LOCAL STORAGE]",
        localStorage.getItem(
          "sb-kyqoygozcxxsmlkkraub-auth-token"
        )
      );
      // Wait for authentication state to propagate
      setSignupSuccess(true);
    } catch (err: any) {
      setLocalError(err.message || 'Failed to create account');
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated && signupSuccess) {
      navigate('/complete-profile', { replace: true });
    } else if (isAuthenticated && !signupSuccess) {
      // If they somehow land on signup but are already authenticated, redirect to chat
      navigate('/chat', { replace: true });
    }
  }, [isAuthenticated, signupSuccess, navigate]);

  const handleGoogleSignIn = async () => {
    try {
      await handleSignIn({ returnTo: '/complete-profile' });
    } catch (err: any) {
      setLocalError(err.message || 'Failed to sign in with Google');
    }
  };

  if (!mounted) return null;

  return (
    <div className={`min-h-screen flex flex-col items-center justify-center p-6 transition-colors duration-500 ${isDark ? "bg-[#08090e]" : "bg-[#f7f8fc]"}`}>
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{
          background: isDark 
            ? "radial-gradient(circle 800px at 50% 100%, rgba(139,92,246,0.08), transparent 70%)"
            : "radial-gradient(circle 800px at 50% 100%, rgba(99,102,241,0.06), transparent 70%)"
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
        <div className="text-center mb-8">
          <h1 
            className="text-3xl font-bold mb-2 tracking-tight"
            style={{ color: isDark ? "#f5f5f7" : "#1d1d1f", fontFamily: "'Inter', sans-serif" }}
          >
            Create Account
          </h1>
          <p style={{ color: isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.5)" }}>
            Join the cognitive revolution
          </p>
        </div>

        <div 
          className="rounded-3xl p-8 shadow-2xl relative overflow-hidden"
          style={{
            background: isDark ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.7)",
            backdropFilter: "blur(24px) saturate(180%)",
            WebkitBackdropFilter: "blur(24px) saturate(180%)",
            border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.06)",
          }}
        >
          <div 
            className="absolute inset-0 pointer-events-none opacity-[0.03] dark:opacity-[0.05]"
            style={{
              backgroundImage: "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)",
              backgroundSize: "20px 20px",
              maskImage: "radial-gradient(circle at center, black, transparent 80%)"
            }}
          />

          {(localError || authError) && (
            <div className="mb-6 rounded-xl bg-red-500/10 p-4 text-[13px] text-red-500 border border-red-500/20 relative z-10 flex items-center">
              {localError || authError}
            </div>
          )}

          <form onSubmit={handleSignupSubmit} className="space-y-4 relative z-10">
            <div>
              <label className="mb-1.5 block text-[13px] font-medium" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-3 h-4 w-4" style={{ color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)" }} />
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full rounded-xl py-2.5 pl-10 pr-4 text-[14px] outline-none transition-all"
                  style={{
                    background: isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.03)",
                    border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.08)",
                    color: isDark ? "#f5f5f7" : "#1d1d1f",
                  }}
                  placeholder="Commander Sentinel"
                />
              </div>
            </div>

            <div>
              <label className="mb-1.5 block text-[13px] font-medium" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-3 h-4 w-4" style={{ color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)" }} />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-xl py-2.5 pl-10 pr-4 text-[14px] outline-none transition-all"
                  style={{
                    background: isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.03)",
                    border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.08)",
                    color: isDark ? "#f5f5f7" : "#1d1d1f",
                  }}
                  placeholder="commander@sentinel.dev"
                />
              </div>
            </div>
            
            <div>
              <label className="mb-1.5 block text-[13px] font-medium" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-3 h-4 w-4" style={{ color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)" }} />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full rounded-xl py-2.5 pl-10 pr-4 text-[14px] outline-none transition-all"
                  style={{
                    background: isDark ? "rgba(0,0,0,0.2)" : "rgba(0,0,0,0.03)",
                    border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.08)",
                    color: isDark ? "#f5f5f7" : "#1d1d1f",
                  }}
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="group relative flex w-full items-center justify-center rounded-xl py-3 text-[14px] font-semibold transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 mt-6"
              style={{
                background: isDark ? "#f5f5f7" : "#1d1d1f",
                color: isDark ? "#1d1d1f" : "#ffffff",
                boxShadow: isDark 
                  ? "0 4px 14px rgba(255,255,255,0.15)"
                  : "0 4px 14px rgba(0,0,0,0.2)",
              }}
            >
              {isSubmitting ? <Loader2 className="animate-spin h-4 w-4" /> : 'Sign Up'}
            </button>
          </form>

          <div className="my-6 flex items-center relative z-10">
            <div className="flex-grow border-t" style={{ borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)" }}></div>
            <span className="mx-4 text-[11px] font-bold tracking-wider uppercase" style={{ color: isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)" }}>OR</span>
            <div className="flex-grow border-t" style={{ borderColor: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)" }}></div>
          </div>

          <button
            onClick={handleGoogleSignIn}
            className="relative z-10 flex w-full items-center justify-center rounded-xl py-3 text-[14px] font-medium transition-all hover:scale-[1.01] active:scale-[0.99]"
            style={{
              background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.03)",
              border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.08)",
              color: isDark ? "#f5f5f7" : "#1d1d1f",
            }}
          >
            <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
              <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            Continue with Google
          </button>

          <p className="mt-8 text-center text-[13px] relative z-10" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>
            Already have an account?{' '}
            <Link to="/login" className="font-semibold hover:underline" style={{ color: "#8b5cf6" }}>
              Log in
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}
