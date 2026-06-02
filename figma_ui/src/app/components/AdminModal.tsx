import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Shield, Lock, Mail, Loader2, ArrowRight } from 'lucide-react';
import { useAuthContext } from '../providers/AuthProvider';
import { useSupabaseAuth } from '../hooks/useSupabaseAuth';
import { submitAdminRequest } from '../api';
import { useTheme } from 'next-themes';

interface AdminModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AdminModal({ isOpen, onClose }: AdminModalProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const { isAuthenticated, isAdmin, role } = useAuthContext();
  const { signInWithEmail } = useSupabaseAuth();

  // Mode: 'login' | 'request' | 'denied'
  const defaultMode = !isAuthenticated ? 'login' : isAdmin ? 'denied' : 'denied';
  const [mode, setMode] = useState<'login' | 'request' | 'denied'>(defaultMode);

  // Login State
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  // Request State
  const [adminReq, setAdminReq] = useState({ name: '', email: '', organization: '', reason: '' });
  const [reqStatus, setReqStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [reqError, setReqError] = useState('');

  // Reset state when opened/closed
  React.useEffect(() => {
    if (isOpen) {
      setMode(!isAuthenticated ? 'login' : isAdmin ? 'denied' : 'denied');
      setReqStatus('idle');
      setReqError('');
      setLoginError('');
      setEmail('');
      setPassword('');
    }
  }, [isOpen, isAuthenticated]);

  if (!isOpen) return null;
  if (isAdmin || role === "owner") return null;

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    if (!email || !password) {
      setLoginError('Please fill in all fields');
      return;
    }
    setIsLoggingIn(true);
    try {
      await signInWithEmail(email, password);
      onClose();
    } catch (err: any) {
      setLoginError(err.message || 'Failed to sign in');
      setIsLoggingIn(false);
    }
  };

  const handleRequestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!adminReq.name || !adminReq.email || !adminReq.reason) {
      setReqError('Please fill in required fields');
      return;
    }
    setReqStatus('submitting');
    const result = await submitAdminRequest(adminReq);
    if (result.status === 'success') {
      setReqStatus('success');
    } else {
      setReqStatus('error');
      setReqError(result.message || 'Failed to submit request');
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="w-[calc(100vw-32px)] sm:max-w-[480px] rounded-2xl p-6 shadow-2xl relative overflow-y-auto"
            style={{
              maxHeight: "90vh",
              background: isDark ? "#1d1d1f" : "#ffffff",
              border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(0,0,0,0.1)",
            }}
          >
            {mode === 'login' && (
              <>
                <div className="flex items-center gap-3 mb-6">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-500">
                    <Shield className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold" style={{ color: isDark ? "#f5f5f7" : "#1d1d1f" }}>Admin Login</h2>
                    <p className="text-[13px]" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>Sign in to Sentinel-E</p>
                  </div>
                </div>

                <form onSubmit={handleLoginSubmit} className="space-y-4">
                  {loginError && (
                    <div className="rounded-xl bg-red-500/10 p-3 text-[13px] text-red-500 border border-red-500/20">
                      {loginError}
                    </div>
                  )}
                  <div>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Mail className="h-[18px] w-[18px]" style={{ color: isDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }} />
                      </div>
                      <input
                        type="email"
                        placeholder="Email address"
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        className="w-full rounded-xl py-3 pl-10 pr-3 text-[14px] outline-none"
                        style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }}
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-[18px] w-[18px]" style={{ color: isDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)" }} />
                      </div>
                      <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        className="w-full rounded-xl py-3 pl-10 pr-3 text-[14px] outline-none"
                        style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }}
                        required
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={isLoggingIn}
                    className="w-full flex items-center justify-center py-3 rounded-xl text-[14px] font-semibold transition-transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
                    style={{ background: isDark ? "#f5f5f7" : "#1d1d1f", color: isDark ? "#1d1d1f" : "#f5f5f7" }}
                  >
                    {isLoggingIn ? <Loader2 className="w-5 h-5 animate-spin" /> : "Sign In to Portal"}
                  </button>
                </form>

                <div className="mt-6 text-center">
                  <button onClick={() => setMode('request')} className="text-[13px] font-medium text-purple-500 hover:text-purple-400">
                    Need access? Request admin approval
                  </button>
                </div>
              </>
            )}

            {mode === 'denied' && (
              <div className="text-center py-4">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-red-500/10 text-red-500">
                  <Shield className="h-8 w-8" />
                </div>
                <h2 className="text-xl font-bold mb-2" style={{ color: isDark ? "#f5f5f7" : "#1d1d1f" }}>Access Denied</h2>
                <p className="text-sm mb-6" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>
                  Admin access has not been approved for your account.
                </p>
                <button
                  onClick={() => setMode('request')}
                  className="w-full py-3 rounded-xl text-[14px] font-semibold transition-transform hover:scale-[1.02]"
                  style={{ background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }}
                >
                  Request Admin Access
                </button>
              </div>
            )}

            {mode === 'request' && (
              <>
                <div className="flex items-center gap-3 mb-6">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/10 text-purple-500">
                    <Shield className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold" style={{ color: isDark ? "#f5f5f7" : "#1d1d1f" }}>Request Access</h2>
                    <p className="text-[13px]" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>Submit request to owner</p>
                  </div>
                </div>

                {reqStatus === 'success' ? (
                  <div className="text-center py-6">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-500 mb-4">
                      <Shield size={24} />
                    </div>
                    <h3 className="font-medium text-emerald-500 mb-2">Request Submitted</h3>
                    <p className="text-[13px] mb-6" style={{ color: isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)" }}>
                      Your request has been submitted and is awaiting approval.
                    </p>
                    <button
                      onClick={onClose}
                      className="w-full py-2.5 rounded-xl text-[14px] font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition-colors"
                    >
                      Close
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleRequestSubmit} className="space-y-4">
                    {reqError && (
                      <div className="rounded-xl bg-red-500/10 p-3 text-[13px] text-red-500 border border-red-500/20">
                        {reqError}
                      </div>
                    )}
                    <div>
                      <label className="text-[12px] font-medium mb-1 block" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>Full Name</label>
                      <input required type="text" value={adminReq.name} onChange={e => setAdminReq({ ...adminReq, name: e.target.value })} className="w-full rounded-xl py-2.5 px-3 text-[14px] outline-none" style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }} />
                    </div>
                    <div>
                      <label className="text-[12px] font-medium mb-1 block" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>Email</label>
                      <input required type="email" value={adminReq.email} onChange={e => setAdminReq({ ...adminReq, email: e.target.value })} className="w-full rounded-xl py-2.5 px-3 text-[14px] outline-none" style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }} />
                    </div>
                    <div>
                      <label className="text-[12px] font-medium mb-1 block" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>Organization (Optional)</label>
                      <input type="text" value={adminReq.organization} onChange={e => setAdminReq({ ...adminReq, organization: e.target.value })} className="w-full rounded-xl py-2.5 px-3 text-[14px] outline-none" style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }} />
                    </div>
                    <div>
                      <label className="text-[12px] font-medium mb-1 block" style={{ color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.7)" }}>Reason for Access</label>
                      <textarea required value={adminReq.reason} onChange={e => setAdminReq({ ...adminReq, reason: e.target.value })} className="w-full rounded-xl py-2.5 px-3 text-[14px] outline-none min-h-[80px]" style={{ background: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }} />
                    </div>
                    <div className="flex gap-3 pt-2">
                      <button type="button" onClick={() => setMode(defaultMode)} className="flex-1 py-2.5 rounded-xl text-[14px] font-medium transition-colors" style={{ background: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)", color: isDark ? "#fff" : "#000" }}>Back</button>
                      <button type="submit" disabled={reqStatus === 'submitting'} className="flex-1 py-2.5 rounded-xl text-[14px] font-medium bg-purple-500 text-white hover:bg-purple-600 transition-colors disabled:opacity-50">
                        {reqStatus === 'submitting' ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : "Submit"}
                      </button>
                    </div>
                  </form>
                )}
              </>
            )}

            {/* Close button in top right */}
            <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)" }}>
                <path d="M18 6 6 18"/><path d="m6 6 12 12"/>
              </svg>
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
