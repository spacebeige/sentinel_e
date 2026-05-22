/**
 * ============================================================
 * App.js — Application Router Shell
 * ============================================================
 */

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { CognitiveStoreProvider } from './stores/cognitiveStore';
import Layout from './layout/Layout';
import LandingPage from './pages/LandingPage';
import ChatPage from './pages/ChatPage';
import PricingPageWrapper from './pages/PricingPageWrapper';
import ModelsPageWrapper from './pages/ModelsPageWrapper';
import CognitiveMissionControl from './pages/CognitiveMissionControl';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider, useAuthContext } from './hooks/useAuthContext';
import useSupabaseAuth from './hooks/useSupabaseAuth';
import useStore from './stores/useStore';
import { API_BASE } from './config';
import api from './services/api';
import SentinelIdentity from './components/SentinelIdentity';
import { LoadingScreen } from './components/LoadingScreen';

function SessionInitializer({ children }) {
  const { isAuthenticated, loading, user } = useAuthContext();
  const reloadHistory = useStore(state => state.reloadHistory);
  const setUserId = useStore(state => state.setUserId);
  const clearSession = useStore(state => state.clearSession);
  const storeUserId = useStore(state => state.userId);
  const storeIsLoaded = useStore(state => state.isLoaded);
  const hasHydrated = useStore(state => state.hasHydrated);
  const initInFlightRef = React.useRef(false);

  const userId = user?.id || null;

  React.useEffect(() => {
    fetch(`${API_BASE}/health`).catch(() => {});
  }, []);

  React.useEffect(() => {
    if (!hasHydrated) return;
    if (loading) return;
    if (!userId && isAuthenticated) return;

    if (isAuthenticated) {
      const switchedUsers = !!storeUserId && storeUserId !== userId;
      if (switchedUsers) {
        clearSession();
      }

      setUserId(userId);

      if ((switchedUsers || !storeIsLoaded) && !initInFlightRef.current) {
        initInFlightRef.current = true;
        const initFlow = async () => {
          try {
            await api.createSession();
            await reloadHistory();
          } catch (error) {
            console.error('INIT FLOW FAILED:', error);
            useStore.setState({ isLoaded: true });
          } finally {
            initInFlightRef.current = false;
          }
        };

        initFlow();
      }
    } else {
      if (storeUserId) {
        clearSession();
      }
      setUserId(null);
    }
  }, [hasHydrated, loading, isAuthenticated, userId, reloadHistory, setUserId, clearSession, storeUserId, storeIsLoaded]);

  return <>{children}</>;
}

function AuthModal() {
  const {
    authModalOpen, authError, authIntent,
    closeAuthModal, handleSignIn, isSupabaseConfigured,
  } = useAuthContext();
  const { signInWithEmail, signUpWithEmail } = useSupabaseAuth();

  const [tab, setTab] = React.useState('signin');
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [confirmPassword, setConfirmPassword] = React.useState('');
  const [localError, setLocalError] = React.useState('');
  const [localSuccess, setLocalSuccess] = React.useState('');
  const [submitting, setSubmitting] = React.useState(false);

  React.useEffect(() => {
    if (!authModalOpen) {
      setTab('signin'); setEmail(''); setPassword('');
      setConfirmPassword(''); setLocalError(''); setLocalSuccess('');
    }
  }, [authModalOpen]);

  if (!authModalOpen) return null;

  const inputCls = 'w-full rounded-xl border px-4 py-3 text-sm sentinel-input focus:outline-none transition-all sentinel-border';

  const handleGoogleSignIn = async () => {
    setSubmitting(true); setLocalError('');
    try { await handleSignIn({ returnTo: authIntent || '/chat' }); }
    catch (error) { setLocalError(error?.message || 'Authentication failed'); }
    finally { setSubmitting(false); }
  };

  const handleEmailSignIn = async (e) => {
    e.preventDefault();
    if (!email || !password) { setLocalError('Email and password are required.'); return; }
    setSubmitting(true); setLocalError('');
    try {
      const result = await signInWithEmail({ email, password });
      if (result?.error) throw result.error;
      closeAuthModal();
    } catch (error) { setLocalError(error?.message || 'Sign in failed. Check your credentials.'); }
    finally { setSubmitting(false); }
  };

  const handleEmailSignUp = async (e) => {
    e.preventDefault();
    if (!email || !password) { setLocalError('Email and password are required.'); return; }
    if (password !== confirmPassword) { setLocalError('Passwords do not match.'); return; }
    if (password.length < 6) { setLocalError('Password must be at least 6 characters.'); return; }
    setSubmitting(true); setLocalError('');
    try {
      const result = await signUpWithEmail({ email, password, options: { emailRedirectTo: window.location.origin + '/chat' } });
      if (result?.error) throw result.error;
      setLocalSuccess('Check your email for a confirmation link to complete sign-up.');
    } catch (error) { setLocalError(error?.message || 'Sign up failed. Please try again.'); }
    finally { setSubmitting(false); }
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    if (!email) { setLocalError('Enter your email address.'); return; }
    setSubmitting(true); setLocalError('');
    try {
      const { supabase } = await import('./lib/supabase');
      const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo: window.location.origin + '/chat' });
      if (error) throw error;
      setLocalSuccess('Password reset email sent. Check your inbox.');
    } catch (error) { setLocalError(error?.message || 'Failed to send reset email.'); }
    finally { setSubmitting(false); }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 px-4 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-3xl sentinel-surface-panel shadow-2xl border p-6 sentinel-text-primary">
        <div className="flex items-start justify-between gap-4 mb-5">
          <div>
            <div className="mb-3"><SentinelIdentity size={40} showLabel label="Sentinel-E" pulse /></div>
            <div className="text-xl font-semibold tracking-tight">
              {tab === 'signin' && 'Sign in to Sentinel-E'}
              {tab === 'signup' && 'Create your account'}
              {tab === 'forgot' && 'Reset your password'}
            </div>
            <div className="text-sm sentinel-text-muted mt-1">
              {tab === 'signin' && 'Access the cognitive runtime.'}
              {tab === 'signup' && 'Join to access multi-model AI.'}
              {tab === 'forgot' && "We'll send a reset link to your email."}
            </div>
          </div>
          <button onClick={closeAuthModal}
            className="rounded-full w-9 h-9 flex items-center justify-center sentinel-text-muted hover:bg-black/5 dark:hover:bg-white/10 text-xl flex-shrink-0"
            aria-label="Close authentication modal">×</button>
        </div>

        {(localError || authError) && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-300 mb-4">
            {localError || authError}
          </div>
        )}
        {localSuccess && (
          <div className="rounded-xl border border-green-500/20 bg-green-500/10 px-4 py-3 text-sm text-green-700 dark:text-green-300 mb-4">
            {localSuccess}
          </div>
        )}
        {!isSupabaseConfigured && (
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-xs text-amber-800 dark:text-amber-200 mb-4">
            Supabase environment variables are missing.
          </div>
        )}

        <div className="space-y-4">
          {tab !== 'forgot' && (
            <>
              <button type="button" onClick={handleGoogleSignIn}
                disabled={submitting || !isSupabaseConfigured}
                className="w-full rounded-xl border sentinel-border bg-white dark:bg-black/20 px-4 py-3 font-semibold sentinel-text-primary hover:bg-black/5 dark:hover:bg-white/5 disabled:opacity-60 flex items-center justify-center gap-2.5 transition-all">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <path d="M17.64 9.2045c0-.638-.0573-1.252-.1636-1.8409H9v3.4814h4.8436c-.2086 1.125-.8427 2.0782-1.7959 2.7164v2.2581h2.9087c1.7018-1.5668 2.6836-3.874 2.6836-6.6149z" fill="#4285F4"/>
                  <path d="M9 18c2.43 0 4.4673-.806 5.9564-2.1805l-2.9087-2.2581c-.806.54-1.8368.859-3.0477.859-2.344 0-4.3282-1.5836-5.036-3.7104H.9574v2.3318C2.4382 15.9832 5.4818 18 9 18z" fill="#34A853"/>
                  <path d="M3.964 10.71c-.18-.54-.2822-1.1168-.2822-1.71s.1023-1.17.2822-1.71V4.9582H.9574C.3477 6.1732 0 7.548 0 9s.3477 2.8268.9574 4.0418L3.964 10.71z" fill="#FBBC05"/>
                  <path d="M9 3.5795c1.3214 0 2.5077.4541 3.4405 1.346l2.5813-2.5813C13.4632.8918 11.4259 0 9 0 5.4818 0 2.4382 2.0168.9574 4.9582L3.964 7.29C4.6718 5.1632 6.656 3.5795 9 3.5795z" fill="#EA4335"/>
                </svg>
                Continue with Google
              </button>
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px" style={{ backgroundColor: 'var(--border-primary)' }} />
                <span className="text-xs sentinel-text-muted font-medium">or</span>
                <div className="flex-1 h-px" style={{ backgroundColor: 'var(--border-primary)' }} />
              </div>
            </>
          )}

          {tab !== 'forgot' && (
            <div className="flex gap-1 p-1 rounded-xl" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
              {[['signin', 'Sign In'], ['signup', 'Sign Up']].map(([t, label]) => (
                <button key={t} onClick={() => { setTab(t); setLocalError(''); setLocalSuccess(''); }}
                  className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                    tab === t ? 'sentinel-surface sentinel-text-primary shadow-sm' : 'sentinel-text-muted'
                  }`}>{label}</button>
              ))}
            </div>
          )}

          {tab === 'signin' && (
            <form onSubmit={handleEmailSignIn} className="space-y-3">
              <input type="email" placeholder="Email address" value={email}
                onChange={e => setEmail(e.target.value)} className={inputCls}
                autoComplete="email" disabled={submitting} />
              <input type="password" placeholder="Password" value={password}
                onChange={e => setPassword(e.target.value)} className={inputCls}
                autoComplete="current-password" disabled={submitting} />
              <div className="flex justify-end">
                <button type="button" onClick={() => { setTab('forgot'); setLocalError(''); setLocalSuccess(''); }}
                  className="text-xs sentinel-text-muted hover:underline">Forgot password?</button>
              </div>
              <button type="submit" disabled={submitting || !isSupabaseConfigured}
                className="w-full rounded-xl px-4 py-3 font-semibold text-white disabled:opacity-60 transition-all"
                style={{ backgroundColor: 'var(--accent-blue)' }}>
                {submitting ? 'Signing in…' : 'Sign In'}
              </button>
            </form>
          )}

          {tab === 'signup' && (
            <form onSubmit={handleEmailSignUp} className="space-y-3">
              <input type="email" placeholder="Email address" value={email}
                onChange={e => setEmail(e.target.value)} className={inputCls}
                autoComplete="email" disabled={submitting} />
              <input type="password" placeholder="Password (min. 6 characters)" value={password}
                onChange={e => setPassword(e.target.value)} className={inputCls}
                autoComplete="new-password" disabled={submitting} />
              <input type="password" placeholder="Confirm password" value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)} className={inputCls}
                autoComplete="new-password" disabled={submitting} />
              <button type="submit" disabled={submitting || !isSupabaseConfigured}
                className="w-full rounded-xl px-4 py-3 font-semibold text-white disabled:opacity-60 transition-all"
                style={{ backgroundColor: 'var(--accent-blue)' }}>
                {submitting ? 'Creating account…' : 'Create Account'}
              </button>
            </form>
          )}

          {tab === 'forgot' && (
            <form onSubmit={handleForgotPassword} className="space-y-3">
              <input type="email" placeholder="Email address" value={email}
                onChange={e => setEmail(e.target.value)} className={inputCls}
                autoComplete="email" disabled={submitting} />
              <button type="submit" disabled={submitting || !isSupabaseConfigured}
                className="w-full rounded-xl px-4 py-3 font-semibold text-white disabled:opacity-60 transition-all"
                style={{ backgroundColor: 'var(--accent-blue)' }}>
                {submitting ? 'Sending…' : 'Send Reset Link'}
              </button>
              <button type="button"
                onClick={() => { setTab('signin'); setLocalError(''); setLocalSuccess(''); }}
                className="w-full text-sm sentinel-text-muted hover:underline py-1">
                ← Back to Sign In
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

function AppContent() {
  const { authResolved, isAuthenticated } = useAuthContext();
  const storeIsLoaded = useStore(state => state.isLoaded);
  const [bootState, setBootState] = React.useState('BOOTING');

  React.useEffect(() => {
    if (authResolved) {
      if (isAuthenticated) {
        if (storeIsLoaded) {
          setBootState('READY');
        }
      } else {
        setBootState('READY');
      }
    }
  }, [authResolved, isAuthenticated, storeIsLoaded]);

  if (bootState === 'BOOTING') {
    return <LoadingScreen message="Booting Runtime..." subtext="Synchronizing cognitive subsystems..." />;
  }

  return (
    <>
      <SessionInitializer>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<LandingPage />} />
            <Route
              path="/chat"
              element={
                <ProtectedRoute>
                  <ChatPage />
                </ProtectedRoute>
              }
            />
            <Route path="/pricing" element={<PricingPageWrapper />} />
            <Route
              path="/models"
              element={
                <ProtectedRoute>
                  <ModelsPageWrapper />
                </ProtectedRoute>
              }
            />
            <Route
              path="/admin"
              element={
                <ProtectedRoute requireAdmin>
                  <CognitiveMissionControl />
                </ProtectedRoute>
              }
            />
          </Route>
        </Routes>
      </SessionInitializer>
      <AuthModal />
    </>
  );
}

export default function App() {
  return (
    <CognitiveStoreProvider>
      <BrowserRouter>
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </BrowserRouter>
    </CognitiveStoreProvider>
  );
}
