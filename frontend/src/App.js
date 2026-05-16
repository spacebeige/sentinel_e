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
import AdminDashboard from './pages/AdminDashboard';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider, useAuthContext } from './hooks/useAuthContext';
import useStore from './stores/useStore';
import { API_BASE } from './config';
import api from './services/api';

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
    authModalOpen,
    authError,
    authIntent,
    closeAuthModal,
    handleSignIn,
    handleEmailSignIn,
    handleEmailSignUp,
    isSupabaseConfigured,
  } = useAuthContext();

  const [mode, setMode] = React.useState('signin'); // 'signin' or 'signup'
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [localError, setLocalError] = React.useState('');
  const [submitting, setSubmitting] = React.useState(false);

  if (!authModalOpen) return null;

  const signInWithGitHub = async () => {
    setSubmitting(true);
    setLocalError('');
    try {
      await handleSignIn({ returnTo: authIntent || '/chat' });
    } catch (error) {
      setLocalError(error?.message || 'Authentication failed');
    } finally {
      setSubmitting(false);
    }
  };

  const handleEmailAuth = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setLocalError('');
    try {
      if (mode === 'signin') {
        await handleEmailSignIn({ email, password });
        closeAuthModal();
      } else {
        await handleEmailSignUp({ email, password });
        setLocalError('Check your email for a confirmation link!');
      }
    } catch (error) {
      setLocalError(error?.message || 'Authentication failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 px-4">
      <div className="w-full max-w-md rounded-3xl sentinel-surface-panel shadow-2xl border p-6 sentinel-text-primary">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <div className="text-2xl font-semibold tracking-tight">
              {mode === 'signin' ? 'Sign in to Sentinel' : 'Create an account'}
            </div>
            <div className="text-sm sentinel-text-muted mt-1">
              {mode === 'signin'
                ? 'Continue to unlock chat and persistent history.'
                : 'Join Sentinel to start your AI-powered journey.'}
            </div>
          </div>
          <button
            onClick={closeAuthModal}
            className="rounded-full w-9 h-9 flex items-center justify-center sentinel-text-muted hover:bg-black/5 dark:hover:bg-white/10"
            aria-label="Close authentication modal"
          >
            ×
          </button>
        </div>

        <div className="space-y-4">
          {(localError || authError) && (
            <div className="rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-200">
              {localError || authError}
            </div>
          )}

          <form onSubmit={handleEmailAuth} className="space-y-3">
            <div>
              <label className="block text-xs font-medium sentinel-text-muted mb-1 ml-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                className="w-full rounded-2xl border sentinel-border bg-black/5 dark:bg-white/5 px-4 py-2.5 outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>
            <div>
              <label className="block text-xs font-medium sentinel-text-muted mb-1 ml-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                className="w-full rounded-2xl border sentinel-border bg-black/5 dark:bg-white/5 px-4 py-2.5 outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>
            <button
              type="submit"
              disabled={submitting || !isSupabaseConfigured}
              className="w-full rounded-2xl bg-blue-600 hover:bg-blue-700 px-4 py-3 font-semibold text-white disabled:opacity-60 transition-colors"
            >
              {submitting ? 'Processing...' : mode === 'signin' ? 'Sign In' : 'Sign Up'}
            </button>
          </form>

          <div className="relative flex items-center py-2">
            <div className="flex-grow border-t sentinel-border"></div>
            <span className="flex-shrink mx-4 text-xs sentinel-text-muted">OR</span>
            <div className="flex-grow border-t sentinel-border"></div>
          </div>

          <button
            type="button"
            onClick={signInWithGitHub}
            disabled={submitting || !isSupabaseConfigured}
            className="w-full rounded-2xl border sentinel-border bg-white dark:bg-black/20 px-4 py-3 font-semibold sentinel-text-primary hover:bg-black/5 dark:hover:bg-white/5 disabled:opacity-60 flex items-center justify-center gap-2 transition-all"
          >
             Continue with GitHub
          </button>

          <div className="text-center text-sm mt-4">
            <button
              onClick={() => setMode(mode === 'signin' ? 'signup' : 'signin')}
              className="text-blue-500 hover:underline"
            >
              {mode === 'signin'
                ? "Don't have an account? Sign up"
                : 'Already have an account? Sign in'}
            </button>
          </div>

          {!isSupabaseConfigured && (
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-xs text-amber-800 dark:text-amber-200">
              Missing REACT_APP_SUPABASE_URL or REACT_APP_SUPABASE_ANON_KEY in frontend environment.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function AppContent() {
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
                  <AdminDashboard />
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
