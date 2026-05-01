/**
 * ============================================================
 * App.js — Application Router Shell
 * ============================================================
 */

import React from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
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
import { auth } from './firebase';
import { API_BASE } from './config';
import api from './services/api';

function SessionInitializer({ children }) {
  const { isAuthenticated, loading } = useAuthContext();
  const reloadHistory = useStore(state => state.reloadHistory);
  const setUserId = useStore(state => state.setUserId);
  const clearSession = useStore(state => state.clearSession);
  const storeUserId = useStore(state => state.userId);
  const storeIsLoaded = useStore(state => state.isLoaded);
  const hasHydrated = useStore(state => state.hasHydrated);

  const userId = auth.currentUser?.uid;

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

      if (switchedUsers || !storeIsLoaded) {
        const initFlow = async () => {
          try {
            await api.createSession();
            await reloadHistory();
          } catch (error) {
            console.error('INIT FLOW FAILED:', error);
            useStore.setState({ isLoaded: true });
          }
        };

        initFlow();
      }
    } else {
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
    handleSignUp,
  } = useAuthContext();
  const navigate = useNavigate();
  const [mode, setMode] = React.useState('signin');
  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [displayName, setDisplayName] = React.useState('');
  const [localError, setLocalError] = React.useState('');
  const [submitting, setSubmitting] = React.useState(false);

  if (!authModalOpen) return null;

  const submit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setLocalError('');

    try {
      if (mode === 'signup') {
        await handleSignUp(email.trim(), password, displayName.trim());
      } else {
        await handleSignIn(email.trim(), password);
      }

      closeAuthModal();
      navigate(authIntent || '/chat', { replace: true });
    } catch (error) {
      setLocalError(error?.message || 'Authentication failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 px-4">
      <div className="w-full max-w-md rounded-3xl bg-white dark:bg-[#111827] shadow-2xl border border-black/5 dark:border-white/10 p-6 text-[#0f172a] dark:text-white">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <div className="text-2xl font-semibold tracking-tight">
              {mode === 'signup' ? 'Create your account' : 'Sign in to Sentinel'}
            </div>
            <div className="text-sm text-[#6b7280] dark:text-white/60 mt-1">
              Use Firebase auth to unlock chat and history.
            </div>
          </div>
          <button
            onClick={closeAuthModal}
            className="rounded-full w-9 h-9 flex items-center justify-center text-[#6b7280] hover:bg-black/5 dark:hover:bg-white/10"
            aria-label="Close authentication modal"
          >
            ×
          </button>
        </div>

        <form onSubmit={submit} className="space-y-3">
          {mode === 'signup' && (
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wide text-[#6b7280] dark:text-white/60 mb-1">
                Display name
              </label>
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full rounded-2xl border border-black/10 dark:border-white/10 bg-white dark:bg-[#0b1220] px-4 py-3 outline-none"
                placeholder="Alex"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-[#6b7280] dark:text-white/60 mb-1">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-2xl border border-black/10 dark:border-white/10 bg-white dark:bg-[#0b1220] px-4 py-3 outline-none"
              placeholder="you@example.com"
              autoComplete="email"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-[#6b7280] dark:text-white/60 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-2xl border border-black/10 dark:border-white/10 bg-white dark:bg-[#0b1220] px-4 py-3 outline-none"
              placeholder="••••••••"
              autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
            />
          </div>

          {(localError || authError) && (
            <div className="rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-200">
              {localError || authError}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-2xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] px-4 py-3 font-semibold text-white disabled:opacity-60"
          >
            {submitting ? 'Working…' : mode === 'signup' ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <button
          type="button"
          onClick={() => setMode((prev) => (prev === 'signin' ? 'signup' : 'signin'))}
          className="mt-4 w-full text-sm text-[#2563eb] dark:text-cyan-300 hover:underline"
        >
          {mode === 'signin' ? 'Need an account? Sign up' : 'Already have an account? Sign in'}
        </button>
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
