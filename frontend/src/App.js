/**
 * ============================================================
 * App.js — Application Router Shell
 * ============================================================
 *
 * ARCHITECTURE:
 *   App.js          → Router + Route definitions (this file)
 *   layout/Layout   → Persistent shell (Navbar + Footer + Theme)
 *   pages/*         → Page wrappers (thin, no logic)
 *   components/ChatEngine → Logic authority (all backend state/handlers)
 *   figma_shell/*   → Visual authority (controlled presentation)
 *   figma_features/* → Standalone marketing pages
 *
 * ROUTING:
 *   /         → Landing Page (default — app opens here)
 *   /chat     → Chat Interface (ChatEngine)
 *   /pricing  → Pricing Page
 *   /models   → Models Page
 *
 * Backend logic is fully encapsulated in ChatEngine.
 * This file contains ZERO backend calls.
 *
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
import { AuthProvider } from './hooks/useAuthContext';
import useStore from './stores/useStore';
import { useAuth } from '@clerk/clerk-react';
import { API_BASE } from './config';

function SessionInitializer({ children }) {
  const { isLoaded, isSignedIn, userId } = useAuth();
  const reloadHistory = useStore(state => state.reloadHistory);
  const setUserId = useStore(state => state.setUserId);
  const clearSession = useStore(state => state.clearSession);
  const storeUserId = useStore(state => state.userId);
  const storeIsLoaded = useStore(state => state.isLoaded);
  const hasHydrated = useStore(state => state.hasHydrated);

  React.useEffect(() => {
    // Fire-and-forget wakeup ping for Render free-tier cold starts.
    fetch(`${API_BASE}/health`).catch(() => {});
  }, []);

  React.useEffect(() => {
    if (!hasHydrated) return;
    if (!isLoaded) return;
    if (!userId && isSignedIn) return;

    if (isSignedIn) {
      const switchedUsers = !!storeUserId && storeUserId !== userId;
      if (switchedUsers) {
        clearSession();
      }

      setUserId(userId);

      // Render cached chats immediately (from persist); then reconcile with API.
      // Only fetch after hydration + userId availability.
      if (switchedUsers || !storeIsLoaded) {
        reloadHistory();
      }
    } else {
      // Preserve cached history across logout/login to avoid UI wipeouts.
      // If a different user signs in, switchedUsers branch above clears it safely.
      setUserId(null);
    }
  }, [hasHydrated, isLoaded, isSignedIn, userId, reloadHistory, setUserId, clearSession, storeUserId, storeIsLoaded]);

  if (!hasHydrated) {
    return null;
  }

  return children;
}

function App() {
  return (
    <CognitiveStoreProvider>
        <BrowserRouter>
          <AuthProvider>
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
          </AuthProvider>
        </BrowserRouter>
    </CognitiveStoreProvider>
  );
}

export default App;
