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
import ErrorBoundary from './components/ErrorBoundary';
import { AuthProvider } from './hooks/useAuthContext';
import useStore from './stores/useStore';
import { useAuth } from '@clerk/clerk-react';

function SessionInitializer({ children }) {
  const { isLoaded, isSignedIn } = useAuth();
  const reloadHistory = useStore(state => state.reloadHistory);
  const resetForNewUser = useStore(state => state.resetForNewUser);
  const isInitialized = useStore(state => state.isInitialized);

  React.useEffect(() => {
    if (!isLoaded) return;
    if (isSignedIn) {
      // Always reload from server on sign-in (not from stale localStorage)
      reloadHistory();
    } else {
      // Clear state on sign-out so next user starts fresh
      resetForNewUser();
    }
  }, [isLoaded, isSignedIn]); // eslint-disable-line react-hooks/exhaustive-deps

  return children;
}

function App() {
  return (
    <CognitiveStoreProvider>
      <ErrorBoundary>
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
      </ErrorBoundary>
    </CognitiveStoreProvider>
  );
}

export default App;
