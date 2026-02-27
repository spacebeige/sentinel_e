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

function App() {
  return (
    <CognitiveStoreProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<LandingPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/pricing" element={<PricingPageWrapper />} />
            <Route path="/models" element={<ModelsPageWrapper />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </CognitiveStoreProvider>
  );
}

export default App;
