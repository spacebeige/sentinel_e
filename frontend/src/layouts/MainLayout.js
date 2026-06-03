/**
 * ============================================================
 * Main Layout Component
 * ============================================================
 *
 * Provides consistent layout structure across all pages:
 * - Header with user info and settings
 * - Sidebar with session history
 * - Main content area
 * - Footer
 */

import React, { useState } from 'react';
import { useAuthContext } from '../hooks/useAuthContext';
import SessionSidebar from '../components/SessionSidebar';
import '../styles/MainLayout.css';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

export const MainLayout = ({ children, showSidebar = true }) => {
  const { user, isAdmin, signOut } = useAuthContext();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="main-layout">
      {/* Header */}
      <header className="main-header">
        <div className="header-content">
          <div className="header-left">
            <button
              className="sidebar-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            >
              {sidebarOpen ? '✕' : '☰'}
            </button>
            <div className="logo-section">
              <h1 style={{ fontFamily: FONT, fontSize: '24px', fontWeight: 700, margin: 0 }}>
                Sentinel-E
              </h1>
              {isAdmin && (
                <span className="admin-badge" style={{ fontFamily: FONT, fontSize: '11px' }}>
                  ADMIN
                </span>
              )}
            </div>
          </div>

          <div className="header-right">
            <div className="user-info" style={{ fontFamily: FONT, fontSize: '13px' }}>
              <span className="user-name">{user?.displayName || user?.email}</span>
              <span className="user-role">{isAdmin ? 'Administrator' : 'User'}</span>
            </div>
            <button
              className="logout-button"
              onClick={signOut}
              style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600 }}
            >
              Sign Out
            </button>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar - Session History */}
        {showSidebar && <SessionSidebar isOpen={sidebarOpen} />}

        {/* Main Content */}
        <main className={`main-content ${!showSidebar ? 'full-width' : ''}`}>
          {children}
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
