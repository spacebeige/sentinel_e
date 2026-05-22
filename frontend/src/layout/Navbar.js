/**
 * Navbar.js — Application Shell Navigation
 * Modal-first auth integrated into the existing design system.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, Moon, Shield, Sun, X } from 'lucide-react';
import { useAuthContext } from '../hooks/useAuthContext';
import { getCurrentTheme, persistTheme, subscribeThemeChanges } from '../services/themeManager';
import SentinelIdentity from '../components/SentinelIdentity';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

const navLinks = [
  { to: '/', label: 'Home', protected: false },
  { to: '/chat', label: 'Chat', protected: true },
  { to: '/models', label: 'Models', protected: true },
  { to: '/pricing', label: 'Pricing', protected: false },
];

export default function Navbar() {
  const location = useLocation();
  const {
    isAdmin,
    isAuthenticated,
    loading,
    openAuthModal,
    signOut,
    user,
  } = useAuthContext();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [dark, setDark] = useState(() => getCurrentTheme() === 'dark');

  useEffect(() => subscribeThemeChanges((theme) => setDark(theme === 'dark')), []);

  const toggleTheme = useCallback(() => {
    const nextTheme = dark ? 'light' : 'dark';
    setDark(nextTheme === 'dark');
    persistTheme(nextTheme);
  }, [dark]);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const isChat = location.pathname === '/chat';
  const displayName = user?.user_metadata?.full_name
    || user?.user_metadata?.name
    || user?.email?.split('@')[0]
    || 'User';
  const displayEmail = user?.email || '';
  const sessionLabel = displayEmail || 'Authenticated session';

  const handleProtectedNavigation = (event, targetPath, isProtected) => {
    if (loading) {
      event.preventDefault();
      return;
    }

    if (isProtected && !isAuthenticated) {
      event.preventDefault();
      openAuthModal({ returnTo: targetPath });
    }
  };

  const renderNavLink = (link, mobile = false) => {
    const isActive = location.pathname === link.to;
    const baseClass = mobile
      ? `block px-4 py-2.5 rounded-xl mb-1 transition-all ${
          isActive
            ? 'sentinel-nav-active'
            : 'sentinel-nav-muted hover:bg-black/5 dark:hover:bg-white/10'
        }`
      : `px-4 py-1.5 rounded-full transition-all ${
          isActive
            ? 'sentinel-nav-active'
            : 'sentinel-nav-muted hover:bg-black/5 dark:hover:bg-white/10'
        }`;

    return (
      <Link
        key={link.to}
        to={link.to}
        onClick={(event) => handleProtectedNavigation(event, link.to, link.protected)}
        className={baseClass}
        style={{ fontFamily: FONT, fontSize: mobile ? '15px' : '14px', fontWeight: 500 }}
      >
        {link.label}
      </Link>
    );
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-xl border-b transition-colors duration-300 ${
        'sentinel-nav-shell'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <SentinelIdentity size={36} />
          <div>
            <span
              className="sentinel-text-primary"
              style={{ fontFamily: FONT, fontWeight: 650, fontSize: '18px', letterSpacing: '-0.02em' }}
            >
              Sentinel-E
            </span>
            <div
              className="sentinel-text-muted"
              style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500 }}
            >
              Multi-model reasoning
            </div>
          </div>
        </Link>

        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => renderNavLink(link))}

          {isAdmin && (
            <Link
              to="/admin"
              className={`px-4 py-1.5 rounded-full transition-all flex items-center gap-1.5 ${
                location.pathname === '/admin'
                  ? 'bg-[#1c1c1e]/10 dark:bg-white/10 text-[#1c1c1e] dark:text-white'
                  : 'text-[#6e6e73] dark:text-[#94a3b8] hover:text-[#1c1c1e] dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/5'
              }`}
              style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500 }}
            >
              <Shield className="w-4 h-4" />
              Admin
            </Link>
          )}
        </div>

        <div className="hidden md:flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="px-3 py-2.5 rounded-2xl transition-all sentinel-icon-button sentinel-theme-toggle flex items-center gap-2"
            title={dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-pressed={dark}
          >
            {dark ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
            <span style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600 }}>
              {dark ? 'Light' : 'Dark'}
            </span>
          </button>

          {loading ? (
            <div className="px-4 py-2 rounded-full sentinel-surface-panel sentinel-text-muted" style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 600 }}>
              Restoring session…
            </div>
          ) : isAuthenticated ? (
            <>
              {!isChat && (
                <Link
                  to="/chat"
                  className="px-5 py-2 rounded-full sentinel-glass-button sentinel-glass-button--primary transition-all"
                  style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                >
                  Open Chat
                </Link>
              )}
              <div className="flex items-center gap-3 px-3 py-2 rounded-2xl border sentinel-surface-panel">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#2d2d2f] to-[#1d1d1f] flex items-center justify-center text-white text-sm font-semibold">
                  {displayName.charAt(0).toUpperCase()}
                </div>
                <div className="max-w-[160px]">
                  <div className="truncate sentinel-text-primary" style={{ fontFamily: FONT, fontSize: '13px', fontWeight: 600 }}>
                    {displayName}
                  </div>
                  <div className="sentinel-text-muted" style={{ fontFamily: FONT, fontSize: '11px', fontWeight: 500 }}>
                    {sessionLabel}
                  </div>
                </div>
              </div>
              <button
                onClick={signOut}
                className="px-4 py-2 rounded-full transition-all sentinel-nav-muted hover:bg-black/5 dark:hover:bg-white/10"
                style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
              >
                Sign out
              </button>
            </>
          ) : (
            <button
              onClick={() => openAuthModal({ returnTo: '/chat' })}
              className="px-5 py-2 rounded-full sentinel-glass-button sentinel-glass-button--primary transition-all"
              style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
            >
              Sign in
            </button>
          )}
        </div>

        <div className="md:hidden flex items-center gap-1">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl sentinel-icon-button sentinel-theme-toggle"
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-pressed={dark}
          >
            {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button
            className="p-2 rounded-xl sentinel-icon-button"
            onClick={() => setMobileOpen((prev) => !prev)}
            aria-label="Toggle navigation menu"
          >
            {mobileOpen
              ? <X className="w-5 h-5" />
              : <Menu className="w-5 h-5" />
            }
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div className="md:hidden backdrop-blur-xl border-b px-6 pb-5 sentinel-nav-shell">
          <div className="pt-2">
            {navLinks.map((link) => renderNavLink(link, true))}

            {isAdmin && (
              <Link
                to="/admin"
                className={`px-4 py-2.5 rounded-xl mb-1 transition-all flex items-center gap-2 ${
                  location.pathname === '/admin'
                    ? 'bg-[#1c1c1e]/10 dark:bg-white/10 text-[#1c1c1e] dark:text-white'
                    : 'text-[#6e6e73] dark:text-[#94a3b8] hover:bg-black/5 dark:hover:bg-white/5'
                }`}
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
              >
                <Shield className="w-4 h-4" />
                Admin Dashboard
              </Link>
            )}
          </div>

          <div className="mt-3 p-3 rounded-2xl border sentinel-surface-panel">
            {loading ? (
              <div className="text-center text-xs sentinel-text-muted" style={{ fontFamily: FONT, fontWeight: 600 }}>
                Restoring session…
              </div>
            ) : isAuthenticated ? (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#2d2d2f] to-[#1d1d1f] flex items-center justify-center text-white text-sm font-semibold">
                    {displayName.charAt(0).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div
                      className="truncate sentinel-text-primary"
                      style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                    >
                      {displayName}
                    </div>
                    <div
                      className="sentinel-text-muted"
                      style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}
                    >
                      {sessionLabel}
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Link
                    to="/chat"
                    className="flex-1 text-center px-4 py-2.5 rounded-xl sentinel-glass-button sentinel-glass-button--primary"
                    style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                  >
                    Open Chat
                  </Link>
                  <button
                    onClick={signOut}
                    className="flex-1 px-4 py-2.5 rounded-xl border sentinel-nav-muted sentinel-border hover:bg-black/5 dark:hover:bg-white/10"
                    style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                  >
                    Sign out
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => openAuthModal({ returnTo: '/chat' })}
                className="block text-center w-full px-5 py-2.5 rounded-xl sentinel-glass-button sentinel-glass-button--primary"
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 600 }}
              >
                Sign in
              </button>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
