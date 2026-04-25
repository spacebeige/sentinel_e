/**
 * Navbar.js — Application Shell Navigation
 * Modal-first auth integrated into the existing design system.
 */
import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, Moon, Shield, Sigma, Sun, X } from 'lucide-react';
import { UserButton } from '@clerk/clerk-react';
import { useAuthContext } from '../hooks/useAuthContext';

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
    user,
  } = useAuthContext();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return true;
    const stored = localStorage.getItem('sentinel-theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (dark) {
      root.classList.add('dark');
      root.setAttribute('data-theme', 'dark');
    } else {
      root.classList.remove('dark');
      root.setAttribute('data-theme', 'light');
    }
    localStorage.setItem('sentinel-theme', dark ? 'dark' : 'light');
  }, [dark]);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  const isChat = location.pathname === '/chat';
  const displayName = user?.name || user?.email?.split('@')[0] || 'Sentinel User';

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
            ? dark ? 'bg-white text-[#1d1d1f]' : 'bg-[#1d1d1f] text-white'
            : dark ? 'text-white/60 hover:bg-white/10' : 'text-[#6e6e73] hover:bg-black/5'
        }`
      : `px-4 py-1.5 rounded-full transition-all ${
          isActive
            ? dark ? 'bg-white text-[#1d1d1f]' : 'bg-[#1d1d1f] text-white'
            : dark ? 'text-white/60 hover:text-white hover:bg-white/10' : 'text-[#6e6e73] hover:text-[#1d1d1f] hover:bg-black/5'
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
        dark
          ? 'bg-[#0a0f1a]/78 border-white/10'
          : 'bg-white/72 border-white/20'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Sigma className="w-4.5 h-4.5 text-white" />
          </div>
          <div>
            <span
              className={dark ? 'text-white' : 'text-[#0f172a]'}
              style={{ fontFamily: FONT, fontWeight: 650, fontSize: '18px', letterSpacing: '-0.02em' }}
            >
              Sentinel-E
            </span>
            <div
              className={dark ? 'text-white/45' : 'text-[#64748b]'}
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
                  ? dark ? 'bg-cyan-400/20 text-cyan-200' : 'bg-cyan-100 text-cyan-700'
                  : dark ? 'text-cyan-300/70 hover:text-cyan-200 hover:bg-cyan-400/10' : 'text-cyan-700 hover:bg-cyan-50'
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
            onClick={() => setDark((prev) => !prev)}
            className={`p-2.5 rounded-2xl transition-all ${
              dark ? 'hover:bg-white/10 text-white/70 hover:text-white' : 'hover:bg-black/5 text-[#6e6e73] hover:text-[#1d1d1f]'
            }`}
            title={dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
          </button>

          {isAuthenticated ? (
            <>
              {!isChat && (
                <Link
                  to="/chat"
                  className="px-5 py-2 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white transition-all hover:opacity-95 shadow-lg shadow-cyan-500/20"
                  style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                >
                  Open Chat
                </Link>
              )}

              <div
                className={`flex items-center justify-center p-1 rounded-full border ${
                  dark ? 'bg-white/6 border-white/10 text-white' : 'bg-white border-black/5 text-[#0f172a]'
                }`}
              >
                <UserButton appearance={{ elements: { userButtonAvatarBox: "w-9 h-9" } }} />
              </div>
            </>
          ) : (
            <button
              onClick={() => openAuthModal({ returnTo: '/chat' })}
              className="px-5 py-2 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white transition-all hover:opacity-95 shadow-lg shadow-cyan-500/20"
              style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
            >
              Login / Sign Up
            </button>
          )}
        </div>

        <div className="md:hidden flex items-center gap-1">
          <button
            onClick={() => setDark((prev) => !prev)}
            className={`p-2 rounded-xl ${dark ? 'text-white/70' : 'text-[#6e6e73]'}`}
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button
            className={`p-2 rounded-xl ${dark ? 'hover:bg-white/10' : 'hover:bg-black/5'}`}
            onClick={() => setMobileOpen((prev) => !prev)}
            aria-label="Toggle navigation menu"
          >
            {mobileOpen
              ? <X className={`w-5 h-5 ${dark ? 'text-white' : ''}`} />
              : <Menu className={`w-5 h-5 ${dark ? 'text-white' : ''}`} />
            }
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div className={`md:hidden backdrop-blur-xl border-b px-6 pb-5 ${
          dark ? 'bg-[#0a0f1a]/92 border-white/10' : 'bg-white/92 border-white/20'
        }`}>
          <div className="pt-2">
            {navLinks.map((link) => renderNavLink(link, true))}

            {isAdmin && (
              <Link
                to="/admin"
                className={`px-4 py-2.5 rounded-xl mb-1 transition-all flex items-center gap-2 ${
                  location.pathname === '/admin'
                    ? dark ? 'bg-cyan-400/20 text-cyan-200' : 'bg-cyan-100 text-cyan-700'
                    : dark ? 'text-cyan-300/70 hover:bg-cyan-400/10' : 'text-cyan-700 hover:bg-cyan-50'
                }`}
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
              >
                <Shield className="w-4 h-4" />
                Admin Dashboard
              </Link>
            )}
          </div>

          <div className={`mt-3 p-3 rounded-2xl border ${dark ? 'bg-white/6 border-white/10' : 'bg-[#f8fafc] border-black/5'}`}>
            {isAuthenticated ? (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <UserButton appearance={{ elements: { userButtonAvatarBox: "w-9 h-9" } }} />
                  <div className="min-w-0">
                    <div
                      className={`truncate ${dark ? 'text-white' : 'text-[#0f172a]'}`}
                      style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                    >
                      {displayName}
                    </div>
                    <div
                      className={dark ? 'text-white/45' : 'text-[#64748b]'}
                      style={{ fontFamily: FONT, fontSize: '12px', fontWeight: 500 }}
                    >
                      {user?.provider ? `${user.provider} session` : 'Authenticated'}
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Link
                    to="/chat"
                    className="flex-1 text-center px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white"
                    style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 600 }}
                  >
                    Open Chat
                  </Link>
                </div>
              </div>
            ) : (
              <button
                onClick={() => openAuthModal({ returnTo: '/chat' })}
                className="block text-center w-full px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white"
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 600 }}
              >
                Login / Sign Up
              </button>
            )}
          </div>
        </div>
      )}
    </nav>
  );
}
