/**
 * Navbar.js — Application Shell Navigation
 * Uses react-router-dom for SPA navigation.
 * Includes dark/light theme toggle with localStorage persistence.
 */
import React, { useState, useEffect, useCallback } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Sigma, Sun, Moon } from 'lucide-react';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

const navLinks = [
  { to: '/', label: 'Home' },
  { to: '/chat', label: 'Chat' },
  { to: '/models', label: 'Models' },
  { to: '/pricing', label: 'Pricing' },
];

export default function Navbar() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false;
    const stored = localStorage.getItem('sentinel-theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Apply theme class to <html> and persist
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

  const toggleTheme = useCallback(() => setDark(d => !d), []);

  // Close mobile menu on route change
  useEffect(() => { setMobileOpen(false); }, [location.pathname]);

  const isChat = location.pathname === '/chat';

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-xl border-b transition-colors duration-300 ${
        dark
          ? 'bg-[#1d1d1f]/80 border-white/10'
          : 'bg-white/70 border-white/20'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
            <Sigma className="w-4 h-4 text-white" />
          </div>
          <span
            className={`tracking-tight ${dark ? 'text-white' : 'text-[#1d1d1f]'}`}
            style={{ fontFamily: FONT, fontWeight: 600, fontSize: '18px' }}
          >
            Sentinel-E
          </span>
        </Link>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                className={`px-4 py-1.5 rounded-full transition-all ${
                  isActive
                    ? dark ? 'bg-white text-[#1d1d1f]' : 'bg-[#1d1d1f] text-white'
                    : dark ? 'text-white/60 hover:text-white hover:bg-white/10' : 'text-[#6e6e73] hover:text-[#1d1d1f] hover:bg-black/5'
                }`}
                style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500 }}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Right side: Theme Toggle + CTA */}
        <div className="hidden md:flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-xl transition-all ${
              dark ? 'hover:bg-white/10 text-white/70 hover:text-white' : 'hover:bg-black/5 text-[#6e6e73] hover:text-[#1d1d1f]'
            }`}
            title={dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
          </button>

          {!isChat && (
            <Link
              to="/chat"
              className="px-5 py-1.5 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white transition-all hover:opacity-90 shadow-lg shadow-blue-500/25"
              style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500 }}
            >
              Try Free
            </Link>
          )}
        </div>

        {/* Mobile: Theme Toggle + Hamburger */}
        <div className="md:hidden flex items-center gap-1">
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-xl ${dark ? 'text-white/70' : 'text-[#6e6e73]'}`}
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button
            className={`p-2 rounded-xl ${dark ? 'hover:bg-white/10' : 'hover:bg-black/5'}`}
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen
              ? <X className={`w-5 h-5 ${dark ? 'text-white' : ''}`} />
              : <Menu className={`w-5 h-5 ${dark ? 'text-white' : ''}`} />
            }
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className={`md:hidden backdrop-blur-xl border-b px-6 pb-4 ${
          dark ? 'bg-[#1d1d1f]/90 border-white/10' : 'bg-white/90 border-white/20'
        }`}>
          {navLinks.map((link) => {
            const isActive = location.pathname === link.to;
            return (
              <Link
                key={link.to}
                to={link.to}
                className={`block px-4 py-2.5 rounded-xl mb-1 transition-all ${
                  isActive
                    ? dark ? 'bg-white text-[#1d1d1f]' : 'bg-[#1d1d1f] text-white'
                    : dark ? 'text-white/60 hover:bg-white/10' : 'text-[#6e6e73] hover:bg-black/5'
                }`}
                style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
              >
                {link.label}
              </Link>
            );
          })}
          <Link
            to="/chat"
            className="block text-center mt-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white"
            style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
          >
            Try Free
          </Link>
        </div>
      )}
    </nav>
  );
}
