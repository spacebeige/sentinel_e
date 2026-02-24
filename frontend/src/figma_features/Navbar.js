/**
 * Navbar — Figma Feature Module (ported from figma_ui)
 * Fixed top nav with mobile menu. Uses <a> instead of react-router Link.
 * Standalone — not wired into the chat engine.
 */
import React, { useState } from 'react';
import { Menu, X, Sigma } from 'lucide-react';

const FONT = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";

const links = [
  { to: '/', label: 'Home' },
  { to: '/chat', label: 'Chat' },
  { to: '/models', label: 'Models' },
  { to: '/pricing', label: 'Pricing' },
];

export function Navbar({ activePath = '/' }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-white/70 border-b border-white/20">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <a href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
            <Sigma className="w-4 h-4 text-white" />
          </div>
          <span className="text-[#1d1d1f] tracking-tight"
            style={{ fontFamily: FONT, fontWeight: 600, fontSize: '18px' }}>
            Sentinel-E
          </span>
        </a>

        <div className="hidden md:flex items-center gap-1">
          {links.map((link) => (
            <a
              key={link.to}
              href={link.to}
              className={`px-4 py-1.5 rounded-full transition-all ${
                activePath === link.to
                  ? 'bg-[#1d1d1f] text-white'
                  : 'text-[#6e6e73] hover:text-[#1d1d1f] hover:bg-black/5'
              }`}
              style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500 }}
            >
              {link.label}
            </a>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-3">
          <a
            href="/chat"
            className="px-5 py-1.5 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white transition-all hover:opacity-90 shadow-lg shadow-blue-500/25"
            style={{ fontFamily: FONT, fontSize: '14px', fontWeight: 500 }}
          >
            Try Free
          </a>
        </div>

        <button className="md:hidden p-2 rounded-xl hover:bg-black/5"
          onClick={() => setMobileOpen(!mobileOpen)}>
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden backdrop-blur-xl bg-white/90 border-b border-white/20 px-6 pb-4">
          {links.map((link) => (
            <a
              key={link.to}
              href={link.to}
              onClick={() => setMobileOpen(false)}
              className={`block px-4 py-2.5 rounded-xl mb-1 transition-all ${
                activePath === link.to
                  ? 'bg-[#1d1d1f] text-white'
                  : 'text-[#6e6e73] hover:bg-black/5'
              }`}
              style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
            >
              {link.label}
            </a>
          ))}
          <a
            href="/chat"
            onClick={() => setMobileOpen(false)}
            className="block text-center mt-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white"
            style={{ fontFamily: FONT, fontSize: '15px', fontWeight: 500 }}
          >
            Try Free
          </a>
        </div>
      )}
    </nav>
  );
}

export default Navbar;
