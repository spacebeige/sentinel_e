import { useState } from "react";
import { Link, useLocation } from "react-router";
import { Menu, X, Sigma } from "lucide-react";

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const links = [
    { to: "/", label: "Home" },
    { to: "/chat", label: "Chat" },
    { to: "/models", label: "Models" },
    { to: "/pricing", label: "Pricing" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-white/70 border-b border-white/20">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#06b6d4] flex items-center justify-center">
            <Sigma className="w-4 h-4 text-white" />
          </div>
          <span className="text-[#1d1d1f] tracking-tight" style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif", fontWeight: 600, fontSize: '18px' }}>
            Sentinel-E
          </span>
        </Link>

        <div className="hidden md:flex items-center gap-1">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-4 py-1.5 rounded-full transition-all ${
                location.pathname === link.to
                  ? "bg-[#1d1d1f] text-white"
                  : "text-[#6e6e73] hover:text-[#1d1d1f] hover:bg-black/5"
              }`}
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 500 }}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-3">
          <Link
            to="/chat"
            className="px-5 py-1.5 rounded-full bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white transition-all hover:opacity-90 shadow-lg shadow-blue-500/25"
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '14px', fontWeight: 500 }}
          >
            Try Free
          </Link>
        </div>

        <button
          className="md:hidden p-2 rounded-xl hover:bg-black/5"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden backdrop-blur-xl bg-white/90 border-b border-white/20 px-6 pb-4">
          {links.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              onClick={() => setMobileOpen(false)}
              className={`block px-4 py-2.5 rounded-xl mb-1 transition-all ${
                location.pathname === link.to
                  ? "bg-[#1d1d1f] text-white"
                  : "text-[#6e6e73] hover:bg-black/5"
              }`}
              style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '15px', fontWeight: 500 }}
            >
              {link.label}
            </Link>
          ))}
          <Link
            to="/chat"
            onClick={() => setMobileOpen(false)}
            className="block text-center mt-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#06b6d4] text-white"
            style={{ fontFamily: "'Inter', -apple-system, sans-serif", fontSize: '15px', fontWeight: 500 }}
          >
            Try Free
          </Link>
        </div>
      )}
    </nav>
  );
}