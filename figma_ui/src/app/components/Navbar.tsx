import { useState } from "react";
import { Link, useLocation } from "react-router";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Activity, Shield, Cpu, DollarSign, Swords, Menu, X, Layers } from "lucide-react";

const NAV_LINKS = [
  { to: "/chat", label: "Deliberation", icon: Brain },
  { to: "/debate", label: "Debate", icon: Swords },
  { to: "/mission-control", label: "Mission Control", icon: Activity },
  { to: "/governance", label: "Governance", icon: Shield },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/pricing", label: "Pricing", icon: DollarSign },
];

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 px-4 pt-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between backdrop-blur-xl bg-[rgba(6,7,8,0.80)] border border-[rgba(110,231,249,0.09)] rounded-xl px-5 py-3">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="relative w-7 h-7 rounded-lg bg-[rgba(110,231,249,0.1)] border border-[rgba(110,231,249,0.2)] flex items-center justify-center">
              <Layers className="w-3.5 h-3.5 text-[#6ee7f9]" />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-[#f3f5f7] font-semibold text-sm tracking-tight">SENTINEL</span>
              <span className="text-[#6ee7f9] text-[9px] font-medium tracking-[0.18em] uppercase">E · Runtime</span>
            </div>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden lg:flex items-center gap-0.5">
            {NAV_LINKS.map(({ to, label, icon: Icon }) => {
              const active = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all duration-200 ${
                    active
                      ? "bg-[rgba(110,231,249,0.1)] text-[#6ee7f9] border border-[rgba(110,231,249,0.18)]"
                      : "text-[#8a9099] hover:text-[#c7cbd1] hover:bg-[rgba(255,255,255,0.04)]"
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* Right side */}
          <div className="hidden lg:flex items-center gap-3">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[rgba(52,211,153,0.08)] border border-[rgba(52,211,153,0.15)]">
              <div className="w-1.5 h-1.5 rounded-full bg-[#34d399] animate-pulse" />
              <span className="text-[#34d399] text-[10px] font-medium tracking-widest uppercase">Live</span>
            </div>
            <Link
              to="/chat"
              className="px-4 py-1.5 rounded-md bg-[#6ee7f9] text-[#060708] text-xs font-semibold tracking-wide hover:bg-[rgba(110,231,249,0.85)] transition-colors"
            >
              Enter Platform
            </Link>
          </div>

          {/* Mobile toggle */}
          <button
            className="lg:hidden text-[#8a9099] hover:text-[#f3f5f7] transition-colors p-1"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18 }}
              className="mt-2 backdrop-blur-xl bg-[rgba(6,7,8,0.92)] border border-[rgba(110,231,249,0.09)] rounded-xl p-2"
            >
              {NAV_LINKS.map(({ to, label, icon: Icon }) => (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMobileOpen(false)}
                  className="flex items-center gap-2.5 px-3 py-2.5 rounded-md text-sm text-[#c7cbd1] hover:text-[#f3f5f7] hover:bg-[rgba(255,255,255,0.04)] transition-colors"
                >
                  <Icon className="w-4 h-4 text-[#6ee7f9]" />
                  {label}
                </Link>
              ))}
              <div className="mt-2 pt-2 border-t border-[rgba(110,231,249,0.08)]">
                <Link
                  to="/chat"
                  onClick={() => setMobileOpen(false)}
                  className="block text-center px-4 py-2 rounded-md bg-[#6ee7f9] text-[#060708] text-sm font-semibold"
                >
                  Enter Platform
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
