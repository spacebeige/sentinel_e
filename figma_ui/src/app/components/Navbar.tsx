import { useState, useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import { Link, useLocation } from "react-router";
import { Menu, X, Moon, Sun } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);
  const isDark = theme === "dark";

  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const navLinks = [
    { to: "/", label: "Home" },
    { to: "/chat", label: "Chat" },
    { to: "/engines", label: "Engines" },
    { to: "/pricing", label: "Access" },
  ];

  const glassBase = isDark
    ? "rgba(12,12,16,0.72)"
    : scrolled
    ? "rgba(255,255,255,0.72)"
    : "rgba(255,255,255,0.50)";

  const borderColor = isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.07)";
  const shadow = scrolled
    ? isDark
      ? "0 8px 32px rgba(0,0,0,0.5)"
      : "0 8px 32px rgba(0,0,0,0.08)"
    : "none";

  if (!mounted) return null;
  return (
    <div className="fixed top-5 left-1/2 -translate-x-1/2 w-[calc(100%-32px)] max-w-4xl z-50">
      <nav
        className="relative flex items-center justify-between h-[52px] px-2 rounded-full transition-all duration-500"
        style={{
          background: glassBase,
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          border: `1px solid ${borderColor}`,
          boxShadow: shadow,
        }}
      >
        {/* LEFT — Brand */}
        <Link
          to="/"
          className="group flex items-center gap-2.5 pl-3 pr-4 h-full rounded-full transition-all duration-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:shadow-[0_4px_16px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_24px_rgba(255,255,255,0.06)] -translate-y-0 hover:-translate-y-[1px]"
        >
          <div
            className="relative flex items-center justify-center rounded-full overflow-hidden transition-all duration-300 group-hover:scale-105"
            style={{
              width: "30px",
              height: "30px",
              background: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.04)",
              boxShadow: isDark
                ? "inset 0 1px 4px rgba(255,255,255,0.2), 0 0 16px rgba(255,255,255,0.15), 0 4px 12px rgba(0,0,0,0.5)"
                : "inset 0 1px 4px rgba(255,255,255,0.8), 0 0 12px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.1)",
              backdropFilter: "blur(8px)",
              WebkitBackdropFilter: "blur(8px)",
            }}
          >
            <div
              className="absolute inset-0 opacity-50 mix-blend-overlay pointer-events-none"
              style={{
                background: "radial-gradient(circle at top left, rgba(255,255,255,0.4) 0%, transparent 60%)",
              }}
            />
            <img
              src="/logo.png"
              alt="Sentinel-E"
              className="h-[16px] w-auto object-contain relative z-10"
              style={{ filter: isDark ? "drop-shadow(0 2px 4px rgba(0,0,0,0.5))" : "drop-shadow(0 1px 2px rgba(0,0,0,0.2))" }}
            />
          </div>
          <span
            className="text-[#1d1d1f] dark:text-[#f5f5f7] hidden sm:block"
            style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: "14px", letterSpacing: "-0.01em" }}
          >
            Sentinel-E
          </span>
        </Link>

        {/* CENTER — Navigation */}
        <div className="hidden md:flex items-center gap-2 px-1">
          {location.pathname === "/" && (
            <>
              {[
                { to: "/chat", label: "Chat" },
                { to: "/engines", label: "Engines" },
                { to: "/pricing", label: "Access" },
              ].map(link => (
                <Link
                  key={link.to}
                  to={link.to}
                  className="inline-flex items-center justify-center px-5 py-2 rounded-full text-[13px] font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] bg-[#1d1d1f] text-[#f5f5f7] dark:bg-[#f5f5f7] dark:text-[#1d1d1f]"
                  style={{ letterSpacing: "-0.01em" }}
                >
                  {link.label}
                </Link>
              ))}
            </>
          )}
        </div>

        {/* RIGHT — Actions */}
        <div className="flex items-center gap-2 pr-1">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="flex items-center justify-center w-9 h-9 rounded-full transition-colors duration-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.07]"
            title="Toggle theme"
            aria-label="Toggle theme"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={isDark ? "sun" : "moon"}
                initial={{ opacity: 0, rotate: -90, scale: 0.8 }}
                animate={{ opacity: 1, rotate: 0, scale: 1 }}
                exit={{ opacity: 0, rotate: 90, scale: 0.8 }}
                transition={{ duration: 0.2 }}
              >
                {isDark
                  ? <Sun className="w-[15px] h-[15px] text-[rgba(255,255,255,0.55)]" />
                  : <Moon className="w-[15px] h-[15px] text-[rgba(0,0,0,0.4)]" />
                }
              </motion.div>
            </AnimatePresence>
          </button>

          {/* Initialize CTA */}
          <Link
            to="/chat"
            className="inline-flex items-center justify-center px-5 py-2 rounded-full text-[13px] font-semibold transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] bg-[#1d1d1f] text-[#f5f5f7] dark:bg-[#f5f5f7] dark:text-[#1d1d1f]"
            style={{
              letterSpacing: "-0.01em",
            }}
          >
            Initialize
          </Link>

          {/* Mobile menu button */}
          <button
            className="md:hidden flex items-center justify-center w-9 h-9 rounded-full hover:bg-black/[0.04] dark:hover:bg-white/[0.06]"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Menu"
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={mobileOpen ? "x" : "menu"}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ duration: 0.15 }}
              >
                {mobileOpen
                  ? <X className="w-5 h-5 text-[#1d1d1f] dark:text-[#f5f5f7]" />
                  : <Menu className="w-5 h-5 text-[#1d1d1f] dark:text-[#f5f5f7]" />
                }
              </motion.div>
            </AnimatePresence>
          </button>
        </div>
      </nav>

      {/* Mobile Sheet */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="absolute top-[60px] left-0 right-0 p-3 rounded-3xl md:hidden"
            style={{
              background: isDark ? "rgba(12,12,16,0.92)" : "rgba(255,255,255,0.92)",
              backdropFilter: "blur(28px)",
              WebkitBackdropFilter: "blur(28px)",
              border: `1px solid ${borderColor}`,
              boxShadow: "0 24px 64px rgba(0,0,0,0.14)",
            }}
          >
            <div className="flex flex-col gap-1 mb-3">
              {navLinks.map((link) => {
                const isActive =
                  link.to === "/" ? location.pathname === "/" : location.pathname.startsWith(link.to);
                return (
                  <Link
                    key={link.to}
                    to={link.to}
                    onClick={() => setMobileOpen(false)}
                    className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors"
                    style={{
                      color: isDark ? "white" : "#1d1d1f",
                      background: isActive
                        ? isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.04)"
                        : "transparent",
                    }}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </div>
            <div className="flex gap-2 pt-2 border-t" style={{ borderColor }}>
              <button
                onClick={toggleTheme}
                className="flex items-center justify-center gap-2 flex-1 py-2.5 rounded-2xl text-[14px] font-medium"
                style={{
                  background: isDark ? "rgba(255,255,255,0.07)" : "rgba(0,0,0,0.04)",
                  color: isDark ? "white" : "#1d1d1f",
                }}
              >
                {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                {isDark ? "Light mode" : "Dark mode"}
              </button>
              <Link
                to="/chat"
                onClick={() => setMobileOpen(false)}
                className="flex items-center justify-center flex-1 py-2.5 rounded-2xl text-[14px] font-semibold"
                style={{
                  background: isDark ? "white" : "#1d1d1f",
                  color: isDark ? "#1d1d1f" : "white",
                }}
              >
                Initialize
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}