import { useState, useEffect } from "react";
import { useTheme } from "next-themes";
import { Link, useLocation } from "react-router";
import { Menu, X, Moon, Sun, User, Settings, LogOut, Shield } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { useAuthContext } from "../providers/AuthProvider";
import { useSupabaseAuth } from "../hooks/useSupabaseAuth";
import { trackLogout } from "../services/analyticsService";

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const { user, isAuthenticated, isAdmin } = useAuthContext();
  const { signOut } = useSupabaseAuth();

  useEffect(() => { setMounted(true); }, []);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isDark = theme === "dark";
  const toggleTheme = () => setTheme(isDark ? "light" : "dark");
  const isLanding = location.pathname === "/";

  const borderColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";

  const mainLinks = [
    { to: "/", label: "Home" },
    { to: "/chat", label: "Chat" },
    { to: "/engines", label: "Engines" },
    { to: "/pricing", label: "Access" },
  ];

  if (!mounted) return null;

  return (
    <div className="fixed top-5 left-1/2 -translate-x-1/2 w-[calc(100%-32px)] max-w-5xl z-50">
      <nav
        className="relative flex items-center h-[54px] px-2 rounded-full transition-all duration-500"
        style={{
          background: isDark ? "rgba(18, 18, 22, 0.5)" : "rgba(255, 255, 255, 0.4)",
          backdropFilter: "blur(32px) saturate(200%)",
          WebkitBackdropFilter: "blur(32px) saturate(200%)",
          border: isDark ? "1px solid rgba(255,255,255,0.06)" : "1px solid rgba(0,0,0,0.04)",
          boxShadow: isDark
            ? "0 12px 48px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)"
            : "0 12px 40px rgba(0,0,0,0.06), inset 0 1px 0 rgba(255,255,255,0.8)",
        }}
      >
        {/* ── LEFT: Brand logo ─────────────────────────────────────────── */}
        <Link
          to="/"
          className="group flex items-center gap-3 pl-1 pr-4 h-full rounded-full flex-shrink-0 transition-all duration-300 hover:bg-black/[0.03] dark:hover:bg-white/[0.04]"
        >
          <div
            className="relative flex items-center justify-center rounded-full overflow-hidden flex-shrink-0 transition-all duration-300 group-hover:scale-110 group-hover:-translate-y-[1px]"
            style={{
              width: "36px",
              height: "36px",
              background: isDark
                ? "radial-gradient(circle at 35% 35%, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.05) 55%, rgba(0,0,0,0.25) 100%)"
                : "radial-gradient(circle at 35% 35%, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.70) 55%, rgba(210,220,255,0.50) 100%)",
              boxShadow: isDark
                ? "inset 0 2px 4px rgba(255,255,255,0.15), inset 0 -2px 6px rgba(0,0,0,0.4), 0 0 32px rgba(120,140,255,0.35), 0 8px 32px rgba(0,0,0,0.8), 0 0 8px rgba(255,255,255,0.1)"
                : "inset 0 2px 4px rgba(255,255,255,0.9), inset 0 -2px 6px rgba(0,0,0,0.05), 0 0 24px rgba(99,102,241,0.25), 0 8px 24px rgba(0,0,0,0.15), 0 0 8px rgba(255,255,255,0.6)",
              backdropFilter: "blur(12px) saturate(180%)",
              WebkitBackdropFilter: "blur(12px) saturate(180%)",
              border: isDark ? "1px solid rgba(255,255,255,0.14)" : "1px solid rgba(255,255,255,0.85)",
            }}
          >
            <div
              className="absolute inset-0 pointer-events-none rounded-full"
              style={{ background: "radial-gradient(ellipse at 28% 22%, rgba(255,255,255,0.60) 0%, transparent 52%)" }}
            />
            <img
              src="/logo.png"
              alt="Sentinel-E"
              className="h-[18px] w-auto object-contain relative z-10"
              style={{
                filter: isDark
                  ? "drop-shadow(0 2px 8px rgba(0,0,0,0.9)) drop-shadow(0 0 18px rgba(180,200,255,0.5)) drop-shadow(0 0 4px rgba(255,255,255,0.4))"
                  : "drop-shadow(0 1px 4px rgba(0,0,0,0.3)) drop-shadow(0 0 14px rgba(99,102,241,0.3)) drop-shadow(0 0 3px rgba(255,255,255,0.8))",
              }}
            />
          </div>
          <span
            className="text-[#1d1d1f] dark:text-[#f5f5f7] hidden lg:block flex-shrink-0"
            style={{ fontFamily: "'Inter', sans-serif", fontWeight: 600, fontSize: "14.5px", letterSpacing: "-0.01em" }}
          >
            Sentinel-E
          </span>
        </Link>

        {/* ── CENTER: Cinematic Routing Buttons ── */}
        <div className="absolute left-1/2 -translate-x-1/2 hidden md:flex items-center gap-1.5">
          {mainLinks.map((link) => {
            const isActive = link.to === "/" ? location.pathname === "/" : location.pathname.startsWith(link.to);
            return (
              <Link
                key={link.to}
                to={link.to}
                className="group relative flex items-center justify-center font-medium overflow-hidden"
                style={{
                  height: "36px",
                  paddingInline: "18px",
                  borderRadius: "999px",
                  fontSize: "13.5px",
                  letterSpacing: "-0.01em",
                  color: isActive
                    ? (isDark ? "#ffffff" : "#000000")
                    : (isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)"),
                  background: isActive
                    ? (isDark ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.6)")
                    : "transparent",
                  boxShadow: isActive
                    ? (isDark
                      ? "inset 0 1px 1px rgba(255,255,255,0.15), 0 0 12px rgba(255,255,255,0.05)"
                      : "inset 0 1px 1px rgba(255,255,255,0.8), 0 2px 8px rgba(0,0,0,0.04)")
                    : "none",
                  border: isActive
                    ? (isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.04)")
                    : "1px solid transparent",
                  backdropFilter: isActive ? "blur(20px) saturate(180%)" : "none",
                  WebkitBackdropFilter: isActive ? "blur(20px) saturate(180%)" : "none",
                  transition: "all 0.3s cubic-bezier(0.16,1,0.3,1)",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.transform = "translateY(-1px) scale(1.02)";
                    e.currentTarget.style.background = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.03)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.transform = "translateY(0) scale(1)";
                    e.currentTarget.style.background = "transparent";
                  }
                }}
              >
                {isActive && (
                  <div
                    className="absolute inset-0 pointer-events-none rounded-full"
                    style={{
                      background: isDark
                        ? "radial-gradient(120% 120% at 50% 0%, rgba(255,255,255,0.15) 0%, transparent 100%)"
                        : "radial-gradient(120% 120% at 50% 0%, rgba(255,255,255,0.9) 0%, transparent 100%)",
                    }}
                  />
                )}
                <span className="relative z-10">{link.label}</span>
              </Link>
            );
          })}
        </div>

        {/* ── RIGHT: Theme toggle + Auth Links ─────────────────────────── */}
        <div className="ml-auto flex items-center gap-2 pr-1 flex-shrink-0">

          {/* Cinematic Theme Toggle Segmented Control */}
          <div
            className="flex items-center p-1 rounded-full relative overflow-hidden hidden md:flex"
            style={{
              background: isDark ? "rgba(0,0,0,0.3)" : "rgba(0,0,0,0.03)",
              border: isDark ? "1px solid rgba(255,255,255,0.05)" : "1px solid rgba(0,0,0,0.04)",
              boxShadow: isDark ? "inset 0 1px 4px rgba(0,0,0,0.5)" : "inset 0 1px 3px rgba(0,0,0,0.03)",
            }}
          >
            <button
              onClick={() => setTheme("light")}
              className="flex items-center justify-center w-7 h-7 rounded-full z-10 transition-all duration-300"
              style={{
                color: !isDark ? "#1d1d1f" : "rgba(255,255,255,0.4)",
              }}
              aria-label="Light theme"
            >
              <Sun className="w-[12px] h-[12px]" />
            </button>
            <button
              onClick={() => setTheme("dark")}
              className="flex items-center justify-center w-7 h-7 rounded-full z-10 transition-all duration-300"
              style={{
                color: isDark ? "#ffffff" : "rgba(0,0,0,0.4)",
              }}
              aria-label="Dark theme"
            >
              <Moon className="w-[12px] h-[12px]" />
            </button>

            {/* Active Pill Indicator */}
            <div
              className="absolute top-1 bottom-1 w-7 rounded-full pointer-events-none transition-transform duration-400"
              style={{
                left: "4px",
                transform: isDark ? "translateX(28px)" : "translateX(0)",
                background: isDark ? "rgba(255,255,255,0.12)" : "#ffffff",
                boxShadow: isDark
                  ? "inset 0 1px 1px rgba(255,255,255,0.1), 0 0 10px rgba(255,255,255,0.05)"
                  : "0 2px 8px rgba(0,0,0,0.08), inset 0 1px 1px rgba(255,255,255,0.8)",
                border: isDark ? "1px solid rgba(255,255,255,0.08)" : "1px solid rgba(0,0,0,0.04)",
                backdropFilter: "blur(12px) saturate(180%)",
                transitionTimingFunction: "cubic-bezier(0.16,1,0.3,1)"
              }}
            />
          </div>

          <div className="hidden md:flex items-center gap-1.5 ml-1">
            {!isAuthenticated ? (
              <>
                <Link
                  to="/login"
                  className="px-4 py-2 rounded-full text-[13px] font-medium transition-all duration-300"
                  style={{
                    color: isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = isDark ? "#ffffff" : "#000000";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)";
                  }}
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="inline-flex items-center justify-center px-4 py-2 rounded-full text-[13px] font-semibold transition-all duration-300 overflow-hidden"
                  style={{
                    letterSpacing: "-0.01em",
                    background: isDark ? "rgba(245,245,247,0.92)" : "#1d1d1f",
                    color: isDark ? "#1d1d1f" : "#f5f5f7",
                    boxShadow: isDark
                      ? "0 4px 14px rgba(255,255,255,0.15), inset 0 1px 1px rgba(255,255,255,0.8)"
                      : "0 4px 14px rgba(0,0,0,0.25), inset 0 1px 1px rgba(255,255,255,0.2)",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-1px) scale(1.02)"}
                  onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0) scale(1)"}
                >
                  Sign Up
                </Link>
              </>
            ) : (
              <>
                <Link
                  to="/profile"
                  className="p-2 rounded-full transition-all duration-300 hover:bg-black/5 dark:hover:bg-white/5"
                  title="Profile"
                  style={{ color: isDark ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.7)" }}
                >
                  <User className="w-[18px] h-[18px]" />
                </Link>
                {isAdmin && (
                  <Link
                    to="/admin"
                    className="p-2 rounded-full transition-all duration-300 hover:bg-black/5 dark:hover:bg-white/5"
                    title="Admin"
                    style={{ color: isDark ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.7)" }}
                  >
                    <Shield className="w-[18px] h-[18px]" />
                  </Link>
                )}
                <Link
                  to="/settings"
                  className="p-2 rounded-full transition-all duration-300 hover:bg-black/5 dark:hover:bg-white/5"
                  title="Settings"
                  style={{ color: isDark ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.7)" }}
                >
                  <Settings className="w-[18px] h-[18px]" />
                </Link>
                <button
                  onClick={() => {
                    if (user) trackLogout(user.id);
                    signOut();
                  }}
                  className="p-2 rounded-full transition-all duration-300 hover:bg-red-500/10"
                  title="Sign Out"
                  style={{ color: isDark ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.7)" }}
                >
                  <LogOut className="w-[18px] h-[18px]" />
                </button>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden flex items-center justify-center w-9 h-9 rounded-full ml-1"
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
            className="absolute top-[64px] left-0 right-0 p-3 rounded-3xl md:hidden"
            style={{
              background: isDark ? "rgba(12,12,16,0.94)" : "rgba(255,255,255,0.94)",
              backdropFilter: "blur(28px)",
              WebkitBackdropFilter: "blur(28px)",
              border: `1px solid ${borderColor}`,
              boxShadow: "0 24px 64px rgba(0,0,0,0.14)",
            }}
          >
            <div className="flex flex-col gap-1 mb-3">
              {mainLinks.map((link) => {
                const isActive = link.to === "/" ? location.pathname === "/" : location.pathname.startsWith(link.to);
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

              <div className="h-px w-full my-2" style={{ background: borderColor }} />

              {!isAuthenticated ? (
                <>
                  <Link
                    to="/login"
                    onClick={() => setMobileOpen(false)}
                    className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors flex items-center"
                    style={{ color: isDark ? "white" : "#1d1d1f" }}
                  >
                    Login
                  </Link>
                  <Link
                    to="/signup"
                    onClick={() => setMobileOpen(false)}
                    className="px-4 py-3 rounded-2xl text-[15px] font-semibold transition-colors flex items-center justify-center mt-2"
                    style={{
                      background: isDark ? "rgba(245,245,247,0.92)" : "#1d1d1f",
                      color: isDark ? "#1d1d1f" : "white",
                    }}
                  >
                    Sign Up
                  </Link>
                </>
              ) : (
                <>
                  <Link
                    to="/profile"
                    onClick={() => setMobileOpen(false)}
                    className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors flex items-center gap-3 w-full text-left"
                    style={{ color: isDark ? "white" : "#1d1d1f" }}
                  >
                    <User className="w-5 h-5" /> Profile
                  </Link>
                  {isAdmin && (
                    <Link
                      to="/admin"
                      onClick={() => setMobileOpen(false)}
                      className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors flex items-center gap-3"
                      style={{ color: isDark ? "white" : "#1d1d1f" }}
                    >
                      <Shield className="w-5 h-5" /> Admin
                    </Link>
                  )}
                  <Link
                    to="/settings"
                    onClick={() => setMobileOpen(false)}
                    className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors flex items-center gap-3"
                    style={{ color: isDark ? "white" : "#1d1d1f" }}
                  >
                    <Settings className="w-5 h-5" /> Settings
                  </Link>
                  <button
                    onClick={() => {
                      signOut();
                      setMobileOpen(false);
                    }}
                    className="px-4 py-3 rounded-2xl text-[15px] font-medium transition-colors flex items-center gap-3 text-red-500 w-full text-left"
                  >
                    <LogOut className="w-5 h-5" /> Sign Out
                  </button>
                </>
              )}
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
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
