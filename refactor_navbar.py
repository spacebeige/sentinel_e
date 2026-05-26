import re

with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/Navbar.tsx", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace('import { useState, useEffect, useRef } from "react";', 'import { useState, useEffect, useRef } from "react";\nimport { useTheme } from "../context/ThemeContext";')

# 2. State
old_state = """export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
  };"""
new_state = """export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { isDark, toggleTheme } = useTheme();
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);"""
content = content.replace(old_state, new_state)

# 3. Logo and Hover
old_logo = """        <Link
          to="/"
          className="flex items-center gap-2.5 pl-3 pr-4 h-full rounded-full transition-all duration-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.04]"
        >
          <img
            src="/sentinel-e.png"
            onError={(e) => {
              if (!e.currentTarget.src.endsWith("/logo.png"))
                e.currentTarget.src = "/logo.png";
            }}
            alt="Sentinel-E"
            className="h-[22px] w-auto object-contain"
            style={{ filter: "none", WebkitFilter: "none" }}
          />"""
new_logo = """        <Link
          to="/"
          className="group flex items-center gap-2.5 pl-3 pr-4 h-full rounded-full transition-all duration-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:shadow-[0_4px_16px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_24px_rgba(255,255,255,0.06)] -translate-y-0 hover:-translate-y-[1px]"
        >
          <img
            src="/sentinel-e(1).png"
            onError={(e) => {
              if (!e.currentTarget.src.endsWith("/logo.png"))
                e.currentTarget.src = "/logo.png";
            }}
            alt="Sentinel-E"
            className="h-[26px] w-auto object-contain transition-all duration-300 group-hover:scale-105"
            style={{ 
              filter: isDark ? "drop-shadow(0 0 12px rgba(255,255,255,0.15))" : "drop-shadow(0 0 8px rgba(0,0,0,0.08))",
              WebkitFilter: isDark ? "drop-shadow(0 0 12px rgba(255,255,255,0.15))" : "drop-shadow(0 0 8px rgba(0,0,0,0.08))" 
            }}
          />"""
content = content.replace(old_logo, new_logo)

# 4. Navbar links hover
old_nav = """              <Link
                key={link.to}
                to={link.to}
                className="relative px-4 py-2 rounded-full text-[13px] font-medium transition-colors duration-200"
                style={{
                  color: isActive
                    ? isDark ? "white" : "#1d1d1f"
                    : isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.4)",
                }}
              >"""
new_nav = """              <Link
                key={link.to}
                to={link.to}
                className="group relative px-4 py-2 rounded-full text-[13px] font-medium transition-all duration-300 hover:-translate-y-[1px]"
                style={{
                  color: isActive
                    ? isDark ? "white" : "#1d1d1f"
                    : isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.6)",
                }}
              >"""
content = content.replace(old_nav, new_nav)

# 5. Right actions hover
old_right = """        <div className="hidden md:flex items-center gap-1.5 pr-1.5">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full transition-colors duration-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.04]"
            style={{ color: isDark ? "rgba(255,255,255,0.45)" : "rgba(0,0,0,0.4)" }}
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          
          <Link
            to="/chat"
            className="px-4 py-1.5 rounded-full text-[13px] font-semibold transition-all duration-200 active:scale-95 hover:opacity-90"
            style={{
              background: isDark ? "white" : "#1d1d1f",
              color: isDark ? "#1d1d1f" : "white",
            }}
          >
            Initialize
          </Link>
        </div>"""
new_right = """        <div className="hidden md:flex items-center gap-1.5 pr-1.5">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full transition-all duration-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] hover:shadow-sm hover:-translate-y-[1px]"
            style={{ color: isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.6)" }}
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          
          <Link
            to="/chat"
            className="px-4 py-1.5 rounded-full text-[13px] font-semibold transition-all duration-300 hover:shadow-[0_4px_16px_rgba(0,0,0,0.12)] dark:hover:shadow-[0_4px_24px_rgba(255,255,255,0.2)] active:scale-95 hover:-translate-y-[1px]"
            style={{
              background: isDark ? "white" : "#1d1d1f",
              color: isDark ? "#1d1d1f" : "white",
            }}
          >
            Initialize
          </Link>
        </div>"""
content = content.replace(old_right, new_right)


with open("/Users/ashwinagarkhed/sentinel_e/figma_ui/src/app/components/Navbar.tsx", "w") as f:
    f.write(content)

print("Navbar updated")
