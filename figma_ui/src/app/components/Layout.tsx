import { useTheme } from "next-themes";
import { Outlet, useLocation } from "react-router";
import { Navbar } from "./Navbar";
import { useLenis } from "../hooks/useLenis";

export function Layout() {
  const location = useLocation();
  const isChat = location.pathname.startsWith("/chat");

  // Cinematic smooth scroll — disabled on /chat (has its own scroll container)
  useLenis(!isChat);

  return (
    <div
      className="min-h-screen bg-white dark:bg-[#0a0d12] transition-colors duration-500"
      style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif" }}
    >
      {/* Navbar hidden on /chat — chat has its own floating semantic topbar */}
      {!isChat && <Navbar />}
      <Outlet />
    </div>
  );
}

