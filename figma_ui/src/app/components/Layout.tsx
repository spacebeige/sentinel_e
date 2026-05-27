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
      className="min-h-screen bg-[#f5f5f7] text-[#1d1d1f] dark:bg-[#09090b] dark:text-[#f5f5f7] transition-colors duration-500"
      style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif" }}
    >
      {location.pathname === "/" && <Navbar />}
      <Outlet />
    </div>
  );
}

