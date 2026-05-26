import { Outlet } from "react-router";
import { Navbar } from "./Navbar";

export function Layout() {
  return (
    <div
      className="min-h-screen bg-white dark:bg-[#0a0d12] transition-colors duration-500"
      style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif" }}
    >
      <Navbar />
      <Outlet />
    </div>
  );
}
