import { Outlet } from "react-router";
import { Navbar } from "./Navbar";
import { NeuralBackground } from "./NeuralBackground";

export function Layout() {
  return (
    <div className="min-h-screen bg-[#060708] font-sans text-[#f3f5f7] relative overflow-x-hidden">
      <NeuralBackground />
      <Navbar />
      <div className="relative z-10">
        <Outlet />
      </div>
    </div>
  );
}
