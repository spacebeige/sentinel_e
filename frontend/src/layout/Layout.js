/**
 * Layout.js — Application Shell
 * Persistent Navbar + Theme Toggle + <Outlet> + Footer (on non-chat pages).
 * The chat page uses full viewport height, so Footer is conditionally hidden.
 */
import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import { Footer } from '../figma_features/Footer';

export default function Layout() {
  const location = useLocation();
  const isChatPage = location.pathname === '/chat';

  return (
    <div className="min-h-screen flex flex-col bg-[#f5f5f7] dark:bg-[#0f0f10] transition-colors duration-300">
      <Navbar />
      <main className={isChatPage ? 'flex-1 pt-14' : 'flex-1'}>
        <Outlet />
      </main>
      {!isChatPage && <Footer />}
    </div>
  );
}
