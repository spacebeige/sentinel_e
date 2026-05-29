import React, { Suspense } from "react";
import { createBrowserRouter, Outlet, useRouteError } from "react-router";
import { Layout } from "./components/Layout";
import HomePage from "./components/HomePage";
import { ChatPage } from "./components/ChatPage";
import { EnginesPage } from "./components/EnginesPage";
import { PricingPage } from "./components/PricingPage";
import { AuthProvider } from "./providers/AuthProvider";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { CinematicErrorBoundary } from "./components/CinematicErrorBoundary";
import LoginPage from "./components/LoginPage";
import SignupPage from "./components/SignupPage";
import AuthCallbackPage from "./components/AuthCallbackPage";
import AdminPage from "./components/AdminPage";
import ForgotPasswordPage from "./components/ForgotPasswordPage";
import ResetPasswordPage from "./components/ResetPasswordPage";
import CompleteProfilePage from "./components/CompleteProfilePage";

const OrchestrationGame = React.lazy(() => import("./orchestration/game/OrchestrationGame"));
const TetrisGame = React.lazy(() => import("./tetris/game/TetrisGame"));

const AuthRoot = () => (
  <AuthProvider>
    <Outlet />
  </AuthProvider>
);

const SuspenseWrapper = ({ children }: { children: React.ReactNode }) => (
  <Suspense fallback={<div className="flex h-screen items-center justify-center bg-black text-white/50 tracking-widest uppercase">Loading Application Environment...</div>}>
    {children}
  </Suspense>
);

export const router = createBrowserRouter([
  {
    element: <AuthRoot />,
    errorElement: (
      <CinematicErrorBoundary>
        <div className="min-h-screen bg-black" />
      </CinematicErrorBoundary>
    ),
    children: [
      {
        path: "/",
        element: <Layout />,
        children: [
          { index: true, element: <HomePage /> },
          { path: "chat", element: <ProtectedRoute><ChatPage /></ProtectedRoute> },
          { path: "engines", element: <ProtectedRoute><EnginesPage /></ProtectedRoute> },
          { path: "models", element: <ProtectedRoute><EnginesPage /></ProtectedRoute> },
          { path: "pricing", element: <PricingPage /> },
          { path: "admin", element: <ProtectedRoute requireAdmin><AdminPage /></ProtectedRoute> },
        ],
      },
      { path: "/login", element: <LoginPage /> },
      { path: "/signup", element: <SignupPage /> },
      { path: "/forgot-password", element: <ForgotPasswordPage /> },
      { path: "/reset-password", element: <ResetPasswordPage /> },
      { path: "/complete-profile", element: <ProtectedRoute><CompleteProfilePage /></ProtectedRoute> },
      { path: "/auth/callback", element: <AuthCallbackPage /> },
      { path: "/game", element: <SuspenseWrapper><OrchestrationGame /></SuspenseWrapper> },
      { path: "/tetris", element: <SuspenseWrapper><TetrisGame /></SuspenseWrapper> },
    ]
  }
]);
