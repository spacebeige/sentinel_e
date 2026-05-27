import React from "react";
import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import HomePage from "./components/HomePage";
import { ChatPage } from "./components/ChatPage";
import { EnginesPage } from "./components/EnginesPage";
import { PricingPage } from "./components/PricingPage";
import OrchestrationGame from "./orchestration/game/OrchestrationGame";
import TetrisGame from "./tetris/game/TetrisGame";

// ChatInteractionProvider lives in App.tsx — DO NOT add it here
export const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <HomePage /> },
      { path: "chat", element: <ChatPage /> },
      { path: "engines", element: <EnginesPage /> },
      { path: "models", element: <EnginesPage /> }, // legacy alias
      { path: "pricing", element: <PricingPage /> },
    ],
  },
  { path: "/game", element: <OrchestrationGame /> },
  { path: "/tetris", element: <TetrisGame /> },
]);
