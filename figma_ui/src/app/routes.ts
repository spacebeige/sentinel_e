import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { HomePage } from "./components/HomePage";
import { ChatPage } from "./components/ChatPage";
import { ModelsPage } from "./components/ModelsPage";
import { PricingPage } from "./components/PricingPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: HomePage },
      { path: "chat", Component: ChatPage },
      { path: "models", Component: ModelsPage },
      { path: "pricing", Component: PricingPage },
    ],
  },
]);
