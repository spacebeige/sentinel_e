import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { HomePage } from "./components/HomePage";
import { ChatPage } from "./components/ChatPage";
import { ModelsPage } from "./components/ModelsPage";
import { PricingPage } from "./components/PricingPage";
import { DebatePage } from "./components/DebatePage";
import { MissionControlPage } from "./components/MissionControlPage";
import { GovernancePage } from "./components/GovernancePage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: HomePage },
      { path: "chat", Component: ChatPage },
      { path: "debate", Component: DebatePage },
      { path: "mission-control", Component: MissionControlPage },
      { path: "governance", Component: GovernancePage },
      { path: "models", Component: ModelsPage },
      { path: "pricing", Component: PricingPage },
    ],
  },
]);
