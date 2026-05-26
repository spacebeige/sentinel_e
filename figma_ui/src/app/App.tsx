import { RouterProvider } from "react-router";
import { router } from "./routes.tsx";
import { ChatInteractionProvider } from "./context/ChatInteractionContext";
import { CinematicErrorBoundary } from "./components/CinematicErrorBoundary";

export default function App() {
  return (
    <CinematicErrorBoundary>
      <ChatInteractionProvider>
        <RouterProvider router={router} />
      </ChatInteractionProvider>
    </CinematicErrorBoundary>
  );
}
