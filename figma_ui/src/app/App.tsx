import { RouterProvider } from "react-router";
import { router } from "./routes.tsx";
import { ChatInteractionProvider } from "./context/ChatInteractionContext";
import { ThemeProvider } from "./context/ThemeContext";
import { CinematicErrorBoundary } from "./components/CinematicErrorBoundary";

export default function App() {
  return (
    <CinematicErrorBoundary>
      <ThemeProvider>
        <ChatInteractionProvider>
          <RouterProvider router={router} />
        </ChatInteractionProvider>
      </ThemeProvider>
    </CinematicErrorBoundary>
  );
}
