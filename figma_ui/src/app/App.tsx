import { RouterProvider } from "react-router";
import { router } from "./routes.tsx";
import { ChatInteractionProvider } from "./context/ChatInteractionContext";
import { ThemeProvider } from "next-themes";
import { CinematicErrorBoundary } from "./components/CinematicErrorBoundary";

export default function App() {
  return (
    <CinematicErrorBoundary>
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
        <ChatInteractionProvider>
          <RouterProvider router={router} />
        </ChatInteractionProvider>
      </ThemeProvider>
    </CinematicErrorBoundary>
  );
}
