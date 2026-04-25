import "./globals.css";
import Providers from "./providers";

export const metadata = {
  title: "SuperTokens Auth App",
  description: "Production-ready auth with SuperTokens + Neon PostgreSQL"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
