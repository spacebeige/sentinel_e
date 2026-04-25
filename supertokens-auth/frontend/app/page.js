"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { doesSessionExist } from "supertokens-auth-react/recipe/session";
import { initSuperTokens } from "../lib/supertokens";

initSuperTokens();

export default function HomePage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    let mounted = true;
    doesSessionExist().then((sessionExists) => {
      if (mounted) {
        setIsAuthenticated(sessionExists);
      }
    });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <main className="container">
      <h1>SuperTokens + Neon Auth Starter</h1>
      <p>
        Sign in with Google or GitHub. Sessions are stored in secure HTTPOnly cookies and
        protected API routes require session validation.
      </p>

      <div className="actions">
        {isAuthenticated ? (
          <>
            <Link href="/dashboard" className="button primary">
              Go to Dashboard
            </Link>
          </>
        ) : (
          <Link href="/auth" className="button primary">
            Sign In
          </Link>
        )}
      </div>
    </main>
  );
}
