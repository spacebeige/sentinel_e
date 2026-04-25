"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { SessionAuth, signOut } from "supertokens-auth-react/recipe/session";

import { fetchCurrentSession, fetchProtectedDashboard, syncUserToDatabase } from "../../lib/api";
import { initSuperTokens } from "../../lib/supertokens";

initSuperTokens();

function DashboardContent() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [sessionData, setSessionData] = useState(null);
  const [protectedData, setProtectedData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;

    async function loadDashboard() {
      try {
        const [sessionRes, protectedRes] = await Promise.all([
          fetchCurrentSession(),
          fetchProtectedDashboard(),
          syncUserToDatabase()
        ]);

        if (!mounted) return;

        setSessionData(sessionRes);
        setProtectedData(protectedRes);
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Failed to load dashboard");
      } finally {
        if (mounted) setLoading(false);
      }
    }

    loadDashboard();

    return () => {
      mounted = false;
    };
  }, []);

  async function handleLogout() {
    await signOut();
    router.push("/auth");
  }

  if (loading) return <main className="container">Loading dashboard...</main>;

  return (
    <main className="container">
      <h1>Protected Dashboard</h1>
      <p>Only authenticated users with a valid SuperTokens session can access this page.</p>

      {error && <p className="error">{error}</p>}

      <section className="card">
        <h2>Session</h2>
        <pre>{JSON.stringify(sessionData, null, 2)}</pre>
      </section>

      <section className="card">
        <h2>Protected API Data</h2>
        <pre>{JSON.stringify(protectedData, null, 2)}</pre>
      </section>

      <button onClick={handleLogout} className="button secondary">
        Logout
      </button>
    </main>
  );
}

export default function DashboardPage() {
  return (
    <SessionAuth>
      <DashboardContent />
    </SessionAuth>
  );
}
