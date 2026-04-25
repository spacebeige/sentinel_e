const API_DOMAIN = process.env.NEXT_PUBLIC_API_DOMAIN || "http://localhost:4000";

export async function fetchCurrentSession() {
  const res = await fetch(`${API_DOMAIN}/api/auth/me`, {
    method: "GET",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    },
    cache: "no-store"
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch session: ${res.status}`);
  }

  return res.json();
}

export async function syncUserToDatabase() {
  const res = await fetch(`${API_DOMAIN}/api/auth/sync-user`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    }
  });

  if (!res.ok) {
    throw new Error(`Failed to sync user: ${res.status}`);
  }

  return res.json();
}

export async function fetchProtectedDashboard() {
  const res = await fetch(`${API_DOMAIN}/api/protected/dashboard`, {
    method: "GET",
    credentials: "include",
    headers: {
      "Content-Type": "application/json"
    },
    cache: "no-store"
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch protected data: ${res.status}`);
  }

  return res.json();
}
