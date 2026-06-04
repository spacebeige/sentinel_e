if (!import.meta.env.VITE_API_URL) {
  throw new Error("VITE_API_URL is not configured");
}

export const API_BASE = import.meta.env.VITE_API_URL;

// Phase 3: Log API configuration
if (typeof window !== 'undefined') {
  console.log(`✓ API: Using ${API_BASE} (from VITE_API_URL)`);
}
