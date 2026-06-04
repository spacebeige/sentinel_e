if (!import.meta.env.VITE_API_URL) {
  console.error("Configuration Error: VITE_API_URL missing");
}

export const API_BASE = import.meta.env.VITE_API_URL || "";

// Phase 3: Log API configuration
if (typeof window !== 'undefined') {
  console.log(`✓ API: Using ${API_BASE} (from VITE_API_URL)`);
}
