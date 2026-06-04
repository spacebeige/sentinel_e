if (!import.meta.env.VITE_API_URL) {
  console.error("Configuration Error: VITE_API_URL missing");
}

export const API_BASE = import.meta.env.VITE_API_URL || "";

console.log("=== RUNTIME DIAGNOSTICS ===");
console.log("VITE_API_URL =", import.meta.env.VITE_API_URL);
console.log("API_BASE =", API_BASE);
console.log("===========================");

// Phase 3: Log API configuration
if (typeof window !== 'undefined') {
  console.log(`✓ API: Using ${API_BASE} (from VITE_API_URL)`);
}
