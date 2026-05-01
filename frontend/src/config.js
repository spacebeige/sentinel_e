// Centralized API base URL for all backend calls
export const API_BASE = process.env.REACT_APP_API_BASE || process.env.REACT_APP_API_URL || "https://sentinel-e.onrender.com";

// Phase 3: Log API configuration
if (typeof window !== 'undefined') {
  const envSource = process.env.REACT_APP_API_BASE ? 'REACT_APP_API_BASE' : 
                   process.env.REACT_APP_API_URL ? 'REACT_APP_API_URL' : 'DEFAULT';
  console.log(`✓ API: Using ${API_BASE} (from ${envSource})`);
}
