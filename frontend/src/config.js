// Centralized API base URL for all backend calls
export const API_BASE = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === "development" ? "http://localhost:8000" : undefined);
if (!API_BASE) {
	throw new Error("API URL environment variable not set.");
}
