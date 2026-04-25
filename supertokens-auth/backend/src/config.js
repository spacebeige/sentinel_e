const path = require("path");
const dotenv = require("dotenv");

dotenv.config({ path: process.env.AUTH_ENV_FILE || path.resolve(__dirname, "../.env") });

function required(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

function optional(name, fallback = "") {
  return process.env[name] || fallback;
}

function parseOrigins(value) {
  return value
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
}

const APP_NAME = optional("APP_NAME", "SuperTokens Auth App");
const API_DOMAIN = required("API_DOMAIN");
const WEBSITE_DOMAIN = required("WEBSITE_DOMAIN");

const config = {
  appName: APP_NAME,
  port: Number(optional("PORT", "4000")),
  nodeEnv: optional("NODE_ENV", "development"),
  apiDomain: API_DOMAIN,
  websiteDomain: WEBSITE_DOMAIN,
  supertokensConnectionURI: required("SUPERTOKENS_CONNECTION_URI"),
  supertokensApiKey: optional("SUPERTOKENS_API_KEY", ""),
  neonDatabaseUrl: required("NEON_DATABASE_URL"),
  corsOrigins: parseOrigins(optional("CORS_ORIGINS", `${WEBSITE_DOMAIN},http://localhost:3000`)),
  googleClientId: required("GOOGLE_CLIENT_ID"),
  googleClientSecret: required("GOOGLE_CLIENT_SECRET"),
  githubClientId: required("GITHUB_CLIENT_ID"),
  githubClientSecret: required("GITHUB_CLIENT_SECRET"),
  apiBasePath: optional("AUTH_API_BASE_PATH", "/auth"),
  websiteBasePath: optional("AUTH_WEBSITE_BASE_PATH", "/auth")
};

module.exports = config;
