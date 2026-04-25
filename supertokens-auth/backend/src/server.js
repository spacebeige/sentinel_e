const express = require("express");
const cors = require("cors");
const morgan = require("morgan");
const { middleware, errorHandler, getAllCORSHeaders } = require("supertokens-node/framework/express");

const config = require("./config");
const { initSuperTokens } = require("./auth/supertokens");
const authRoutes = require("./routes/authRoutes");
const protectedRoutes = require("./routes/protectedRoutes");

initSuperTokens();

const app = express();
const superTokensErrorHandler = errorHandler();

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin) return callback(null, true);
      if (config.corsOrigins.includes(origin)) return callback(null, true);
      return callback(new Error(`CORS blocked origin: ${origin}`));
    },
    allowedHeaders: ["content-type", ...getAllCORSHeaders()],
    credentials: true
  })
);

app.use(morgan("combined"));
app.use(express.json());

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

// SuperTokens middleware handles /auth/* routes.
app.use(middleware());

app.use("/api/auth", authRoutes);
app.use("/api/protected", protectedRoutes);

app.use((err, _req, res, _next) => {
  if (err && err.message && err.message.startsWith("CORS blocked origin")) {
    return res.status(403).json({ error: err.message });
  }
  return superTokensErrorHandler(err, _req, res, _next);
});

const server = app.listen(config.port, () => {
  console.log(`Backend API listening on http://localhost:${config.port}`);
  console.log(`SuperTokens API base path: ${config.apiBasePath}`);
});

function shutdown(signal) {
  console.log(`Received ${signal}. Shutting down gracefully...`);
  server.close(() => process.exit(0));
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
