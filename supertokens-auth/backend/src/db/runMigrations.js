const fs = require("fs");
const path = require("path");
const pool = require("./pool");

async function runMigrations() {
  const migrationsDir = path.resolve(__dirname, "../../../db/migrations");
  const files = fs
    .readdirSync(migrationsDir)
    .filter((file) => file.endsWith(".sql"))
    .sort();

  if (files.length === 0) {
    console.log("No migration files found.");
    return;
  }

  for (const file of files) {
    const fullPath = path.join(migrationsDir, file);
    const sql = fs.readFileSync(fullPath, "utf8");

    console.log(`Running migration: ${file}`);
    await pool.query(sql);
  }

  console.log("All migrations executed successfully.");
}

runMigrations()
  .catch((error) => {
    console.error("Migration failed:", error);
    process.exitCode = 1;
  })
  .finally(async () => {
    await pool.end();
  });
