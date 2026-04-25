const { Pool } = require("pg");
const config = require("../config");

const pool = new Pool({
  connectionString: config.neonDatabaseUrl,
  ssl: {
    rejectUnauthorized: false
  }
});

module.exports = pool;
