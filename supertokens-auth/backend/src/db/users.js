const pool = require("./pool");

async function upsertUser({ supertokensUserId, email, provider }) {
  const query = `
    INSERT INTO users (supertokens_user_id, email, provider)
    VALUES ($1, $2, $3)
    ON CONFLICT (supertokens_user_id)
    DO UPDATE SET
      email = EXCLUDED.email,
      provider = EXCLUDED.provider,
      updated_at = NOW()
    RETURNING id, supertokens_user_id, email, provider, created_at, updated_at;
  `;

  const { rows } = await pool.query(query, [supertokensUserId, email, provider]);
  return rows[0];
}

async function getUserBySupertokensId(supertokensUserId) {
  const query = `
    SELECT id, supertokens_user_id, email, provider, created_at, updated_at
    FROM users
    WHERE supertokens_user_id = $1
    LIMIT 1;
  `;

  const { rows } = await pool.query(query, [supertokensUserId]);
  return rows[0] || null;
}

module.exports = {
  upsertUser,
  getUserBySupertokensId
};
