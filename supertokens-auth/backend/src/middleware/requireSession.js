const { verifySession } = require("supertokens-node/recipe/session/framework/express");

const requireSession = verifySession();

module.exports = {
  requireSession
};
