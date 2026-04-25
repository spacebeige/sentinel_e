const express = require("express");
const supertokens = require("supertokens-node");

const { requireSession } = require("../middleware/requireSession");
const { getUserBySupertokensId, upsertUser } = require("../db/users");

const router = express.Router();

router.get("/me", requireSession, async (req, res, next) => {
  try {
    const supertokensUserId = req.session.getUserId();
    const appUser = await getUserBySupertokensId(supertokensUserId);

    res.status(200).json({
      session: {
        userId: supertokensUserId,
        handle: req.session.getHandle(),
        accessTokenPayload: req.session.getAccessTokenPayload()
      },
      user: appUser
    });
  } catch (error) {
    next(error);
  }
});

router.post("/sync-user", requireSession, async (req, res, next) => {
  try {
    const supertokensUserId = req.session.getUserId();
    const authUser = await supertokens.getUser(supertokensUserId);

    if (!authUser) {
      return res.status(404).json({ error: "Authenticated user not found in SuperTokens" });
    }

    const provider = authUser.thirdParty?.[0]?.id || "unknown";
    const email = authUser.emails?.[0] || null;

    const user = await upsertUser({
      supertokensUserId,
      email,
      provider
    });

    return res.status(200).json({ user });
  } catch (error) {
    next(error);
  }
});

router.post("/logout", requireSession, async (req, res, next) => {
  try {
    await req.session.revokeSession();
    res.status(200).json({ message: "Logged out successfully" });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
