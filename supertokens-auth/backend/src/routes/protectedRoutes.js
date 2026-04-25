const express = require("express");

const { requireSession } = require("../middleware/requireSession");

const router = express.Router();

router.get("/dashboard", requireSession, async (req, res) => {
  res.status(200).json({
    message: "You are authenticated and can access protected resources.",
    userId: req.session.getUserId(),
    accessTokenPayload: req.session.getAccessTokenPayload()
  });
});

module.exports = router;
