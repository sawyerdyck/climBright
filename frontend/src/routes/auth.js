const express = require("express");
const bcrypt = require("bcryptjs");
const rateLimit = require("express-rate-limit");

const User = require("../models/User");
const { signSession, setSessionCookie, clearSessionCookie, requireAuth } = require("../middleware/auth");

const router = express.Router();
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 60,
  standardHeaders: true,
  legacyHeaders: false,
});
router.use(authLimiter);

router.post("/register", async (req, res) => {
  const { email, password } = req.body || {};
  const safeEmail = typeof email === "string" ? email.trim().toLowerCase() : "";
  if (!safeEmail || !password) return res.status(400).json({ error: "email and password required" });
  if (typeof password !== "string" || password.length < 6) {
    return res.status(400).json({ error: "password must be at least 6 characters" });
  }

  const existing = await User.findOne({ email: { $eq: safeEmail } });
  if (existing) return res.status(409).json({ error: "email already registered" });

  const passwordHash = await bcrypt.hash(password, 10);
  const user = await User.create({ email: safeEmail, passwordHash });

  const token = signSession({ userId: user._id.toString(), email: user.email });
  setSessionCookie(res, token);

  return res.json({ ok: true, user: { id: user._id.toString(), email: user.email } });
});

router.post("/login", async (req, res) => {
  const { email, password } = req.body || {};
  const safeEmail = typeof email === "string" ? email.trim().toLowerCase() : "";
  if (!safeEmail || !password) return res.status(400).json({ error: "email and password required" });

  const user = await User.findOne({ email: { $eq: safeEmail } });
  if (!user) return res.status(401).json({ error: "invalid credentials" });

  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) return res.status(401).json({ error: "invalid credentials" });

  const token = signSession({ userId: user._id.toString(), email: user.email });
  setSessionCookie(res, token);

  return res.json({ ok: true, user: { id: user._id.toString(), email: user.email } });
});

router.get("/me", requireAuth, async (req, res) => {
  return res.json({ ok: true, user: { id: req.user.userId, email: req.user.email } });
});

router.post("/logout", (_req, res) => {
  clearSessionCookie(res);
  return res.json({ ok: true });
});

module.exports = router;
