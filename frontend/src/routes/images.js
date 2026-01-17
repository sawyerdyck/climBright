const express = require("express");

const ImageUpload = require("../models/ImageUpload");

const router = express.Router();

router.post("/", async (req, res) => {
  const { imageBase64, originalName, mimeType, aiEndpoint, aiResponseRaw, holds, bestHold } = req.body || {};

  if (!imageBase64 || typeof imageBase64 !== "string") {
    return res.status(400).json({ error: "imageBase64 is required" });
  }

  const doc = await ImageUpload.create({
    userId: req.user.userId,
    imageBase64,
    originalName,
    mimeType,
    aiEndpoint,
    aiResponseRaw,
    holds: Array.isArray(holds)
      ? holds.map((h) => ({ raw: h.raw ?? h, confidence: h.confidence }))
      : [],
    bestHold,
  });

  return res.json({ ok: true, id: doc._id.toString() });
});

router.get("/mine", async (req, res) => {
  const docs = await ImageUpload.find({ userId: req.user.userId })
    .sort({ createdAt: -1 })
    .select({ imageBase64: 0 })
    .limit(50);

  return res.json({ ok: true, items: docs });
});

module.exports = router;
