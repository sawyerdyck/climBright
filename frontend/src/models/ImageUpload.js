const mongoose = require("mongoose");

const HoldSchema = new mongoose.Schema(
  {
    raw: { type: Object, required: true },
    confidence: { type: Number, required: false },
  },
  { _id: false }
);

const ImageUploadSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    originalName: { type: String },
    mimeType: { type: String },

    // NOTE: storing base64 in Mongo is OK for small images; large images may exceed the 16MB document limit.
    imageBase64: { type: String, required: true },

    aiEndpoint: { type: String },
    aiResponseRaw: { type: Object },
    holds: { type: [HoldSchema], default: [] },
    bestHold: { type: Object },
  },
  { timestamps: true }
);

module.exports = mongoose.model("ImageUpload", ImageUploadSchema);
