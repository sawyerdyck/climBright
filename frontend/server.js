const path = require("path");

require("dotenv").config();

const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const cookieParser = require("cookie-parser");

const { requireAuth } = require("./src/middleware/auth");
const authRoutes = require("./src/routes/auth");
const imageRoutes = require("./src/routes/images");

const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;
const MONGODB_URI = process.env.MONGODB_URI;
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN;

const app = express();

app.use(
  cors({
    origin: FRONTEND_ORIGIN ? [FRONTEND_ORIGIN] : true,
    credentials: true,
  })
);

app.use(express.json({ limit: "12mb" }));
app.use(cookieParser());

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

// Serve the prototype frontend pages without exposing the whole folder
app.get("/", (_req, res) => res.redirect("/login"));
app.get("/login", (_req, res) => res.sendFile(path.join(__dirname, "login.html")));
app.get("/register", (_req, res) => res.sendFile(path.join(__dirname, "register.html")));
app.get("/holds", (_req, res) => res.sendFile(path.join(__dirname, "holds.html")));
app.get("/wall", (_req, res) => res.sendFile(path.join(__dirname, "wall.html")));

app.get("/styles.css", (_req, res) => res.sendFile(path.join(__dirname, "styles.css")));
app.get("/auth.js", (_req, res) => res.sendFile(path.join(__dirname, "auth.js")));
app.get("/holds.js", (_req, res) => res.sendFile(path.join(__dirname, "holds.js")));
app.get("/wall.js", (_req, res) => res.sendFile(path.join(__dirname, "wall.js")));

app.use("/api/auth", authRoutes);
app.use("/api/images", requireAuth, imageRoutes);

async function start() {
  if (!MONGODB_URI) {
    throw new Error("Missing MONGODB_URI in environment");
  }

  await mongoose.connect(MONGODB_URI);

  app.listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`Server listening on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
