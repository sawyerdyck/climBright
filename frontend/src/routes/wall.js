const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const express = require("express");

const router = express.Router();

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function decodeBase64Payload(base64) {
  // Accept either pure base64 or data: URLs
  const s = String(base64 || "");
  const commaIdx = s.indexOf(",");
  const payload = commaIdx >= 0 ? s.slice(commaIdx + 1) : s;
  return Buffer.from(payload, "base64");
}

function parseJsonFromStdout(stdout) {
  const text = String(stdout || "").trim();
  if (!text) throw new Error("Pathfinder returned no output");

  // In case there are extra log lines, try to parse the last JSON object.
  const lines = text.split("\n").filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const candidate = lines[i].trim();
    if (!candidate.startsWith("{") || !candidate.endsWith("}")) continue;
    try {
      return JSON.parse(candidate);
    } catch {
      // keep searching
    }
  }

  // fallback: try whole string
  return JSON.parse(text);
}

router.post("/analyze", async (req, res) => {
  const { imageBase64, filename, holds } = req.body || {};
  if (!imageBase64) return res.status(400).json({ error: "imageBase64 is required" });

  const rootDir = path.resolve(__dirname, "..", "..", ".."); // -> climBright
  const uploadsDir = path.join(rootDir, "db", "uploads");
  ensureDir(uploadsDir);

  const safeName = String(filename || "wall.jpg").replace(/[^a-zA-Z0-9._-]/g, "_");
  const ts = Date.now();
  const imagePath = path.join(uploadsDir, `${ts}_${safeName}`);

  try {
    const buf = decodeBase64Payload(imageBase64);
    fs.writeFileSync(imagePath, buf);
  } catch {
    return res.status(400).json({ error: "Invalid base64 image" });
  }

  const safeHolds = Array.isArray(holds) ? holds : [];
  const holdsJsonPath = path.join(uploadsDir, `${ts}_holds.json`);
  fs.writeFileSync(holdsJsonPath, JSON.stringify({ holds: safeHolds }, null, 2));

  const venvPyUnix = path.join(rootDir, "env", "bin", "python");
  const venvPyWin = path.join(rootDir, "env", "Scripts", "python.exe");
  const venvPyWinShim = path.join(rootDir, "env", "Scripts", "python");
  let pythonBin = process.env.PYTHON_BIN;

  // Prefer the repo venv if present (it has Pillow/google-genai, etc.).
  // This prevents default configs like PYTHON_BIN=python3 from breaking.
  if (
    (!pythonBin || pythonBin === "python3" || pythonBin === "python") &&
    (fs.existsSync(venvPyUnix) || fs.existsSync(venvPyWin) || fs.existsSync(venvPyWinShim))
  ) {
    pythonBin = fs.existsSync(venvPyUnix)
      ? venvPyUnix
      : fs.existsSync(venvPyWin)
        ? venvPyWin
        : venvPyWinShim;
  }

  if (!pythonBin) {
    pythonBin = process.platform === "win32" ? "python" : "python3";
  }
  const pathfinderPath = path.join(rootDir, "pathfinder.py");

  const args = [pathfinderPath, "--image", imagePath, "--json", holdsJsonPath];

  const child = spawn(pythonBin, args, {
    env: process.env,
    cwd: rootDir,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let out = "";
  let err = "";
  let responded = false;

  function reply(status, payload) {
    if (responded) return;
    responded = true;
    if (status === "error") {
      res.status(payload.status || 502).json(payload.body);
    } else {
      res.json(payload.body);
    }
  }

  child.stdout.on("data", (d) => {
    out += d.toString();
  });
  child.stderr.on("data", (d) => {
    err += d.toString();
  });

  child.on("error", (spawnErr) => {
    const message =
      spawnErr.code === "ENOENT"
        ? `Unable to execute python interpreter \"${pythonBin}\". Install Python or set PYTHON_BIN to a valid executable.`
        : `Failed to launch pathfinder: ${spawnErr.message}`;
    reply("error", {
      status: 502,
      body: { error: message },
    });
  });

  child.on("close", (code) => {
    if (responded) return;

    if (code !== 0) {
      return reply("error", {
        status: 502,
        body: { error: `pathfinder failed (${code})`, details: err || out },
      });
    }

    try {
      const coach = parseJsonFromStdout(out);
      return reply("ok", { body: { ok: true, coach } });
    } catch (e) {
      return reply("error", {
        status: 502,
        body: { error: "Failed to parse pathfinder output", details: out || err },
      });
    }
  });
});

module.exports = router;
