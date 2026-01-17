const API_BASE = ""; // same origin as the Express server
const FASTAPI_URL = "https://your-fastapi-host.example.com/analyze"; // placeholder

async function apiJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

async function requireSessionOrRedirect() {
  try {
    const data = await apiJson("/api/auth/me", { method: "GET" });
    const authStatus = document.getElementById("authStatus");
    if (authStatus) authStatus.textContent = `Signed in as ${data.user.email}`;

    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
      logoutBtn.hidden = false;
      logoutBtn.addEventListener("click", async () => {
        await apiJson("/api/auth/logout", { method: "POST" });
        window.location.href = "/login";
      });
    }

    return data.user;
  } catch {
    window.location.href = "/login";
    return null;
  }
}

function setHoldResult(text, isError = false) {
  const el = document.getElementById("holdResultText");
  if (!el) return;
  el.classList.toggle("placeholder", !isError);
  el.style.color = isError ? "#ff7b72" : "";
  el.textContent = text;
}

function handlePreview(input, preview) {
  preview.innerHTML = "";
  const file = input.files[0];
  if (!file) return;

  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
}

function setupUpload(boxId, previewId, onFileSelected) {
  const box = document.getElementById(boxId);
  const input = box.querySelector("input");
  const preview = document.getElementById(previewId);

  box.addEventListener("click", () => input.click());

  box.addEventListener("dragover", (e) => {
    e.preventDefault();
    box.style.borderColor = "#3ddc97";
  });

  box.addEventListener("dragleave", () => {
    box.style.borderColor = "#30363d";
  });

  box.addEventListener("drop", async (e) => {
    e.preventDefault();
    input.files = e.dataTransfer.files;
    handlePreview(input, preview);
    box.style.borderColor = "#30363d";

    const file = input.files?.[0];
    if (file && typeof onFileSelected === "function") await onFileSelected(file);
  });

  input.addEventListener("change", async () => {
    handlePreview(input, preview);
    const file = input.files?.[0];
    if (file && typeof onFileSelected === "function") await onFileSelected(file);
  });
}

async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const commaIdx = dataUrl.indexOf(",");
      resolve(commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl);
    };
    reader.readAsDataURL(file);
  });
}

function getHoldConfidence(hold) {
  if (!hold || typeof hold !== "object") return 0;
  const candidates = [hold.confidence, hold.conf, hold.score, hold.prob, hold.probability];
  for (const v of candidates) {
    const n = Number(v);
    if (!Number.isNaN(n)) return n;
  }
  return 0;
}

function getHoldLabel(hold) {
  if (!hold || typeof hold !== "object") return "Unknown";
  return hold.type || hold.label || hold.name || hold.class || hold.grip_type || hold.gripType || "Unknown";
}

function pickBestHold(holds) {
  if (!Array.isArray(holds) || holds.length === 0) return null;
  let best = holds[0];
  let bestC = getHoldConfidence(best);
  for (const h of holds.slice(1)) {
    const c = getHoldConfidence(h);
    if (c > bestC) {
      best = h;
      bestC = c;
    }
  }
  return best;
}

async function analyzeAndStoreHoldImage(file) {
  // Only allow JPEG / PNG (works for click-select + drag/drop)
  const allowedMime = new Set(["image/jpeg", "image/png"]);
  const ext = (file?.name || "").toLowerCase();
  const hasAllowedExt = ext.endsWith(".jpg") || ext.endsWith(".jpeg") || ext.endsWith(".png");
  const hasAllowedMime = allowedMime.has(file?.type);

  // Some browsers/flows may give an empty MIME type; fall back to extension check.
  if (!hasAllowedMime && !(file?.type === "" && hasAllowedExt)) {
    setHoldResult("Only JPG/JPEG or PNG files are allowed.", true);
    return;
  }

  setHoldResult("Encoding image and calling AI...");

  let imageBase64;
  try {
    imageBase64 = await fileToBase64(file);
  } catch {
    setHoldResult("Failed to convert image to base64.", true);
    return;
  }

  let aiJson = null;
  try {
    const res = await fetch(FASTAPI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });

    aiJson = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(aiJson?.error || `AI request failed (${res.status})`);
  } catch (err) {
    setHoldResult(`AI error: ${err.message}`, true);
    // still attempt to store the upload (without AI result) below
  }

  const holds = aiJson?.holds;
  const bestHold = pickBestHold(holds);

  if (bestHold) {
    const c = getHoldConfidence(bestHold);
    const label = getHoldLabel(bestHold);
    const pct = c <= 1 ? c * 100 : c;
    setHoldResult(`Best match: ${label} (${pct.toFixed(1)}% confidence)`);
  } else if (aiJson) {
    setHoldResult("AI responded, but no holds were returned.", true);
  }

  try {
    await apiJson("/api/images", {
      method: "POST",
      body: JSON.stringify({
        imageBase64,
        originalName: file.name,
        mimeType: file.type,
        aiEndpoint: FASTAPI_URL,
        aiResponseRaw: aiJson,
        holds: Array.isArray(holds)
          ? holds.map((h) => ({ raw: h, confidence: getHoldConfidence(h) }))
          : [],
        bestHold: bestHold || null,
      }),
    });
  } catch (err) {
    // non-fatal: the user still got the AI result
    if (aiJson) setHoldResult(`Saved analysis, but store failed: ${err.message}`, true);
  }
}

(async function init() {
  await requireSessionOrRedirect();
  setupUpload("holdUpload", "holdPreview", analyzeAndStoreHoldImage);
})();
