const API_BASE = "";

async function getFastApiUrl() {
  const res = await fetch("/config.json", { credentials: "include" });
  const cfg = await res.json().catch(() => ({}));
  return cfg.fastapiUrl || "";
}

async function apiJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
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

// --- Result display ---
const resultText = document.getElementById("holdResultText");
const holdList = document.getElementById("holdList");

function showLoading() {
  if (resultText) {
    resultText.classList.remove("placeholder");
    resultText.innerHTML = '<span class="spinner"></span><span class="loading-text">Analyzing hold…</span>';
  }
  if (holdList) holdList.hidden = true;
}

function showResult(text, isError = false) {
  if (!resultText) return;
  resultText.classList.toggle("placeholder", false);
  resultText.style.color = isError ? "var(--danger)" : "";
  resultText.textContent = text;
}

function showHoldsList(holds, bestIdx) {
  if (!holdList || !holds.length) return;
  holdList.hidden = false;
  holdList.innerHTML = holds.map((h, i) => {
    const label = getHoldLabel(h);
    const conf = getHoldConfidence(h);
    const pct = (conf <= 1 ? conf * 100 : conf).toFixed(1);
    const isBest = i === bestIdx;
    return `<li class="${isBest ? "best" : ""}">
      <span class="hold-label">${label}${isBest ? " ★" : ""}</span>
      <span class="hold-conf">${pct}%</span>
    </li>`;
  }).join("");
}

// --- Upload / Preview ---
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
  if (!box) return;
  const input = box.querySelector("input");
  const preview = document.getElementById(previewId);

  box.addEventListener("click", () => input.click());

  box.addEventListener("dragover", (e) => {
    e.preventDefault();
    box.classList.add("dragover");
  });

  box.addEventListener("dragleave", () => box.classList.remove("dragover"));

  box.addEventListener("drop", async (e) => {
    e.preventDefault();
    box.classList.remove("dragover");
    input.files = e.dataTransfer.files;
    handlePreview(input, preview);
    const file = input.files?.[0];
    if (file) await onFileSelected(file);
  });

  input.addEventListener("change", async () => {
    handlePreview(input, preview);
    const file = input.files?.[0];
    if (file) await onFileSelected(file);
  });
}

// --- Helpers ---
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
  for (const v of [hold.confidence, hold.conf, hold.score, hold.prob, hold.probability]) {
    const n = Number(v);
    if (!Number.isNaN(n)) return n;
  }
  return 0;
}

function getHoldLabel(hold) {
  if (!hold || typeof hold !== "object") return "Unknown";
  return hold.type || hold.label || hold.name || hold.class || hold.grip_type || hold.gripType || "Unknown";
}

// --- Main analysis ---
async function analyzeAndStoreHoldImage(file) {
  const ext = (file?.name || "").toLowerCase();
  const hasAllowedExt = ext.endsWith(".jpg") || ext.endsWith(".jpeg") || ext.endsWith(".png");
  const hasAllowedMime = file?.type === "image/jpeg" || file?.type === "image/png";

  if (!hasAllowedMime && !(file?.type === "" && hasAllowedExt)) {
    showResult("Only JPG/JPEG or PNG files are allowed.", true);
    return;
  }

  const fastapiUrl = await getFastApiUrl();
  if (!fastapiUrl) {
    showResult("FASTAPI_URL is not configured. Set it in frontend/.env and restart the web server.", true);
    return;
  }

  showLoading();

  let imageBase64;
  try {
    imageBase64 = await fileToBase64(file);
  } catch {
    showResult("Failed to read file.", true);
    return;
  }

  let aiJson = null;
  try {
    const res = await fetch(fastapiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || "image/jpeg",
        data: imageBase64,
      }),
    });
    aiJson = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(aiJson?.detail || aiJson?.error || `AI request failed (${res.status})`);
  } catch (err) {
    showResult(`AI error: ${err.message}`, true);
  }

  const holds = Array.isArray(aiJson?.holds) ? aiJson.holds : [];

  if (holds.length > 0) {
    // Find best
    let bestIdx = 0;
    let bestConf = getHoldConfidence(holds[0]);
    for (let i = 1; i < holds.length; i++) {
      const c = getHoldConfidence(holds[i]);
      if (c > bestConf) { bestIdx = i; bestConf = c; }
    }
    const best = holds[bestIdx];
    const pct = (bestConf <= 1 ? bestConf * 100 : bestConf).toFixed(1);
    showResult(`Best match: ${getHoldLabel(best)} (${pct}% confidence)`);
    showHoldsList(holds, bestIdx);
  } else if (aiJson) {
    showResult("AI responded, but no holds detected in this image.", true);
  }

  // Store in backend (non-fatal)
  try {
    await apiJson("/api/images", {
      method: "POST",
      body: JSON.stringify({
        imageBase64,
        originalName: file.name,
        mimeType: file.type,
        aiEndpoint: fastapiUrl,
        aiResponseRaw: aiJson,
        holds: holds.map((h) => ({ raw: h, confidence: getHoldConfidence(h) })),
        bestHold: holds.length > 0 ? holds[0] : null,
      }),
    });
  } catch {
    // non-fatal
  }
}

(async function init() {
  await requireSessionOrRedirect();
  setupUpload("holdUpload", "holdPreview", analyzeAndStoreHoldImage);
})();
