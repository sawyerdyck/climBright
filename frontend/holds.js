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

// --- DOM refs ---
const resultText = document.getElementById("holdResultText");
const holdList = document.getElementById("holdList");
const holdImage = document.getElementById("holdImage");
const holdImageWrapper = document.getElementById("holdImageWrapper");
const holdOverlay = document.getElementById("holdOverlay");

// --- Result display ---
function showLoading() {
  if (resultText) {
    resultText.classList.remove("placeholder");
    resultText.innerHTML = '<div class="skeleton skeleton-lg"></div><div class="skeleton"></div><div class="skeleton" style="width:60%"></div>';
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
  holdList.classList.remove('fade-in-up'); void holdList.offsetWidth; holdList.classList.add('fade-in-up');
  holdList.textContent = "";
  holds.forEach((h, i) => {
    const label = getHoldLabel(h);
    const conf = getHoldConfidence(h);
    const pct = (conf <= 1 ? conf * 100 : conf).toFixed(1);
    const isBest = i === bestIdx;

    const li = document.createElement("li");
    if (isBest) li.classList.add("best");
    li.dataset.holdIdx = String(i);

    const labelSpan = document.createElement("span");
    labelSpan.className = "hold-label";
    labelSpan.textContent = `${label}${isBest ? " ★" : ""}`;

    const confSpan = document.createElement("span");
    confSpan.className = "hold-conf";
    confSpan.textContent = `${pct}%`;

    li.appendChild(labelSpan);
    li.appendChild(confSpan);
    holdList.appendChild(li);
  });

  // Hover on list item highlights the bbox
  holdList.querySelectorAll("li").forEach((li) => {
    li.addEventListener("mouseenter", () => highlightHold(Number(li.dataset.holdIdx)));
    li.addEventListener("mouseleave", () => highlightHold(-1));
  });
}

// --- Bounding box overlay ---
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function clearOverlay() {
  if (!holdOverlay) return;
  while (holdOverlay.firstChild) holdOverlay.removeChild(holdOverlay.firstChild);
}

// Colors for different holds
const HOLD_COLORS = [
  "#3ddc97", "#f0b429", "#ff7b72", "#79c0ff", "#d2a8ff",
  "#ffa657", "#7ee787", "#ff9bce", "#a5d6ff", "#ffd700",
];

let drawnRects = [];

function drawBoundingBoxes(holds, imgW, imgH) {
  if (!holdOverlay || !imgW || !imgH) return;
  clearOverlay();
  drawnRects = [];

  holdOverlay.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
  holdOverlay.setAttribute("preserveAspectRatio", "xMidYMid meet");

  holds.forEach((hold, i) => {
    const bbox = hold.bbox || hold.box;
    if (!Array.isArray(bbox) || bbox.length !== 4) return;

    const [x1, y1, x2, y2] = bbox;
    const color = HOLD_COLORS[i % HOLD_COLORS.length];

    // Bounding box rect
    const rect = svgEl("rect", {
      x: x1, y: y1, width: x2 - x1, height: y2 - y1,
      fill: "none", stroke: color, "stroke-width": 3,
      rx: 4, opacity: 0.85,
    });
    rect.dataset.idx = i;
    holdOverlay.appendChild(rect);

    // Label background
    const label = getHoldLabel(hold);
    const labelText = `${label}`;
    const fontSize = Math.max(12, Math.min(18, (x2 - x1) * 0.15));
    const padding = 4;
    const textY = y1 > fontSize + padding * 2 ? y1 - padding : y2 + fontSize + padding;

    const bg = svgEl("rect", {
      x: x1, y: textY - fontSize, width: label.length * fontSize * 0.6 + padding * 2, height: fontSize + padding,
      fill: color, rx: 3, opacity: 0.9,
    });
    holdOverlay.appendChild(bg);

    const text = svgEl("text", {
      x: x1 + padding, y: textY - padding,
      "font-size": fontSize, "font-weight": 600, fill: "#0e1117",
    });
    text.textContent = labelText;
    holdOverlay.appendChild(text);

    drawnRects.push({ rect, bg, text, color });
  });
}

function highlightHold(idx) {
  drawnRects.forEach((item, i) => {
    const isHighlighted = i === idx;
    item.rect.setAttribute("stroke-width", isHighlighted ? 5 : 3);
    item.rect.setAttribute("opacity", idx === -1 ? 0.85 : isHighlighted ? 1 : 0.35);
    item.bg.setAttribute("opacity", idx === -1 ? 0.9 : isHighlighted ? 1 : 0.3);
    item.text.setAttribute("opacity", idx === -1 ? 1 : isHighlighted ? 1 : 0.3);
  });
}

// --- Upload / Preview ---
function handlePreview(input) {
  const file = input.files[0];
  if (!file || !holdImage || !holdImageWrapper) return;

  holdImageWrapper.hidden = false;
  holdImageWrapper.classList.remove('fade-in-up'); void holdImageWrapper.offsetWidth; holdImageWrapper.classList.add('fade-in-up');
  holdImage.src = URL.createObjectURL(file);
  clearOverlay();
}

function setupUpload(boxId, onFileSelected) {
  const box = document.getElementById(boxId);
  if (!box) return;
  const input = box.querySelector("input");

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
    handlePreview(input);
    const file = input.files?.[0];
    if (file) await onFileSelected(file);
  });

  input.addEventListener("change", async () => {
    handlePreview(input);
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
  // Persist image for cross-page navigation
  storeImage(file);

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
    showResult(`${holds.length} hold${holds.length > 1 ? "s" : ""} detected. Best: ${getHoldLabel(best)} (${pct}%)`);
    showHoldsList(holds, bestIdx);

    // Draw bounding boxes on the image once it's loaded
    if (holdImage) {
      const drawBoxes = () => {
        const imgW = holdImage.naturalWidth || holdImage.width;
        const imgH = holdImage.naturalHeight || holdImage.height;
        if (imgW && imgH) drawBoundingBoxes(holds, imgW, imgH);
      };
      if (holdImage.complete && holdImage.naturalWidth) drawBoxes();
      else holdImage.addEventListener("load", drawBoxes, { once: true });
    }
  } else if (aiJson) {
    showResult("AI responded, but no holds detected in this image.", true);
  }

  // Persist analysis results for cross-page navigation
  if (holds.length > 0) {
    storeAnalysis("holds", { holds, bestIdx: holds.indexOf(holds.reduce((a, b) => getHoldConfidence(a) >= getHoldConfidence(b) ? a : b, holds[0])) });
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
  setupUpload("holdUpload", analyzeAndStoreHoldImage);

  // Restore previously uploaded image and analysis if navigating back
  const stored = getStoredImage();
  const analyzeBtn = document.getElementById("analyzeStoredBtn");

  if (stored && holdImage && holdImageWrapper) {
    holdImage.src = stored.dataUrl;
    holdImageWrapper.hidden = false;

    // Restore analysis results (holds, bboxes)
    const analysis = getStoredAnalysis("holds");
    if (analysis && Array.isArray(analysis.holds) && analysis.holds.length > 0) {
      const holds = analysis.holds;
      let bestIdx = 0;
      let bestConf = getHoldConfidence(holds[0]);
      for (let i = 1; i < holds.length; i++) {
        const c = getHoldConfidence(holds[i]);
        if (c > bestConf) { bestIdx = i; bestConf = c; }
      }
      const best = holds[bestIdx];
      const pct = (bestConf <= 1 ? bestConf * 100 : bestConf).toFixed(1);
      showResult(`${holds.length} hold${holds.length > 1 ? "s" : ""} detected. Best: ${getHoldLabel(best)} (${pct}%)`);
      showHoldsList(holds, bestIdx);

      const drawBoxes = () => {
        const imgW = holdImage.naturalWidth || holdImage.width;
        const imgH = holdImage.naturalHeight || holdImage.height;
        if (imgW && imgH) drawBoundingBoxes(holds, imgW, imgH);
      };
      if (holdImage.complete && holdImage.naturalWidth) drawBoxes();
      else holdImage.addEventListener("load", drawBoxes, { once: true });
    } else if (analyzeBtn) {
      // Image loaded but no analysis — offer to analyze it
      analyzeBtn.hidden = false;
      analyzeBtn.addEventListener("click", () => {
        analyzeBtn.hidden = true;
        const file = dataUrlToFile(stored.dataUrl, stored.name, stored.type);
        analyzeAndStoreHoldImage(file);
      });
    }
  }
})();
