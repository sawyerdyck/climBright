const API_BASE = "";

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

async function getFastApiUrl() {
  const res = await fetch("/config.json", { credentials: "include" });
  const cfg = await res.json().catch(() => ({}));
  return cfg.fastapiUrl || "";
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

// DOM refs
const wallImage = document.getElementById("wallImage");
const wallWrapper = document.getElementById("wallImageWrapper");
const wallContainer = document.getElementById("wallContainer");
const holdInfoText = document.getElementById("holdInfoText");
const overlaySvg = document.getElementById("wallOverlay");
const coachSummary = document.getElementById("coachSummary");
const coachContent = document.getElementById("coachContent");

let currentHolds = [];
let currentCoach = null;

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

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function clearOverlay() {
  if (!overlaySvg) return;
  while (overlaySvg.firstChild) overlaySvg.removeChild(overlaySvg.firstChild);
}

function bboxToCenter(bbox) {
  const [x1, y1, x2, y2] = bbox;
  return { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2 };
}

function setInfoHtml(html) {
  if (!holdInfoText) return;
  holdInfoText.classList.remove("placeholder");
  holdInfoText.innerHTML = html;
}

function showLoading() {
  if (holdInfoText) {
    holdInfoText.classList.remove("placeholder");
    holdInfoText.innerHTML = '<span class="spinner"></span><span class="loading-text">Analyzing wall…</span>';
  }
  if (coachSummary) coachSummary.hidden = true;
}

// --- Route resolution ---
function resolveRouteSteps(coach, holds) {
  if (!coach) return [];
  for (const name of ["routeA", "route", "routeB"]) {
    const route = coach[name];
    if (!route) continue;
    if (Array.isArray(route.steps)) return route.steps;
    if (Array.isArray(route)) {
      return route.map((step, idx) => {
        if (Array.isArray(step?.bbox)) return step;
        if (step?.center_norm) return step;
        const id = step?.id ?? step?.hold_id ?? step?.holdId;
        if (id === undefined) return null;
        const hold = holds.find((h) => String(h.id) === String(id));
        if (!hold) return null;
        return { ...step, bbox: hold.bbox, center_norm: hold.center_norm, hold_id: hold.id, instruction: step?.instruction || `Step ${idx + 1}` };
      }).filter(Boolean);
    }
  }
  return [];
}

function stepToPoint(step, holds) {
  if (Array.isArray(step.center_norm) && wallImage) {
    const W = wallImage.naturalWidth || wallImage.width;
    const H = wallImage.naturalHeight || wallImage.height;
    return { cx: step.center_norm[0] * W, cy: step.center_norm[1] * H };
  }
  if (Array.isArray(step.bbox)) return bboxToCenter(step.bbox);
  const id = step?.hold_id ?? step?.id ?? step?.holdId;
  if (id !== undefined) {
    const hold = holds.find((h) => String(h.id) === String(id));
    if (Array.isArray(hold?.bbox)) return bboxToCenter(hold.bbox);
  }
  return null;
}

// --- Rendering ---
function renderOverlay(holds, coach) {
  if (!overlaySvg || !wallImage) return;
  clearOverlay();

  const imgW = wallImage.naturalWidth || wallImage.width;
  const imgH = wallImage.naturalHeight || wallImage.height;
  if (!imgW || !imgH) return;

  overlaySvg.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
  overlaySvg.setAttribute("preserveAspectRatio", "none");

  const steps = resolveRouteSteps(coach, holds);
  if (!steps.length) return;

  const points = [];
  steps.forEach((step, idx) => {
    const pt = stepToPoint(step, holds);
    if (!pt) return;
    points.push({ ...pt, idx });
  });

  // Draw path lines
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    overlaySvg.appendChild(svgEl("line", {
      x1: a.cx, y1: a.cy, x2: b.cx, y2: b.cy,
      stroke: "rgba(61,220,151,0.85)", "stroke-width": 6, "stroke-linecap": "round",
    }));
  }

  // Draw numbered nodes
  points.forEach(({ cx, cy, idx }) => {
    overlaySvg.appendChild(svgEl("circle", {
      cx, cy, r: 16,
      fill: "rgba(14,17,23,0.8)", stroke: "rgba(61,220,151,0.95)", "stroke-width": 4,
    }));
    const text = svgEl("text", {
      x: cx, y: cy + 6, "text-anchor": "middle",
      "font-size": 16, "font-weight": 700, fill: "rgba(61,220,151,0.98)",
    });
    text.textContent = String(idx + 1);
    overlaySvg.appendChild(text);
  });
}

function renderHolds(holds, imgW, imgH) {
  if (!wallWrapper) return;
  document.querySelectorAll(".hold-marker").forEach((m) => m.remove());

  holds.forEach((hold) => {
    const bbox = hold.bbox;
    if (!Array.isArray(bbox) || bbox.length !== 4) return;
    const { cx, cy } = bboxToCenter(bbox);

    const marker = document.createElement("div");
    marker.className = "hold-marker";
    marker.style.left = `${(cx / imgW) * 100}%`;
    marker.style.top = `${(cy / imgH) * 100}%`;
    marker.addEventListener("click", () => selectHold(hold));
    wallWrapper.appendChild(marker);
  });
}

function selectHold(hold) {
  const conf = Number(hold.confidence || 0);
  const pct = (conf <= 1 ? conf * 100 : conf).toFixed(1);

  // Find step number if in route
  const steps = resolveRouteSteps(currentCoach, currentHolds);
  let stepNum = null;
  for (let i = 0; i < steps.length; i++) {
    const sid = steps[i]?.id ?? steps[i]?.hold_id;
    if (sid !== undefined && String(sid) === String(hold.id)) { stepNum = i + 1; break; }
  }

  setInfoHtml(`
    <strong>Type:</strong> ${hold.type || "Unknown"}<br/>
    <strong>Confidence:</strong> ${pct}%<br/>
    ${stepNum ? `<strong>Route step:</strong> #${stepNum}<br/>` : ""}
    <strong>Hold ID:</strong> ${hold.id}
  `);
}

function renderCoachSummary(coach) {
  if (!coachSummary || !coachContent || !coach) return;

  const diff = (coach.difficulty || "Unknown").toLowerCase();
  const diffClass = diff === "easy" ? "easy" : diff === "hard" ? "hard" : "moderate";
  const notes = coach.notes || "No additional notes.";
  const routeALen = Array.isArray(coach.routeA) ? coach.routeA.length : 0;
  const routeBLen = Array.isArray(coach.routeB) ? coach.routeB.length : 0;

  coachContent.innerHTML = `
    <p><strong>Difficulty:</strong> <span class="difficulty-badge ${diffClass}">${coach.difficulty || "Unknown"}</span></p>
    <p style="margin-top:0.75rem"><strong>Route A:</strong> ${routeALen} holds</p>
    ${routeBLen ? `<p><strong>Route B:</strong> ${routeBLen} holds</p>` : ""}
    <p style="margin-top:0.75rem; color: var(--muted); font-size: 0.9rem">${notes}</p>
  `;
  coachSummary.hidden = false;
}

// --- Wall analysis ---
async function analyzeWall(file) {
  if (!file) return;

  showLoading();
  if (wallContainer) wallContainer.hidden = false;
  clearOverlay();

  const imageBase64 = await fileToBase64(file);
  const base64Payload = imageBase64.includes(",") ? imageBase64.split(",")[1] : imageBase64;

  const fastapiUrl = await getFastApiUrl();
  if (!fastapiUrl) {
    setInfoHtml('<span style="color:var(--danger)">FASTAPI_URL is not configured. Set it in frontend/.env.</span>');
    return;
  }

  // 1) FastAPI: detect + classify holds
  let aiJson;
  try {
    const res = await fetch(fastapiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file.name, content_type: file.type || "image/jpeg", data: base64Payload }),
    });
    aiJson = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(aiJson?.detail || aiJson?.error || `FastAPI error (${res.status})`);
  } catch (err) {
    setInfoHtml(`<span style="color:var(--danger)">AI error: ${err.message}</span>`);
    return;
  }

  currentHolds = Array.isArray(aiJson?.holds) ? aiJson.holds : [];

  // 2) Backend: pathfinder
  let result;
  try {
    result = await apiJson("/api/wall/analyze", {
      method: "POST",
      body: JSON.stringify({ imageBase64, filename: file.name, holds: currentHolds }),
    });
  } catch (err) {
    setInfoHtml(`<span style="color:var(--danger)">Route error: ${err.message}</span>`);
    // Still show holds on the image
    wallImage.onload = () => {
      const w = wallImage.naturalWidth, h = wallImage.naturalHeight;
      renderHolds(currentHolds, w, h);
    };
    wallImage.src = URL.createObjectURL(file);
    return;
  }

  currentCoach = result.coach || null;

  wallImage.onload = () => {
    const w = wallImage.naturalWidth, h = wallImage.naturalHeight;
    renderHolds(currentHolds, w, h);
    renderOverlay(currentHolds, currentCoach);
    renderCoachSummary(currentCoach);
    setInfoHtml(`<span style="color:var(--muted)">${currentHolds.length} holds detected. Click one for details.</span>`);
  };
  wallImage.src = URL.createObjectURL(file);
}

// --- Upload setup ---
function setupWallImageUpload() {
  const box = document.getElementById("wallUpload");
  if (!box || !wallImage) return;
  const input = box.querySelector("input");

  box.addEventListener("click", () => input.click());

  box.addEventListener("dragover", (e) => {
    e.preventDefault();
    box.classList.add("dragover");
  });

  box.addEventListener("dragleave", () => box.classList.remove("dragover"));

  box.addEventListener("drop", (e) => {
    e.preventDefault();
    box.classList.remove("dragover");
    input.files = e.dataTransfer.files;
    const file = input.files?.[0];
    if (file) analyzeWall(file).catch((err) => setInfoHtml(`<span style="color:var(--danger)">${err.message}</span>`));
  });

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) analyzeWall(file).catch((err) => setInfoHtml(`<span style="color:var(--danger)">${err.message}</span>`));
  });
}

(async function init() {
  await requireSessionOrRedirect();
  setupWallImageUpload();
})();
