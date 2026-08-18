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
let selectedMarkerEl = null;

// Route display config
const ROUTE_STYLES = {
  A: { color: "rgba(61,220,151,0.9)", dash: "", label: "Route A", labelColor: "#3ddc97" },
  B: { color: "rgba(255,166,87,0.9)", dash: "12,6", label: "Route B", labelColor: "#ffa657" },
};

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
    holdInfoText.innerHTML = '<div class="skeleton skeleton-lg"></div><div class="skeleton"></div><div class="skeleton" style="width:60%"></div>';
  }
  if (coachSummary) coachSummary.hidden = true;
}

// --- Route resolution ---
function resolveRouteFromArray(route, holds) {
  if (!Array.isArray(route)) return [];
  if (route.length > 0 && route[0].steps) return route[0].steps;

  return route.map((step, idx) => {
    if (Array.isArray(step?.bbox) || step?.center_norm) return step;
    const id = step?.id ?? step?.hold_id ?? step?.holdId;
    if (id === undefined) return null;
    const hold = holds.find((h) => String(h.id) === String(id));
    if (!hold) return null;
    return { ...step, bbox: hold.bbox, center_norm: hold.center_norm, hold_id: hold.id };
  }).filter(Boolean);
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
function drawRoute(steps, holds, style, labelPrefix) {
  if (!steps.length) return;

  const imgW = wallImage.naturalWidth || wallImage.width;
  const imgH = wallImage.naturalHeight || wallImage.height;
  const pad = 18; // keep nodes this far from edges so they're visible

  const points = [];
  steps.forEach((step, idx) => {
    const pt = stepToPoint(step, holds);
    if (!pt) return;
    // Clamp to stay within visible bounds
    const cx = Math.max(pad, Math.min(imgW - pad, pt.cx));
    const cy = Math.max(pad, Math.min(imgH - pad, pt.cy));
    points.push({ cx, cy, idx });
  });

  if (!points.length) return;

  // Draw path lines
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    const attrs = {
      x1: a.cx, y1: a.cy, x2: b.cx, y2: b.cy,
      stroke: style.color, "stroke-width": 5, "stroke-linecap": "round",
    };
    if (style.dash) attrs["stroke-dasharray"] = style.dash;
    overlaySvg.appendChild(svgEl("line", attrs));
  }

  // Draw numbered nodes
  points.forEach(({ cx, cy, idx }) => {
    overlaySvg.appendChild(svgEl("circle", {
      cx, cy, r: 14,
      fill: "rgba(14,17,23,0.8)", stroke: style.color, "stroke-width": 3,
    }));
    const text = svgEl("text", {
      x: cx, y: cy + 5, "text-anchor": "middle",
      "font-size": 13, "font-weight": 700, fill: style.color,
    });
    text.textContent = `${labelPrefix}${idx + 1}`;
    overlaySvg.appendChild(text);
  });
}

function renderOverlay(holds, coach) {
  if (!overlaySvg || !wallImage) return;
  clearOverlay();

  const imgW = wallImage.naturalWidth || wallImage.width;
  const imgH = wallImage.naturalHeight || wallImage.height;
  if (!imgW || !imgH) return;

  overlaySvg.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
  overlaySvg.setAttribute("preserveAspectRatio", "none");

  if (!coach) return;

  const stepsA = resolveRouteFromArray(coach.routeA, holds);
  const stepsB = resolveRouteFromArray(coach.routeB, holds);

  // Draw Route B first (behind) then Route A on top
  if (stepsB.length > 0) drawRoute(stepsB, holds, ROUTE_STYLES.B, "B");
  if (stepsA.length > 0) drawRoute(stepsA, holds, ROUTE_STYLES.A, "");
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
    marker.dataset.holdId = hold.id;
    marker.style.left = `${(cx / imgW) * 100}%`;
    marker.style.top = `${(cy / imgH) * 100}%`;
    marker.title = `Hold ${hold.id} — ${hold.type || "Unknown"}`;
    marker.addEventListener("click", () => selectHold(hold));
    wallWrapper.appendChild(marker);
  });
}

function selectHold(hold) {
  if (selectedMarkerEl) selectedMarkerEl.classList.remove('selected');
  const newMarker = wallWrapper?.querySelector(`.hold-marker[data-hold-id="${hold.id}"]`);
  if (newMarker) { newMarker.classList.add('selected'); selectedMarkerEl = newMarker; }

  const conf = Number(hold.confidence || 0);
  const pct = (conf <= 1 ? conf * 100 : conf).toFixed(1);

  // Check if hold is in route A or B
  const stepsA = currentCoach ? resolveRouteFromArray(currentCoach.routeA, currentHolds) : [];
  const stepsB = currentCoach ? resolveRouteFromArray(currentCoach.routeB, currentHolds) : [];

  const inA = stepsA.findIndex((s) => String(s?.id ?? s?.hold_id) === String(hold.id));
  const inB = stepsB.findIndex((s) => String(s?.id ?? s?.hold_id) === String(hold.id));

  let routeInfo = "";
  if (inA >= 0) routeInfo += `<span style="color:${ROUTE_STYLES.A.labelColor}">Route A step #${inA + 1}</span><br/>`;
  if (inB >= 0) routeInfo += `<span style="color:${ROUTE_STYLES.B.labelColor}">Route B step #${inB + 1}</span><br/>`;

  setInfoHtml(`
    <strong>Type:</strong> ${hold.type || "Unknown"}<br/>
    <strong>Confidence:</strong> ${pct}%<br/>
    ${routeInfo}
    <strong>Hold ID:</strong> ${hold.id}
  `);
  holdInfoText.classList.remove('fade-in-up'); void holdInfoText.offsetWidth; holdInfoText.classList.add('fade-in-up');
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
    <div class="route-legend" style="margin-top:0.75rem">
      <p><span style="color:${ROUTE_STYLES.A.labelColor}; font-weight:600">━━ Route A</span> (standard) — ${routeALen} holds</p>
      ${routeBLen ? `<p><span style="color:${ROUTE_STYLES.B.labelColor}; font-weight:600">╌╌ Route B</span> (harder) — ${routeBLen} holds</p>` : ""}
    </div>
    <p style="margin-top:0.75rem; color: var(--muted); font-size: 0.9rem">${notes}</p>
  `;
  coachSummary.hidden = false;
  coachSummary.classList.remove('fade-in-up'); void coachSummary.offsetWidth; coachSummary.classList.add('fade-in-up');
}

// --- Wall analysis ---
async function analyzeWall(file) {
  if (!file) return;

  // Persist image for cross-page navigation
  storeImage(file);

  showLoading();
  if (wallContainer) wallContainer.hidden = false;
  wallContainer.classList.add('fade-in-up');
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
    wallImage.onload = () => {
      const w = wallImage.naturalWidth, h = wallImage.naturalHeight;
      renderHolds(currentHolds, w, h);
    };
    wallImage.src = URL.createObjectURL(file);
    return;
  }

  currentCoach = result.coach || null;

  // Persist analysis for cross-page navigation
  storeAnalysis("wall", { holds: currentHolds, coach: currentCoach });

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

  // Restore previously uploaded image and analysis if navigating from another page
  const stored = getStoredImage();
  if (stored && wallImage && wallContainer) {
    wallImage.src = stored.dataUrl;
    wallContainer.hidden = false;

    // Restore analysis results (holds, routes, coach)
    const analysis = getStoredAnalysis("wall");
    if (analysis && Array.isArray(analysis.holds) && analysis.holds.length > 0) {
      currentHolds = analysis.holds;
      currentCoach = analysis.coach || null;

      wallImage.onload = () => {
        const w = wallImage.naturalWidth, h = wallImage.naturalHeight;
        renderHolds(currentHolds, w, h);
        if (currentCoach) {
          renderOverlay(currentHolds, currentCoach);
          renderCoachSummary(currentCoach);
        }
        setInfoHtml(`<span style="color:var(--muted)">${currentHolds.length} holds detected. Click one for details.</span>`);
      };
      // If image already loaded (cached)
      if (wallImage.complete && wallImage.naturalWidth) {
        const w = wallImage.naturalWidth, h = wallImage.naturalHeight;
        renderHolds(currentHolds, w, h);
        if (currentCoach) {
          renderOverlay(currentHolds, currentCoach);
          renderCoachSummary(currentCoach);
        }
        setInfoHtml(`<span style="color:var(--muted)">${currentHolds.length} holds detected. Click one for details.</span>`);
      }
    } else {
      setInfoHtml('<span style="color:var(--muted)">Previous image restored. Upload again to re-analyze.</span>');
    }
  }
})();
