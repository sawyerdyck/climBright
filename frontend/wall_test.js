const API_BASE = ""; // same origin as the Express server

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

const wallImage = document.getElementById("wallImage");
const wallWrapper = document.getElementById("wallImageWrapper");
const holdInfoText = document.getElementById("holdInfoText");
const overlaySvg = document.getElementById("wallOverlay");
const pop = document.getElementById("wallPopover");
const popClose = document.getElementById("wallPopoverClose");
const popTitle = document.getElementById("wallPopoverTitle");
const popSub = document.getElementById("wallPopoverSub");
const popText = document.getElementById("wallPopoverText");

const routeSelect = document.getElementById("routeSelect");
const toggleBoxes = document.getElementById("toggleBoxes");
const toggleNumbers = document.getElementById("toggleNumbers");
const toggleArrows = document.getElementById("toggleArrows");
const routeJsonFile = document.getElementById("routeJsonFile");



let currentHolds = [];
let currentCoach = null;
let currentPreviewUrl = null;

function setInfoHtml(html) {
  if (!holdInfoText) return;
  holdInfoText.classList.remove("placeholder");
  holdInfoText.innerHTML = html;
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
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function clearSvg() {
  if (!overlaySvg) return;
  while (overlaySvg.firstChild) overlaySvg.removeChild(overlaySvg.firstChild);
}

function hidePopover() {
  if (!pop) return;
  pop.classList.add("hidden");
}

function showPopoverAt(xPx, yPx, title, subtitle, text) {
  if (!pop) return;
  popTitle.textContent = title;
  popSub.textContent = subtitle;
  popText.textContent = text;

  // place near click, clamp inside wrapper
  const W = wallWrapper.clientWidth;
  const H = wallWrapper.clientHeight;

  let left = xPx + 12;
  let top = yPx + 12;

  left = Math.max(12, Math.min(left, W - 360));
  top = Math.max(12, Math.min(top, H - 220));

  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
  pop.classList.remove("hidden");
}

function bboxToCenterAndSize(bbox) {
  const [x1, y1, x2, y2] = bbox;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const bw = Math.max(1, x2 - x1);
  const bh = Math.max(1, y2 - y1);
  return { x1, y1, x2, y2, cx, cy, bw, bh };
}

// Handles BOTH coach formats:
// - NEW: coach.routeA.steps[] (Gemini output style)
// - OLD: coach.routeA[] as list of holds with id
function getRouteSteps(coach, routeName, holds) {
  const route = coach?.[routeName];
  if (!route) return [];

  // If Gemini-style: route.steps exists
  if (Array.isArray(route.steps)) return route.steps;

  // If legacy-style: route is an array of holds with ids
  if (Array.isArray(route)) {
    // Build "steps" from detected holds in that order (best effort)
    return route
      .map((h, idx) => {
        const hold = holds.find(x => x.id === h.id) || holds.find(x => String(x.id) === String(h.id));
        if (!hold?.bbox) return null;
        const { cx, cy, bw, bh } = bboxToCenterAndSize(hold.bbox);
        return {
          instruction: `Step ${idx + 1}`,
          hold_id: hold.id,
          type: hold.type || "Unknown",
          // Use pixel-based info (we'll draw in pixel-space via viewBox)
          _px: { cx, cy, bw, bh, bbox: hold.bbox }
        };
      })
      .filter(Boolean);
  }

  return [];
}

function stepToPxRect(step, holds) {
  // Gemini-style provides center_norm + bbox_wh_norm: convert to px using natural dims
  if (step.center_norm && step.bbox_wh_norm) {
    const W = wallImage.naturalWidth;
    const H = wallImage.naturalHeight;
    const cx = step.center_norm[0] * W;
    const cy = step.center_norm[1] * H; // NO Y-FLIP (your overlay is working now)
    const bw = step.bbox_wh_norm[0] * W;
    const bh = step.bbox_wh_norm[1] * H;
    return { cx, cy, bw, bh, x: cx - bw / 2, y: cy - bh / 2 };
  }

  // Legacy path from bbox
  if (step._px) {
    const { cx, cy, bw, bh } = step._px;
    return { cx, cy, bw, bh, x: cx - bw / 2, y: cy - bh / 2 };
  }

  // Fallback: find by hold_id in holds
  const hold = holds.find(h => String(h.id) === String(step.hold_id));
  if (hold?.bbox) {
    const { cx, cy, bw, bh } = bboxToCenterAndSize(hold.bbox);
    return { cx, cy, bw, bh, x: cx - bw / 2, y: cy - bh / 2 };
  }

  return null;
}

function renderOverlay(holds, coach) {
  if (!overlaySvg || !wallImage || !wallWrapper) return;

  clearSvg();
  hidePopover();

  const imgW = wallImage.naturalWidth || wallImage.width;
  const imgH = wallImage.naturalHeight || wallImage.height;

  // SVG matches the image's natural pixel coordinate space
  overlaySvg.setAttribute("viewBox", `0 0 ${imgW} ${imgH}`);
  overlaySvg.setAttribute("preserveAspectRatio", "none");

  // route controls enabled only if we have coach output
  if (routeSelect) routeSelect.disabled = !coach;

  const routeName = routeSelect?.value || "routeA";
  const steps = coach ? getRouteSteps(coach, routeName, holds) : [];

  // Optional: draw all holds faintly as context (boxes)
  // (Comment this out if you want ONLY route highlights)
  if (Array.isArray(holds) && holds.length && toggleBoxes?.checked) {
    holds.forEach((h) => {
      if (!h.bbox) return;
      const [x1, y1, x2, y2] = h.bbox;
      const r = svgEl("rect", {
        x: x1, y: y1, width: x2 - x1, height: y2 - y1,
        rx: 10, ry: 10,
        fill: "rgba(61,220,151,0.04)",
        stroke: "rgba(157,167,179,0.22)",
        "stroke-width": 2
      });
      overlaySvg.appendChild(r);
    });
  }

  // Draw route steps (bright)
  const pts = [];
  steps.forEach((s, i) => {
    const px = stepToPxRect(s, holds);
    if (!px) return;

    pts.push({ cx: px.cx, cy: px.cy });

    // box
    const box = svgEl("rect", {
      x: px.x, y: px.y, width: px.bw, height: px.bh,
      rx: 12, ry: 12,
      fill: "rgba(61,220,151,0.08)",
      stroke: "rgba(61,220,151,0.95)",
      "stroke-width": 4,
      style: "cursor:pointer"
    });

    box.addEventListener("click", (ev) => {
      ev.stopPropagation();

      // Update existing right-side info panel too
      const hold = holds.find(h => String(h.id) === String(s.hold_id));
      if (hold) selectHold(hold);

      // Position popover in wrapper pixel space (need to convert from natural->display)
      const dispW = wallWrapper.clientWidth;
      const dispH = wallWrapper.clientHeight;
      const xDisp = (px.cx / imgW) * dispW;
      const yDisp = (px.cy / imgH) * dispH;

      showPopoverAt(
        xDisp,
        yDisp,
        `${s.type || "Hold"} (ID ${s.hold_id})`,
        `Step ${i + 1} · ${routeName}`,
        s.instruction || "—"
      );
    });

    overlaySvg.appendChild(box);

    // number
    if (toggleNumbers?.checked) {
      const c = svgEl("circle", {
        cx: px.cx, cy: px.cy, r: 18,
        fill: "rgba(11,15,20,0.8)",
        stroke: "rgba(61,220,151,0.95)",
        "stroke-width": 3
      });
      overlaySvg.appendChild(c);

      const t = svgEl("text", {
        x: px.cx, y: px.cy + 6,
        "text-anchor": "middle",
        "font-size": 16,
        "font-weight": 800,
        fill: "rgba(61,220,151,0.98)"
      });
      t.textContent = String(i + 1);
      overlaySvg.appendChild(t);
    }
  });

  // arrows
  if (toggleArrows?.checked && pts.length >= 2) {
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      const line = svgEl("line", {
        x1: a.cx, y1: a.cy, x2: b.cx, y2: b.cy,
        stroke: "rgba(61,220,151,0.9)",
        "stroke-width": 4,
        "stroke-linecap": "round"
      });
      overlaySvg.appendChild(line);
    }
  }
}

function waitForWallImage(callback) {
  if (!wallImage) return;
  if (wallImage.complete && wallImage.naturalWidth) {
    callback();
    return;
  }
  wallImage.onload = () => {
    callback();
  };
}


function selectHold(hold) {
  const conf = typeof hold.confidence === "number" ? hold.confidence : Number(hold.confidence || 0);
  const pct = conf <= 1 ? conf * 100 : conf;
  const inRouteA = Array.isArray(currentCoach?.routeA)
    ? currentCoach.routeA.some((h) => h.id === hold.id)
    : false;
  const inRouteB = Array.isArray(currentCoach?.routeB)
    ? currentCoach.routeB.some((h) => h.id === hold.id)
    : false;

  setInfoHtml(
    `
    <strong>Type:</strong> ${hold.type || "Unknown"}<br/>
    <strong>Confidence:</strong> ${pct.toFixed(1)}%<br/>
    <strong>In Route A:</strong> ${inRouteA ? "Yes" : "No"}<br/>
    <strong>In Route B:</strong> ${inRouteB ? "Yes" : "No"}
    `
  );
}

function showCoachSummary(coach) {
  if (!coach) return;
  const difficulty = coach.difficulty || "Unknown";
  const notes = coach.notes || "";
  setInfoHtml(
    `
    <strong>Suggested Difficulty:</strong> ${difficulty}<br/>
    <strong>Notes:</strong> ${notes}
    `
  );
}

async function analyzeWall(file) {
  if (!file) return;

  if (holdInfoText) {
    holdInfoText.classList.add("placeholder");
    holdInfoText.textContent = "Analyzing wall…";
  }

  if (wallImage) {
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
    currentPreviewUrl = URL.createObjectURL(file);
    wallImage.src = currentPreviewUrl;
  }

  const imageBase64 = await fileToBase64(file);
  const base64Payload = String(imageBase64).includes(",")
    ? String(imageBase64).split(",")[1]
    : String(imageBase64);

  // 1) Frontend calls FastAPI directly to get holds
  const fastapiUrl = await getFastApiUrl();
  if (!fastapiUrl) {
    setInfoHtml('<span style="color:#ff7b72">FASTAPI_URL is not configured.</span>');
    return;
  }

  let aiJson;
  try {
    const res = await fetch(fastapiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || "image/jpeg",
        data: base64Payload,
      }),
    });
    aiJson = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(aiJson?.detail || aiJson?.error || `FastAPI request failed (${res.status})`);
  } catch (err) {
    setInfoHtml(`<span style="color:#ff7b72">AI error: ${err.message}</span>`);
    return;
  }

  currentHolds = Array.isArray(aiJson?.holds) ? aiJson.holds : [];

  // 2) Backend only runs pathfinder.py using the image + holds
  const result = await apiJson("/api/wall/analyze", {
    method: "POST",
    body: JSON.stringify({
      imageBase64,
      filename: file.name,
      holds: currentHolds,
    }),
  });

  currentCoach = result.coach || null;

  waitForWallImage(() => {
    renderOverlay(currentHolds, currentCoach);
    showCoachSummary(currentCoach);
  });
}

function setupWallImageUpload() {
  const box = document.getElementById("wallUpload");
  if (!box || !wallImage) return;
  const input = box.querySelector("input");

  box.addEventListener("click", () => input.click());

  box.addEventListener("dragover", (e) => {
    e.preventDefault();
    box.style.borderColor = "#3ddc97";
  });

  box.addEventListener("dragleave", () => {
    box.style.borderColor = "#30363d";
  });

  box.addEventListener("drop", (e) => {
    e.preventDefault();
    input.files = e.dataTransfer.files;
    box.style.borderColor = "#30363d";

    const file = input.files?.[0];
    if (!file) return;
    analyzeWall(file).catch((err) => {
      setInfoHtml(`<span style="color:#ff7b72">${err.message}</span>`);
    });
  });

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;

    analyzeWall(file).catch((err) => {
      setInfoHtml(`<span style="color:#ff7b72">${err.message}</span>`);
    });
  });
}
async function readJsonFile(file) {
  const text = await file.text();
  return JSON.parse(text);
}
async function loadTestImageAndJson(jsonFileObj) {
  // 1) Hardcode image path (put this image in /public or same static folder)
  // Example: /assets/walls/altitude2_classified.jpg
  const TEST_IMAGE_URL = "../test_data_sd/altitude2_classified.jpg";

  // 2) Load JSON from uploaded file
  const coach = await readJsonFile(jsonFileObj);
  currentCoach = coach;

  // 3) In test mode we might not have full holds bboxes.
  // If your JSON is Gemini-style (center_norm + bbox_wh_norm), overlay can draw without holds.
  currentHolds = []; // optional

  // 4) Wait for image to load, then render
  waitForWallImage(() => {
    renderOverlay(currentHolds, currentCoach);
    showCoachSummary(currentCoach);
  });
  wallImage.src = TEST_IMAGE_URL;

  // enable route controls since we have coach json
  if (routeSelect) routeSelect.disabled = false;
}


(async function init() {
  await requireSessionOrRedirect();
  setupWallImageUpload();
  if (routeJsonFile) {
  routeJsonFile.addEventListener("change", async () => {
    const f = routeJsonFile.files?.[0];
    if (!f) return;

    try {
      await loadTestImageAndJson(f);
    } catch (err) {
      setInfoHtml(`<span style="color:#ff7b72">Invalid JSON: ${err.message}</span>`);
    }
  });
}

  if (routeSelect) routeSelect.addEventListener("change", () => renderOverlay(currentHolds, currentCoach));
  if (toggleBoxes) toggleBoxes.addEventListener("change", () => renderOverlay(currentHolds, currentCoach));
  if (toggleNumbers) toggleNumbers.addEventListener("change", () => renderOverlay(currentHolds, currentCoach));
  if (toggleArrows) toggleArrows.addEventListener("change", () => renderOverlay(currentHolds, currentCoach));

  if (wallWrapper) wallWrapper.addEventListener("click", () => hidePopover());
  if (popClose) popClose.addEventListener("click", (e) => { e.stopPropagation(); hidePopover(); });

  window.addEventListener("resize", () => renderOverlay(currentHolds, currentCoach));

})();
