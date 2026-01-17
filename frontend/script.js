const API_BASE = ""; // same origin as the Express server
const FASTAPI_URL = "https://your-fastapi-host.example.com/analyze"; // placeholder (replace later)

function setupUpload(boxId, previewId, onFileSelected) {
  const box = document.getElementById(boxId);
  if (!box) return; // section may be disabled in HTML

  const input = box.querySelector("input");
  const preview = document.getElementById(previewId);

  box.addEventListener("click", () => input.click());

  box.addEventListener("dragover", e => {
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

    if (typeof onFileSelected === "function") {
      const file = input.files?.[0];
      if (file) await onFileSelected(file);
    }
  });

  input.addEventListener("change", async () => {
    handlePreview(input, preview);
    if (typeof onFileSelected === "function") {
      const file = input.files?.[0];
      if (file) await onFileSelected(file);
    }
  });
}

function handlePreview(input, preview) {
  preview.innerHTML = "";
  const file = input.files[0];
  if (!file) return;

  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
}

function setHoldResult(text, isError = false) {
  const el = document.getElementById("holdResultText");
  if (!el) return;
  el.classList.toggle("placeholder", !isError);
  el.style.color = isError ? "#ff7b72" : "";
  el.textContent = text;
}

async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const commaIdx = dataUrl.indexOf(",");
      // Returns just the base64 payload (no data: prefix)
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
  return (
    hold.type ||
    hold.label ||
    hold.name ||
    hold.class ||
    hold.grip_type ||
    hold.gripType ||
    "Unknown"
  );
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

// Auth UI
const authStatus = document.getElementById("authStatus");
const openAuthBtn = document.getElementById("openAuthBtn");
const logoutBtn = document.getElementById("logoutBtn");
const authModal = document.getElementById("authModal");
const closeAuthBtn = document.getElementById("closeAuthBtn");
const tabLogin = document.getElementById("tabLogin");
const tabRegister = document.getElementById("tabRegister");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const authError = document.getElementById("authError");

let currentUser = null;

function setAuthError(msg) {
  if (!authError) return;
  if (!msg) {
    authError.hidden = true;
    authError.textContent = "";
    return;
  }
  authError.hidden = false;
  authError.textContent = msg;
}

function setModalOpen(open) {
  if (!authModal) return;
  authModal.classList.toggle("open", open);
  authModal.setAttribute("aria-hidden", open ? "false" : "true");
  if (!open) setAuthError("");
}

function setTab(which) {
  const isLogin = which === "login";
  if (tabLogin) tabLogin.classList.toggle("active", isLogin);
  if (tabRegister) tabRegister.classList.toggle("active", !isLogin);
  if (loginForm) loginForm.hidden = !isLogin;
  if (registerForm) registerForm.hidden = isLogin;
  setAuthError("");
}

function renderAuthUI() {
  if (authStatus) {
    authStatus.textContent = currentUser ? `Signed in as ${currentUser.email}` : "Not signed in";
  }
  if (openAuthBtn) openAuthBtn.hidden = !!currentUser;
  if (logoutBtn) logoutBtn.hidden = !currentUser;
}

async function refreshMe() {
  try {
    const data = await apiJson("/api/auth/me", { method: "GET" });
    currentUser = data.user;
  } catch {
    currentUser = null;
  }
  renderAuthUI();
}

if (openAuthBtn) openAuthBtn.addEventListener("click", () => setModalOpen(true));
if (closeAuthBtn) closeAuthBtn.addEventListener("click", () => setModalOpen(false));
if (authModal)
  authModal.addEventListener("click", (e) => {
    if (e.target === authModal) setModalOpen(false);
  });
if (tabLogin) tabLogin.addEventListener("click", () => setTab("login"));
if (tabRegister) tabRegister.addEventListener("click", () => setTab("register"));

if (loginForm)
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const email = document.getElementById("loginEmail").value;
      const password = document.getElementById("loginPassword").value;
      const data = await apiJson("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      currentUser = data.user;
      renderAuthUI();
      setModalOpen(false);
    } catch (err) {
      setAuthError(err.message);
    }
  });

if (registerForm)
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const email = document.getElementById("registerEmail").value;
      const password = document.getElementById("registerPassword").value;
      const data = await apiJson("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      currentUser = data.user;
      renderAuthUI();
      setModalOpen(false);
    } catch (err) {
      setAuthError(err.message);
    }
  });

if (logoutBtn)
  logoutBtn.addEventListener("click", async () => {
    try {
      await apiJson("/api/auth/logout", { method: "POST" });
    } finally {
      currentUser = null;
      renderAuthUI();
    }
  });

// Upload -> base64 -> FastAPI -> show best hold -> store in Mongo
async function analyzeAndStoreHoldImage(file) {
  if (!currentUser) {
    setHoldResult("Please login to analyze and store uploads.", true);
    setModalOpen(true);
    setTab("login");
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
    aiJson = await res.json();
    if (!res.ok) throw new Error(aiJson?.error || `AI request failed (${res.status})`);
  } catch (err) {
    setHoldResult(`AI error: ${err.message}`, true);
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

  // Store in backend regardless (if we have base64); include AI response if available
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
    // Non-fatal for the user-facing analysis
    if (aiJson) setHoldResult(`Saved analysis, but store failed: ${err.message}`, true);
  }
}

// Init uploads
setupUpload("holdUpload", "holdPreview", analyzeAndStoreHoldImage);


const wallImage = document.getElementById("wallImage");
const wallWrapper = document.getElementById("wallImageWrapper");
const holdInfoText = document.getElementById("holdInfoText");

// DEMO data – this will come from your backend later
const demoHolds = [
  { id: 1, x: 0.3, y: 0.7, type: "Crimp", difficulty: "V3", confidence: 0.88 },
  { id: 2, x: 0.5, y: 0.5, type: "Jug", difficulty: "V2", confidence: 0.94 },
  { id: 3, x: 0.7, y: 0.3, type: "Sloper", difficulty: "V5", confidence: 0.81 }
];

function renderHolds(holds) {
  if (!wallWrapper) return;
  // Remove existing markers
  document.querySelectorAll(".hold-marker").forEach(m => m.remove());

  holds.forEach(hold => {
    const marker = document.createElement("div");
    marker.className = "hold-marker";

    marker.style.left = `${hold.x * 100}%`;
    marker.style.top = `${hold.y * 100}%`;

    marker.addEventListener("click", () => {
      selectHold(hold);
    });

    wallWrapper.appendChild(marker);
  });
}

function selectHold(hold) {
  if (!holdInfoText) return;
  holdInfoText.innerHTML = `
    <strong>Type:</strong> ${hold.type}<br/>
    <strong>Difficulty:</strong> ${hold.difficulty}<br/>
    <strong>Confidence:</strong> ${(hold.confidence * 100).toFixed(1)}%
  `;
}

// Hook into existing upload logic
function setupWallImageUpload() {
  const box = document.getElementById("wallUpload");
  if (!box || !wallImage || !wallWrapper) return;
  const input = box.querySelector("input");

  box.addEventListener("click", () => input.click());

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;

    wallImage.src = URL.createObjectURL(file);
    wallImage.onload = () => {
      renderHolds(demoHolds); // ← replace with backend response later
    };
  });
}

setupWallImageUpload();

// Initialize auth state
setTab("login");
refreshMe();
