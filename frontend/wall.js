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

// DEMO data – replace with backend/FastAPI response later
const demoHolds = [
  { id: 1, x: 0.3, y: 0.7, type: "Crimp", difficulty: "V3", confidence: 0.88 },
  { id: 2, x: 0.5, y: 0.5, type: "Jug", difficulty: "V2", confidence: 0.94 },
  { id: 3, x: 0.7, y: 0.3, type: "Sloper", difficulty: "V5", confidence: 0.81 },
];

function renderHolds(holds) {
  if (!wallWrapper) return;
  document.querySelectorAll(".hold-marker").forEach((m) => m.remove());

  holds.forEach((hold) => {
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
    wallImage.src = URL.createObjectURL(file);
    wallImage.onload = () => renderHolds(demoHolds);
  });

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;

    wallImage.src = URL.createObjectURL(file);
    wallImage.onload = () => renderHolds(demoHolds);
  });
}

(async function init() {
  await requireSessionOrRedirect();
  setupWallImageUpload();
})();
