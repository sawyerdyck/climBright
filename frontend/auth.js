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

function setAuthError(msg) {
  const authError = document.getElementById("authError");
  if (!authError) return;
  if (!msg) {
    authError.hidden = true;
    authError.textContent = "";
    return;
  }
  authError.hidden = false;
  authError.textContent = msg;
}

function setAuthStatus(text) {
  const el = document.getElementById("authStatus");
  if (el) el.textContent = text;
}

async function refreshMe() {
  try {
    const data = await apiJson("/api/auth/me", { method: "GET" });
    setAuthStatus(`Signed in as ${data.user.email}`);
    // If already signed in, go to holds
    if (window.location.pathname === "/login" || window.location.pathname === "/register") {
      window.location.href = "/holds";
    }
  } catch {
    setAuthStatus("Not signed in");
  }
}

(async function init() {
  await refreshMe();

  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      setAuthError("");
      try {
        const email = document.getElementById("loginEmail").value;
        const password = document.getElementById("loginPassword").value;
        await apiJson("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
        window.location.href = "/holds";
      } catch (err) {
        setAuthError(err.message);
      }
    });
  }

  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      setAuthError("");
      try {
        const email = document.getElementById("registerEmail").value;
        const password = document.getElementById("registerPassword").value;
        await apiJson("/api/auth/register", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
        window.location.href = "/holds";
      } catch (err) {
        setAuthError(err.message);
      }
    });
  }
})();
