// Theme toggle: dark/light mode with localStorage persistence and system preference default.
(function () {
  const STORAGE_KEY = "climbright-theme";

  function getPreferred() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_KEY, theme);
    // Update toggle button icon
    const btn = document.getElementById("themeToggle");
    if (btn) btn.textContent = theme === "dark" ? "☀️" : "🌙";
  }

  // Apply immediately (before DOM renders to avoid flash)
  apply(getPreferred());

  // Bind toggle button once DOM is ready
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    btn.textContent = getPreferred() === "dark" ? "☀️" : "🌙";
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      document.documentElement.classList.add('theme-transitioning');
      apply(current === "dark" ? "light" : "dark");
      setTimeout(() => document.documentElement.classList.remove('theme-transitioning'), 500);
    });
  });
})();
