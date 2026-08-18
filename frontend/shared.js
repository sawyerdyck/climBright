// Shared image + analysis persistence between Hold Buddy and Wall Analysis.
// Stores the last uploaded image AND its AI results in sessionStorage
// so switching pages doesn't lose work.

const IMAGE_STORAGE_KEY = "climbright-last-image";
const ANALYSIS_STORAGE_KEY = "climbright-analysis";

// --- Image persistence ---

function storeImage(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      sessionStorage.setItem(IMAGE_STORAGE_KEY, JSON.stringify({
        dataUrl: reader.result,
        name: file.name,
        type: file.type,
      }));
    } catch {
      // sessionStorage full or unavailable — non-fatal
    }
  };
  reader.readAsDataURL(file);
}

function getStoredImage() {
  try {
    const raw = sessionStorage.getItem(IMAGE_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function dataUrlToFile(dataUrl, name, type) {
  const [header, base64] = dataUrl.split(",");
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], name, { type: type || "image/jpeg" });
}

// --- Analysis results persistence ---

function storeAnalysis(page, data) {
  try {
    const existing = JSON.parse(sessionStorage.getItem(ANALYSIS_STORAGE_KEY) || "{}");
    existing[page] = data;
    sessionStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify(existing));
  } catch {
    // non-fatal
  }
}

function getStoredAnalysis(page) {
  try {
    const existing = JSON.parse(sessionStorage.getItem(ANALYSIS_STORAGE_KEY) || "{}");
    return existing[page] || null;
  } catch {
    return null;
  }
}

function clearAnalysis(page) {
  try {
    if (page) {
      const existing = JSON.parse(sessionStorage.getItem(ANALYSIS_STORAGE_KEY) || "{}");
      delete existing[page];
      sessionStorage.setItem(ANALYSIS_STORAGE_KEY, JSON.stringify(existing));
    } else {
      sessionStorage.removeItem(ANALYSIS_STORAGE_KEY);
    }
  } catch {
    // non-fatal
  }
}
