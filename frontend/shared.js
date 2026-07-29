// Shared image persistence between Hold Buddy and Wall Analysis.
// Stores the last uploaded image in sessionStorage so it survives page navigation.

const IMAGE_STORAGE_KEY = "climbright-last-image";

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
