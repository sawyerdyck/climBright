function setupUpload(boxId, previewId) {
  const box = document.getElementById(boxId);
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

  box.addEventListener("drop", e => {
    e.preventDefault();
    input.files = e.dataTransfer.files;
    handlePreview(input, preview);
    box.style.borderColor = "#30363d";
  });

  input.addEventListener("change", () => handlePreview(input, preview));
}

function handlePreview(input, preview) {
  preview.innerHTML = "";
  const file = input.files[0];
  if (!file) return;

  const img = document.createElement("img");
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
}

// Init uploads
setupUpload("holdUpload", "holdPreview");
setupUpload("wallUpload", "wallPreview");


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
  holdInfoText.innerHTML = `
    <strong>Type:</strong> ${hold.type}<br/>
    <strong>Difficulty:</strong> ${hold.difficulty}<br/>
    <strong>Confidence:</strong> ${(hold.confidence * 100).toFixed(1)}%
  `;
}

// Hook into existing upload logic
function setupWallImageUpload() {
  const box = document.getElementById("wallUpload");
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
