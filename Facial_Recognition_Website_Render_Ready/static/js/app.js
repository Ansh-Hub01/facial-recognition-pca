const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const fileBox = document.getElementById("selected-file");
const fileName = document.getElementById("file-name");
const button = document.getElementById("recognize-btn");
const empty = document.getElementById("empty-result");
const result = document.getElementById("recognition-result");
const errorBox = document.getElementById("error-message");
const status = document.getElementById("result-status");
let selectedFile = null;

function chooseFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Please upload a JPG, PNG or WEBP image.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showError("The image must be smaller than 8 MB.");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileBox.hidden = false;
  dropzone.style.display = "none";
  button.disabled = false;
  errorBox.hidden = true;
}
function resetFile() {
  selectedFile = null;
  input.value = "";
  fileBox.hidden = true;
  dropzone.style.display = "flex";
  button.disabled = true;
}
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  empty.hidden = true;
  result.hidden = true;
  status.textContent = "NEEDS ATTENTION";
}
input.addEventListener("change", () => chooseFile(input.files[0]));
document.getElementById("remove-file").addEventListener("click", resetFile);
["dragenter", "dragover"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", e => chooseFile(e.dataTransfer.files[0]));

form.addEventListener("submit", async e => {
  e.preventDefault();
  if (!selectedFile) return;
  const data = new FormData();
  data.append("image", selectedFile);
  button.disabled = true;
  button.textContent = "Analyzing image…";
  status.textContent = "PROCESSING";
  errorBox.hidden = true;
  try {
    const response = await fetch("/api/recognize", {method: "POST", body: data});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Recognition failed.");
    document.getElementById("preview-image").src = payload.preview;
    document.getElementById("predicted-id").textContent = `Person ${String(payload.identity).padStart(2, "0")}`;
    document.getElementById("confidence-note").textContent = payload.confidence_note;
    const nearest = document.getElementById("nearest-list");
    nearest.innerHTML = "";
    payload.nearest_matches.forEach((match, index) => {
      const row = document.createElement("div");
      row.className = "nearest-row";
      const label = document.createElement("span");
      label.textContent = `${index + 1}. Person ${String(match.identity).padStart(2, "0")}`;
      const distance = document.createElement("b");
      distance.textContent = `Distance ${match.distance.toFixed(3)}`;
      row.append(label, distance);
      nearest.appendChild(row);
    });
    empty.hidden = true;
    result.hidden = false;
    status.textContent = "COMPLETE";
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = !selectedFile;
    button.innerHTML = 'Recognize face <span>→</span>';
  }
});

fetch("/api/status")
  .then(r => r.json())
  .then(data => {
    if (!data.ready) throw new Error(data.error || "Model unavailable");
    document.getElementById("stat-images").textContent = data.images;
    document.getElementById("stat-people").textContent = data.identities;
  })
  .catch(() => {
    // Keep dataset facts visible; show an actionable message in the browser console.
    console.warn("Model status is not ready. Check Flask terminal and dataset download.");
  });
