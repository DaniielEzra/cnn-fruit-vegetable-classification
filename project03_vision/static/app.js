// Fruit & Vegetable Classifier: browser logic.
// 1. Let the user choose / drop / photograph an image and show a preview.
// 2. Send it to POST /predict when "Classify Image" is clicked.
// 3. Show the predicted class and a progress bar for every class.

const MAX_BYTES = 10 * 1024 * 1024; // must match MAX_UPLOAD_BYTES in app.py
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "gif", "tif", "tiff"];
const REQUEST_TIMEOUT_MS = 60000;

const el = (id) => document.getElementById(id);
const dropZone = el("drop-zone");
const preview = el("preview");
const dropEmpty = el("drop-empty");
const fileNameText = el("file-name");
const fileInput = el("file-input");
const cameraInput = el("camera-input");
const chooseBtn = el("choose-btn");
const cameraBtn = el("camera-btn");
const classifyBtn = el("classify-btn");
const downloadBtn = el("download-btn");
const resetBtn = el("reset-btn");
const errorBox = el("error");
const resultEmpty = el("result-empty");
const loading = el("loading");
const resultBox = el("result");
const sticker = el("sticker");
const predictionText = el("prediction");
const confidenceText = el("confidence");
const resultMessage = el("result-message");
const scoresList = el("scores");
const serverStatus = el("server-status");
const warningBox = el("warning");
const thresholdNote = el("threshold-note");

let selectedFile = null;
let previewUrl = null;
let modelReady = false;

// ---------- Server / model status ----------
async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    modelReady = Boolean(data.model_loaded);
    // The threshold comes from model_info.json (written by the notebook).
    if (typeof data.confidence_threshold === "number") {
      thresholdNote.textContent = `${Math.round(data.confidence_threshold * 100)}%`;
    }
    warningBox.textContent = data.warning || "";
    warningBox.hidden = !data.warning;
    if (modelReady) {
      serverStatus.textContent = "Model ready";
      serverStatus.className = "status status--ready";
    } else {
      serverStatus.textContent = "Model not loaded";
      serverStatus.className = "status status--error";
      showError(data.message || "The model is not loaded on the server.");
    }
  } catch {
    modelReady = false;
    serverStatus.textContent = "Server offline";
    serverStatus.className = "status status--error";
    showError("Cannot reach the server. Check that uvicorn is running.");
  }
  updateButtons();
}

// ---------- Helpers ----------
function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.round(bytes / 1024)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateButtons(isBusy = false) {
  const hasFile = selectedFile !== null;
  classifyBtn.disabled = !hasFile || isBusy;
  downloadBtn.disabled = !hasFile || isBusy;
  chooseBtn.disabled = isBusy;
  cameraBtn.disabled = isBusy;
  resetBtn.hidden = !hasFile;
  resetBtn.disabled = isBusy;
}

function showResultArea(which) {
  resultEmpty.hidden = which !== "empty";
  loading.hidden = which !== "loading";
  resultBox.hidden = which !== "result";
}

// ---------- Choosing an image ----------
function validateFile(file) {
  const extension = (file.name.split(".").pop() || "").toLowerCase();
  const looksLikeImage = file.type.startsWith("image/") || ALLOWED_EXTENSIONS.includes(extension);
  if (!looksLikeImage) return "This file is not an image. Choose a JPEG, PNG, WEBP, BMP or GIF file.";
  if (file.size === 0) return "This file is empty. Choose another image.";
  if (file.size > MAX_BYTES) return `This image is ${formatSize(file.size)}. The limit is 10 MB.`;
  return null;
}

function selectFile(file) {
  if (!file) return;
  clearError();
  const problem = validateFile(file);
  if (problem) {
    showError(problem);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);

  // Browsers apply EXIF rotation to <img> automatically, the server does the same.
  preview.onerror = () => {
    preview.hidden = true;
    dropEmpty.hidden = false;
    showError("This browser cannot preview the image, but you can still try to classify it.");
  };
  preview.src = previewUrl;
  preview.hidden = false;
  dropEmpty.hidden = true;
  dropZone.classList.add("has-image");

  fileNameText.textContent = `${file.name || "photo"} (${formatSize(file.size)})`;
  fileNameText.hidden = false;

  showResultArea("empty");
  updateButtons();
}

function resetAll() {
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  preview.hidden = true;
  dropEmpty.hidden = false;
  dropZone.classList.remove("has-image");
  fileNameText.hidden = true;
  fileInput.value = "";
  cameraInput.value = "";
  clearError();
  showResultArea("empty");
  updateButtons();
  chooseBtn.focus();
}

chooseBtn.addEventListener("click", () => fileInput.click());
// On phones this opens the camera. On computers it opens the normal file picker.
cameraBtn.addEventListener("click", () => cameraInput.click());
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
cameraInput.addEventListener("change", () => selectFile(cameraInput.files[0]));
resetBtn.addEventListener("click", resetAll);

dropZone.addEventListener("click", () => { if (!selectedFile) fileInput.click(); });
dropZone.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && !selectedFile) {
    event.preventDefault();
    fileInput.click();
  }
});
["dragenter", "dragover"].forEach((type) =>
  dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  })
);
["dragleave", "drop"].forEach((type) =>
  dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  })
);
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

// ---------- Download the selected image ----------
downloadBtn.addEventListener("click", () => {
  if (!selectedFile) return;
  const link = document.createElement("a");
  link.href = previewUrl;
  link.download = selectedFile.name || "image.jpg";
  document.body.appendChild(link);
  link.click();
  link.remove();
});

// ---------- Classification ----------
classifyBtn.addEventListener("click", classify);

async function classify() {
  if (!selectedFile) return;
  clearError();
  showResultArea("loading");
  updateButtons(true);

  const formData = new FormData();
  formData.append("file", selectedFile, selectedFile.name || "image.jpg");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData, signal: controller.signal });
    let data = null;
    try { data = await response.json(); } catch { /* not JSON */ }

    if (!response.ok) {
      const message = data && data.error ? data.error.message : `The server answered with error ${response.status}.`;
      throw new Error(message);
    }
    showPrediction(data);
  } catch (error) {
    showResultArea("empty");
    if (error.name === "AbortError") {
      showError("The server took too long to answer. Try again.");
    } else if (error instanceof TypeError) {
      showError("Cannot reach the server. Check that uvicorn is running and try again.");
    } else {
      showError(error.message);
    }
  } finally {
    clearTimeout(timer);
    updateButtons(false);
  }
}

function showPrediction(data) {
  const unknown = data.is_unknown;
  predictionText.textContent = unknown ? "Unknown" : data.prediction;
  confidenceText.textContent = unknown
    ? `Best guess ${data.top_class}, ${(data.confidence * 100).toFixed(1)}%`
    : `${(data.confidence * 100).toFixed(1)}% confident`;
  sticker.classList.toggle("is-unknown", unknown);
  resultMessage.textContent = unknown
    ? `No class reached ${Math.round(data.threshold * 100)}%, so the answer is unknown. The image may not show one of the five classes, or the photo may be unclear.`
    : data.message;

  // Bars sorted from highest to lowest.
  const scores = [...data.scores].sort((a, b) => b.probability - a.probability);
  scoresList.innerHTML = "";
  scores.forEach((score, index) => {
    const percent = score.probability * 100;
    const row = document.createElement("li");
    row.className = "score-row" + (index === 0 ? " is-top" : "");
    row.innerHTML = `
      <span class="score-label"></span>
      <div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100">
        <div class="bar-fill"></div>
      </div>
      <span class="score-value"></span>`;
    row.querySelector(".score-label").textContent = score.class_name;
    row.querySelector(".score-value").textContent = `${percent.toFixed(1)}%`;
    const bar = row.querySelector(".bar");
    bar.setAttribute("aria-valuenow", percent.toFixed(1));
    bar.setAttribute("aria-label", `${score.class_name} confidence`);
    scoresList.appendChild(row);
    // Animate after the row is on the page.
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${Math.max(percent, 0.5)}%`;
    });
  });

  // Restart the sticker animation.
  sticker.style.animation = "none";
  void sticker.offsetWidth;
  sticker.style.animation = "";

  showResultArea("result");

  // On phones the result is below the buttons: bring it into view.
  if (window.matchMedia("(max-width: 760px)").matches) {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el("result-heading").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
}

checkHealth();
updateButtons();
