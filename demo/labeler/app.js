const CLASSES = [
  "USD_1",
  "USD_5",
  "USD_10",
  "USD_20",
  "USD_50",
  "USD_100",
  "KHR_500",
  "KHR_1000",
  "KHR_2000",
  "KHR_5000",
  "KHR_10000",
  "KHR_20000",
  "KHR_50000",
];

const COLORS = [
  "#e11d48",
  "#16a34a",
  "#ca8a04",
  "#0284c7",
  "#ea580c",
  "#9333ea",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#be123c",
  "#0f766e",
  "#7c3aed",
  "#92400e",
];

const state = {
  boxes: [],
  imagePath: "",
  naturalWidth: 0,
  naturalHeight: 0,
  selectedId: "",
  drawing: null,
};

const imagePath = document.getElementById("imagePath");
const loadButton = document.getElementById("loadButton");
const classSelect = document.getElementById("classSelect");
const importButton = document.getElementById("importButton");
const clearButton = document.getElementById("clearButton");
const undoButton = document.getElementById("undoButton");
const exportButton = document.getElementById("exportButton");
const stage = document.getElementById("stage");
const image = document.getElementById("image");
const overlay = document.getElementById("overlay");
const emptyState = document.getElementById("emptyState");
const status = document.getElementById("status");
const labelText = document.getElementById("labelText");
const boxList = document.getElementById("boxList");

for (const [index, name] of CLASSES.entries()) {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = name;
  classSelect.append(option);
}

function repoUrl(path) {
  return `/${path.replaceAll("\\", "/").replace(/^\/+/, "")}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function imageRect() {
  const rect = image.getBoundingClientRect();
  const stageRect = stage.getBoundingClientRect();
  return {
    left: rect.left - stageRect.left,
    top: rect.top - stageRect.top,
    width: rect.width,
    height: rect.height,
  };
}

function pointerToImage(event) {
  const rect = image.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
    y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
  };
}

function normalizeBox(box) {
  const x1 = clamp(Math.min(box.x1, box.x2), 0, 1);
  const y1 = clamp(Math.min(box.y1, box.y2), 0, 1);
  const x2 = clamp(Math.max(box.x1, box.x2), 0, 1);
  const y2 = clamp(Math.max(box.y1, box.y2), 0, 1);
  return { ...box, x1, y1, x2, y2 };
}

function yoloLine(box) {
  const item = normalizeBox(box);
  const cx = (item.x1 + item.x2) / 2;
  const cy = (item.y1 + item.y2) / 2;
  const width = item.x2 - item.x1;
  const height = item.y2 - item.y1;
  return [item.classId, cx, cy, width, height].map((value, index) => (index === 0 ? value : value.toFixed(6))).join(" ");
}

function syncLabelText() {
  labelText.value = state.boxes.map(yoloLine).join("\n");
}

function svgRect(box, extraClass = "") {
  const item = normalizeBox(box);
  const x = item.x1 * 100;
  const y = item.y1 * 100;
  const width = (item.x2 - item.x1) * 100;
  const height = (item.y2 - item.y1) * 100;
  const color = COLORS[item.classId % COLORS.length];
  const selected = item.id === state.selectedId;
  return `
    <g data-id="${item.id || ""}" class="${extraClass}">
      <rect x="${x}" y="${y}" width="${width}" height="${height}" fill="transparent" stroke="${color}" stroke-width="${selected ? 0.75 : 0.45}" vector-effect="non-scaling-stroke" />
      <text x="${x}" y="${Math.max(2.5, y - 1)}" fill="${color}" font-size="2.8" font-weight="700">${CLASSES[item.classId] || item.classId}</text>
    </g>
  `;
}

function renderOverlay() {
  const rect = imageRect();
  overlay.style.left = `${rect.left}px`;
  overlay.style.top = `${rect.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.setAttribute("viewBox", "0 0 100 100");
  overlay.innerHTML = state.boxes.map((box) => svgRect(box)).join("") + (state.drawing ? svgRect(state.drawing, "preview") : "");
  overlay.querySelectorAll("g[data-id]").forEach((node) => {
    node.addEventListener("pointerdown", (event) => {
      const id = node.getAttribute("data-id");
      if (id) {
        event.stopPropagation();
        state.selectedId = id;
        render();
      }
    });
  });
}

function renderList() {
  boxList.innerHTML = "";
  for (const box of state.boxes) {
    const row = document.createElement("div");
    row.className = `box-row${box.id === state.selectedId ? " is-selected" : ""}`;
    row.innerHTML = `
      <div>
        <select aria-label="Box class">
          ${CLASSES.map((name, index) => `<option value="${index}" ${index === box.classId ? "selected" : ""}>${name}</option>`).join("")}
        </select>
        <small>${yoloLine(box)}</small>
      </div>
      <button class="delete" type="button">Delete</button>
    `;
    row.addEventListener("click", () => {
      state.selectedId = box.id;
      render();
    });
    row.querySelector("select").addEventListener("change", (event) => {
      box.classId = Number(event.target.value);
      syncLabelText();
      renderOverlay();
    });
    row.querySelector(".delete").addEventListener("click", (event) => {
      event.stopPropagation();
      state.boxes = state.boxes.filter((item) => item.id !== box.id);
      if (state.selectedId === box.id) {
        state.selectedId = "";
      }
      render();
    });
    boxList.append(row);
  }
}

function render() {
  const loaded = Boolean(state.naturalWidth);
  clearButton.disabled = state.boxes.length === 0;
  undoButton.disabled = state.boxes.length === 0;
  exportButton.disabled = !loaded;
  status.textContent = loaded
    ? `${state.naturalWidth}x${state.naturalHeight}, ${state.boxes.length} boxes`
    : "No image loaded";
  renderOverlay();
  renderList();
  syncLabelText();
}

function loadImage() {
  state.imagePath = imagePath.value.trim();
  if (!state.imagePath) {
    status.textContent = "Image path is required";
    return;
  }
  image.onload = () => {
    state.naturalWidth = image.naturalWidth;
    state.naturalHeight = image.naturalHeight;
    image.style.display = "block";
    overlay.style.display = "block";
    emptyState.style.display = "none";
    state.boxes = [];
    state.selectedId = "";
    render();
  };
  image.onerror = () => {
    status.textContent = `Load failed: ${state.imagePath}`;
  };
  image.src = repoUrl(state.imagePath);
}

function importLabels() {
  if (!state.naturalWidth) {
    status.textContent = "Load an image before importing labels";
    return;
  }
  const boxes = [];
  for (const [lineIndex, rawLine] of labelText.value.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const parts = line.split(/\s+/).map(Number);
    if (parts.length !== 5 || parts.some((value) => Number.isNaN(value))) {
      status.textContent = `Bad label line ${lineIndex + 1}`;
      return;
    }
    const [classId, cx, cy, width, height] = parts;
    boxes.push(
      normalizeBox({
        id: crypto.randomUUID(),
        classId,
        x1: cx - width / 2,
        y1: cy - height / 2,
        x2: cx + width / 2,
        y2: cy + height / 2,
      }),
    );
  }
  state.boxes = boxes;
  state.selectedId = boxes.at(-1)?.id || "";
  render();
}

function exportLabels() {
  const blob = new Blob([state.boxes.map(yoloLine).join("\n") + "\n"], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  const stem = (state.imagePath.split(/[\\/]/).pop() || "labels").replace(/\.[^.]+$/, "");
  link.href = URL.createObjectURL(blob);
  link.download = `${stem}.txt`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function startDraw(event) {
  if (!state.naturalWidth || event.button !== 0) {
    return;
  }
  const point = pointerToImage(event);
  state.drawing = {
    id: "preview",
    classId: Number(classSelect.value),
    x1: point.x,
    y1: point.y,
    x2: point.x,
    y2: point.y,
  };
  overlay.setPointerCapture(event.pointerId);
  renderOverlay();
}

function moveDraw(event) {
  if (!state.drawing) {
    return;
  }
  const point = pointerToImage(event);
  state.drawing.x2 = point.x;
  state.drawing.y2 = point.y;
  renderOverlay();
}

function endDraw(event) {
  if (!state.drawing) {
    return;
  }
  const item = normalizeBox({ ...state.drawing, id: crypto.randomUUID() });
  state.drawing = null;
  overlay.releasePointerCapture(event.pointerId);
  if ((item.x2 - item.x1) * image.clientWidth >= 6 && (item.y2 - item.y1) * image.clientHeight >= 6) {
    state.boxes.push(item);
    state.selectedId = item.id;
  }
  render();
}

loadButton.addEventListener("click", loadImage);
importButton.addEventListener("click", importLabels);
exportButton.addEventListener("click", exportLabels);
clearButton.addEventListener("click", () => {
  state.boxes = [];
  state.selectedId = "";
  render();
});
undoButton.addEventListener("click", () => {
  const removed = state.boxes.pop();
  if (removed?.id === state.selectedId) {
    state.selectedId = state.boxes.at(-1)?.id || "";
  }
  render();
});
overlay.addEventListener("pointerdown", startDraw);
overlay.addEventListener("pointermove", moveDraw);
overlay.addEventListener("pointerup", endDraw);
overlay.addEventListener("pointercancel", () => {
  state.drawing = null;
  renderOverlay();
});
window.addEventListener("resize", renderOverlay);

const params = new URLSearchParams(window.location.search);
const requestedImage = params.get("image");
if (requestedImage) {
  imagePath.value = requestedImage;
}
if (requestedImage || params.get("autoload") === "1") {
  loadImage();
}
