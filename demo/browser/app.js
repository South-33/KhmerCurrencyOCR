const STACK_CONFIG_URL = "/configs/cashsnap_two_stage_oldcommon_browser_stack.json";
const DEFAULT_CLASS_NAMES = [
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
const VALUES = {
  KHR_500: 500,
  KHR_1000: 1000,
  KHR_2000: 2000,
  KHR_5000: 5000,
  KHR_10000: 10000,
  KHR_20000: 20000,
  KHR_50000: 50000,
};
const COLORS = [
  "#e43d30",
  "#2478c2",
  "#f0a202",
  "#1b998b",
  "#9b5de5",
  "#ef476f",
  "#06d6a0",
  "#ffd166",
  "#118ab2",
  "#f78c6b",
  "#00b4d8",
  "#f72585",
  "#90be6d",
];

const state = {
  config: null,
  detectorSession: null,
  classifierSession: null,
  image: null,
  detections: [],
  conf: 0.05,
};

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
const imageInput = document.getElementById("imageInput");
const runButton = document.getElementById("runButton");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const modelStatus = document.getElementById("modelStatus");
const emptyState = document.getElementById("emptyState");
const totalCount = document.getElementById("totalCount");
const totalValue = document.getElementById("totalValue");
const countsList = document.getElementById("countsList");

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const configResponse = await fetch(STACK_CONFIG_URL);
    if (!configResponse.ok) {
      throw new Error(`Config HTTP ${configResponse.status}`);
    }
    state.config = await configResponse.json();
    state.conf = state.config.detector.proposal_confidence ?? state.conf;
    confSlider.value = String(state.conf);
    confValue.textContent = state.conf.toFixed(2);
    const options = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };
    [state.detectorSession, state.classifierSession] = await Promise.all([
      ort.InferenceSession.create(repoUrl(state.config.detector.path), options),
      ort.InferenceSession.create(repoUrl(state.config.fragment_classifier.path), options),
    ]);
    modelStatus.textContent = "Models ready";
    modelStatus.className = "status ready";
    updateRunState();
  } catch (error) {
    modelStatus.textContent = "Model load failed";
    modelStatus.className = "status error";
    console.error(error);
  }
}

function updateRunState() {
  runButton.disabled = !state.detectorSession || !state.classifierSession || !state.image;
}

function repoUrl(path) {
  return `/${path.replaceAll("\\", "/").replace(/^\/+/, "")}`;
}

function drawBaseImage() {
  if (!state.image) {
    return;
  }
  const maxSide = 1800;
  const scale = Math.min(1, maxSide / Math.max(state.image.width, state.image.height));
  canvas.width = Math.round(state.image.width * scale);
  canvas.height = Math.round(state.image.height * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
}

function preprocess(image) {
  const inputSize = state.config?.detector?.input_size ?? 416;
  const scale = Math.min(inputSize / image.width, inputSize / image.height);
  const resizedWidth = Math.round(image.width * scale);
  const resizedHeight = Math.round(image.height * scale);
  const padX = Math.floor((inputSize - resizedWidth) / 2);
  const padY = Math.floor((inputSize - resizedHeight) / 2);

  const prepCanvas = document.createElement("canvas");
  prepCanvas.width = inputSize;
  prepCanvas.height = inputSize;
  const prepCtx = prepCanvas.getContext("2d");
  prepCtx.fillStyle = "rgb(114,114,114)";
  prepCtx.fillRect(0, 0, inputSize, inputSize);
  prepCtx.drawImage(image, padX, padY, resizedWidth, resizedHeight);

  const pixels = prepCtx.getImageData(0, 0, inputSize, inputSize).data;
  const input = new Float32Array(3 * inputSize * inputSize);
  for (let i = 0; i < inputSize * inputSize; i += 1) {
    input[i] = pixels[i * 4] / 255;
    input[i + inputSize * inputSize] = pixels[i * 4 + 1] / 255;
    input[i + inputSize * inputSize * 2] = pixels[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]),
    scale,
    padX,
    padY,
  };
}

function parseOutput(output, meta) {
  const detections = [];
  for (let i = 0; i < output.dims[1]; i += 1) {
    const offset = i * 6;
    const score = output.data[offset + 4];
    if (score < state.conf) {
      continue;
    }
    const classId = Math.round(output.data[offset + 5]);
    const classNames = state.config?.detector?.classes ?? DEFAULT_CLASS_NAMES;
    const x1 = (output.data[offset] - meta.padX) / meta.scale;
    const y1 = (output.data[offset + 1] - meta.padY) / meta.scale;
    const x2 = (output.data[offset + 2] - meta.padX) / meta.scale;
    const y2 = (output.data[offset + 3] - meta.padY) / meta.scale;
    detections.push({
      classId,
      detectorName: classNames[classId] || `class_${classId}`,
      detectorScore: score,
      fragmentName: "",
      fragmentScore: 0,
      name: classNames[classId] || `class_${classId}`,
      score,
      x1: clamp(x1, 0, state.image.width),
      y1: clamp(y1, 0, state.image.height),
      x2: clamp(x2, 0, state.image.width),
      y2: clamp(y2, 0, state.image.height),
    });
  }
  return detections;
}

function preprocessClassifierBatch(image, detections) {
  const size = state.config?.fragment_classifier?.input_size ?? 224;
  const area = size * size;
  const input = new Float32Array(detections.length * 3 * area);
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = size;
  cropCanvas.height = size;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
  const mean = state.config?.fragment_classifier?.normalization?.mean ?? [0.485, 0.456, 0.406];
  const std = state.config?.fragment_classifier?.normalization?.std ?? [0.229, 0.224, 0.225];

  detections.forEach((detection, batchIndex) => {
    const boxWidth = detection.x2 - detection.x1;
    const boxHeight = detection.y2 - detection.y1;
    const padX = boxWidth * 0.08;
    const padY = boxHeight * 0.08;
    const sx = clamp(detection.x1 - padX, 0, image.width);
    const sy = clamp(detection.y1 - padY, 0, image.height);
    const ex = clamp(detection.x2 + padX, 0, image.width);
    const ey = clamp(detection.y2 + padY, 0, image.height);

    cropCtx.fillStyle = "white";
    cropCtx.fillRect(0, 0, size, size);
    cropCtx.drawImage(image, sx, sy, Math.max(1, ex - sx), Math.max(1, ey - sy), 0, 0, size, size);
    const pixels = cropCtx.getImageData(0, 0, size, size).data;
    const batchOffset = batchIndex * 3 * area;
    for (let i = 0; i < area; i += 1) {
      input[batchOffset + i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
      input[batchOffset + area + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
      input[batchOffset + area * 2 + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
  });

  return new ort.Tensor("float32", input, [detections.length, 3, size, size]);
}

function softmaxRow(data, offset, count) {
  let maxLogit = -Infinity;
  for (let i = 0; i < count; i += 1) {
    maxLogit = Math.max(maxLogit, data[offset + i]);
  }
  let sum = 0;
  const probs = [];
  for (let i = 0; i < count; i += 1) {
    const value = Math.exp(data[offset + i] - maxLogit);
    probs.push(value);
    sum += value;
  }
  return probs.map((value) => value / sum);
}

async function classifyFragments(detections) {
  if (!detections.length) {
    return detections;
  }
  const tensor = preprocessClassifierBatch(state.image, detections);
  const feeds = { [state.classifierSession.inputNames[0]]: tensor };
  const outputs = await state.classifierSession.run(feeds);
  const logits = outputs[state.classifierSession.outputNames[0]];
  const fragmentClassNames = state.config?.fragment_classifier?.classes ?? ["KHR_1000", "KHR_10000", "KHR_20000", "KHR_5000"];
  const classCount = fragmentClassNames.length;
  return detections.map((detection, index) => {
    const probs = softmaxRow(logits.data, index * classCount, classCount);
    let bestIndex = 0;
    for (let i = 1; i < probs.length; i += 1) {
      if (probs[i] > probs[bestIndex]) {
        bestIndex = i;
      }
    }
    const fragmentName = fragmentClassNames[bestIndex];
    const fragmentScore = probs[bestIndex];
    const overrideConf = state.config?.fusion?.detector_override_confidence ?? 0.17;
    const useDetector = detection.detectorScore >= overrideConf;
    return {
      ...detection,
      fragmentName,
      fragmentScore,
      name: useDetector ? detection.detectorName : fragmentName,
      score: useDetector ? detection.detectorScore : fragmentScore,
    };
  });
}

function boxIou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union ? intersection / union : 0;
}

function nms(detections) {
  const nmsIou = state.config?.fusion?.nms_iou ?? 0.85;
  const pending = [...detections].sort((a, b) => b.detectorScore - a.detectorScore);
  const kept = [];
  for (const detection of pending) {
    if (kept.every((keptDetection) => boxIou(detection, keptDetection) < nmsIou)) {
      kept.push(detection);
    }
  }
  return kept.sort((a, b) => b.score - a.score);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function renderDetections() {
  drawBaseImage();
  const sx = canvas.width / state.image.width;
  const sy = canvas.height / state.image.height;
  ctx.lineWidth = Math.max(3, Math.round(canvas.width / 420));
  ctx.font = `${Math.max(18, Math.round(canvas.width / 42))}px Aptos, Segoe UI, sans-serif`;
  ctx.textBaseline = "top";

  for (const detection of state.detections) {
    const color = COLORS[detection.classId % COLORS.length];
    const x = detection.x1 * sx;
    const y = detection.y1 * sy;
    const w = (detection.x2 - detection.x1) * sx;
    const h = (detection.y2 - detection.y1) * sy;
    const label = `${detection.name} ${detection.score.toFixed(2)}`;
    const textWidth = ctx.measureText(label).width + 12;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(x, y, w, h);
    ctx.fillRect(x, Math.max(0, y - 34), textWidth, 34);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 6, Math.max(0, y - 30));
  }
  renderSummary();
}

function renderSummary() {
  const counts = new Map();
  let value = 0;
  for (const detection of state.detections) {
    counts.set(detection.name, (counts.get(detection.name) || 0) + 1);
    value += VALUES[detection.name] || 0;
  }
  totalCount.textContent = String(state.detections.length);
  totalValue.textContent = `KHR ${value.toLocaleString()}`;
  countsList.innerHTML = "";
  for (const [name, count] of [...counts.entries()].sort()) {
    const classNames = state.config?.detector?.classes ?? DEFAULT_CLASS_NAMES;
    const classId = classNames.indexOf(name);
    const row = document.createElement("div");
    row.className = "count-row";
    row.innerHTML = `<span><i class="swatch" style="background:${COLORS[classId % COLORS.length]}"></i>${name}</span><strong>${count}</strong>`;
    countsList.append(row);
  }
}

async function runModel() {
  if (!state.detectorSession || !state.classifierSession || !state.image) {
    return;
  }
  runButton.disabled = true;
  runButton.textContent = "Running";
  const meta = preprocess(state.image);
  const feeds = { [state.detectorSession.inputNames[0]]: meta.tensor };
  const outputs = await state.detectorSession.run(feeds);
  const output = outputs[state.detectorSession.outputNames[0]];
  const proposals = parseOutput(output, meta);
  state.detections = nms(await classifyFragments(proposals));
  renderDetections();
  runButton.textContent = "Run";
  updateRunState();
}

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(url);
    state.image = image;
    state.detections = [];
    emptyState.classList.add("hidden");
    drawBaseImage();
    renderSummary();
    updateRunState();
  };
  image.src = url;
});

confSlider.addEventListener("input", () => {
  state.conf = Number(confSlider.value);
  confValue.textContent = state.conf.toFixed(2);
});

runButton.addEventListener("click", runModel);

loadModel();
